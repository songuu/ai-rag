/**
 * Agentic RAG - 基于 LangGraph 的代理化检索增强生成系统
 * 
 * 核心功能：
 * 1. 输入优化：查询预处理与改写
 * 2. 检索判断：智能决策是否需要检索
 * 3. 质量评估：自动评分与重试机制
 * 4. 幻觉检查：事实验证与一致性检查
 * 5. 自省模式：检索结果打分
 * 6. 透传原始问题：同时使用原始查询和优化查询检索
 * 7. 检索评估节点：快速判断检索质量
 * 8. LangSmith 追踪：详细调试和可观测性
 */

import { StateGraph, Annotation, END, START } from '@langchain/langgraph';
import { Ollama, OllamaEmbeddings } from '@langchain/ollama';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { getMilvusInstance, MilvusConfig, getModelDimension, selectModelByDimension } from './milvus-client';

// LangSmith 追踪配置
const LANGSMITH_ENABLED = process.env.LANGCHAIN_TRACING_V2 === 'true';
const LANGSMITH_PROJECT = process.env.LANGCHAIN_PROJECT || 'agentic-rag';

// 追踪日志
function trace(step: string, data: any) {
  const timestamp = new Date().toISOString();
  console.log(`[LangSmith Trace] [${timestamp}] [${step}]`, JSON.stringify(data, null, 2));
  
  // 如果启用了 LangSmith，这里的日志会被 LangChain 自动收集
  if (LANGSMITH_ENABLED) {
    console.log(`[LangSmith] Project: ${LANGSMITH_PROJECT}, Step: ${step}`);
  }
}

// ==================== 类型定义 ====================

/** 检索结果 */
export interface RetrievedDocument {
  content: string;
  metadata: Record<string, any>;
  score: number;
  relevanceScore?: number; // 自省评分
  factualScore?: number;   // 事实性评分
}

/** 查询分析结果 */
export interface QueryAnalysis {
  originalQuery: string;
  rewrittenQuery: string;
  intent: 'factual' | 'exploratory' | 'comparison' | 'procedural' | 'unknown';
  complexity: 'simple' | 'moderate' | 'complex';
  needsRetrieval: boolean;
  keywords: string[];
  confidence: number;
}

/** 检索质量评估 */
export interface RetrievalQuality {
  overallScore: number;       // 0-1 总体评分
  relevanceScore: number;     // 相关性评分
  coverageScore: number;      // 覆盖度评分
  diversityScore: number;     // 多样性评分
  isAcceptable: boolean;      // 是否可接受
  suggestions: string[];      // 改进建议
}

/** 幻觉检查结果 */
export interface HallucinationCheck {
  hasHallucination: boolean;
  confidence: number;
  problematicClaims: string[];
  supportedClaims: string[];
  overallFactualScore: number;
}

/** 自省评分结果 */
export interface SelfReflectionScore {
  documentScores: Array<{
    index: number;
    relevance: number;
    usefulness: number;
    factuality: number;
    overall: number;
    reasoning: string;
  }>;
  queryAlignmentScore: number;
  contextCompleteness: number;
  recommendation: 'use' | 'expand' | 'rewrite' | 'skip';
}

/** 工作流步骤 */
export interface WorkflowStep {
  step: string;
  status: 'pending' | 'running' | 'completed' | 'skipped' | 'error';
  startTime?: number;
  endTime?: number;
  duration?: number;
  input?: any;
  output?: any;
  error?: string;
}

/** 检索评分结果 */
export interface RetrievalGradeResult {
  isRelevant: boolean;
  score: number;              // 0-1 综合评分
  keywordMatchScore: number;  // 关键词匹配分数
  semanticScore: number;      // 语义相关性分数
  hasAnswerSignals: boolean;  // 是否包含答案信号
  reasoning: string;          // 评分理由
  documentGrades: Array<{
    index: number;
    isRelevant: boolean;
    score: number;
    matchedKeywords: string[];
    reasoning: string;
  }>;
}

/** Agent 工作流状态 */
export interface AgentState {
  // 输入 - 透传原始问题
  query: string;                    // 原始用户查询（始终保留）
  originalQuery: string;            // 原始查询副本（永不修改）
  processedQuery: string;           // 处理后的查询（可能被改写）
  topK: number;
  similarityThreshold: number;
  maxRetries: number;
  
  // 分析结果
  queryAnalysis?: QueryAnalysis;
  
  // 检索相关 - 支持双查询检索
  retrievedDocuments: RetrievedDocument[];
  originalQueryResults: RetrievedDocument[];   // 原始查询的检索结果
  processedQueryResults: RetrievedDocument[];  // 处理后查询的检索结果
  retrievalQuality?: RetrievalQuality;
  selfReflection?: SelfReflectionScore;
  retrievalGrade?: RetrievalGradeResult;       // 检索评分结果
  
  // 生成相关
  context: string;
  answer: string;
  
  // 幻觉检查
  hallucinationCheck?: HallucinationCheck;
  
  // 流程控制
  currentStep: string;
  retryCount: number;
  shouldRewrite: boolean;
  shouldRetrieve: boolean;
  gradePassThreshold: number;  // 检索评分通过阈值
  
  // 工作流追踪
  workflowSteps: WorkflowStep[];
  
  // 元数据
  startTime: number;
  endTime?: number;
  totalDuration?: number;
  error?: string;
  
  // 调试信息
  debugInfo?: {
    milvusQueryVector?: number[];
    milvusRawScores?: number[];
    embeddingModel?: string;
    collectionDimension?: number;
  };
}

// ==================== 状态图定义 ====================

const AgentStateAnnotation = Annotation.Root({
  // 输入 - 透传原始问题
  query: Annotation<string>({ reducer: (_, b) => b, default: () => '' }),
  originalQuery: Annotation<string>({ reducer: (_, b) => b, default: () => '' }),
  processedQuery: Annotation<string>({ reducer: (_, b) => b, default: () => '' }),
  topK: Annotation<number>({ reducer: (_, b) => b, default: () => 5 }),
  similarityThreshold: Annotation<number>({ reducer: (_, b) => b, default: () => 0.3 }),
  maxRetries: Annotation<number>({ reducer: (_, b) => b, default: () => 2 }),
  
  queryAnalysis: Annotation<QueryAnalysis | undefined>({ reducer: (_, b) => b }),
  
  // 检索相关 - 支持双查询检索
  retrievedDocuments: Annotation<RetrievedDocument[]>({ reducer: (_, b) => b, default: () => [] }),
  originalQueryResults: Annotation<RetrievedDocument[]>({ reducer: (_, b) => b, default: () => [] }),
  processedQueryResults: Annotation<RetrievedDocument[]>({ reducer: (_, b) => b, default: () => [] }),
  retrievalQuality: Annotation<RetrievalQuality | undefined>({ reducer: (_, b) => b }),
  selfReflection: Annotation<SelfReflectionScore | undefined>({ reducer: (_, b) => b }),
  retrievalGrade: Annotation<RetrievalGradeResult | undefined>({ reducer: (_, b) => b }),
  
  context: Annotation<string>({ reducer: (_, b) => b, default: () => '' }),
  answer: Annotation<string>({ reducer: (_, b) => b, default: () => '' }),
  
  hallucinationCheck: Annotation<HallucinationCheck | undefined>({ reducer: (_, b) => b }),
  
  currentStep: Annotation<string>({ reducer: (_, b) => b, default: () => 'start' }),
  retryCount: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  shouldRewrite: Annotation<boolean>({ reducer: (_, b) => b, default: () => false }),
  shouldRetrieve: Annotation<boolean>({ reducer: (_, b) => b, default: () => true }),
  gradePassThreshold: Annotation<number>({ reducer: (_, b) => b, default: () => 0.5 }),
  
  workflowSteps: Annotation<WorkflowStep[]>({ 
    reducer: (a, b) => [...a, ...b], 
    default: () => [] 
  }),
  
  startTime: Annotation<number>({ reducer: (_, b) => b, default: () => Date.now() }),
  endTime: Annotation<number | undefined>({ reducer: (_, b) => b }),
  totalDuration: Annotation<number | undefined>({ reducer: (_, b) => b }),
  error: Annotation<string | undefined>({ reducer: (_, b) => b }),
  
  debugInfo: Annotation<AgentState['debugInfo'] | undefined>({ reducer: (_, b) => b }),
});

// ==================== Agentic RAG 系统类 ====================

export interface AgenticRAGConfig {
  ollamaBaseUrl?: string;
  llmModel?: string;
  embeddingModel?: string;
  milvusConfig?: Partial<MilvusConfig>;
  enableHallucinationCheck?: boolean;
  enableSelfReflection?: boolean;
  onStepUpdate?: (step: WorkflowStep) => void;
}

export class AgenticRAGSystem {
  private llm: Ollama;
  private embeddings: OllamaEmbeddings;
  private milvusConfig: MilvusConfig;
  private config: AgenticRAGConfig;
  private graph: any; // StateGraph 实例
  private ollamaBaseUrl: string;
  private requestedEmbeddingModel: string;

  constructor(config: AgenticRAGConfig = {}) {
    const {
      ollamaBaseUrl = 'http://localhost:11434',
      llmModel = 'llama3.1',
      embeddingModel = 'nomic-embed-text',
      milvusConfig = {},
      enableHallucinationCheck = true,
      enableSelfReflection = true,
    } = config;

    this.config = {
      ...config,
      enableHallucinationCheck,
      enableSelfReflection,
    };

    this.ollamaBaseUrl = ollamaBaseUrl;
    this.requestedEmbeddingModel = embeddingModel;

    this.llm = new Ollama({
      baseUrl: ollamaBaseUrl,
      model: llmModel,
      temperature: 0,
    });

    this.embeddings = new OllamaEmbeddings({
      baseUrl: ollamaBaseUrl,
      model: embeddingModel,
    });

    this.milvusConfig = {
      address: milvusConfig.address || 'localhost:19530',
      collectionName: milvusConfig.collectionName || 'rag_documents',
      embeddingDimension: milvusConfig.embeddingDimension || 768,
      indexType: milvusConfig.indexType || 'IVF_FLAT',
      metricType: milvusConfig.metricType || 'COSINE',
    };

    this.graph = this.buildGraph();
  }

  /**
   * 根据 Milvus 集合维度选择合适的 embedding 模型
   */
  private async getMatchingEmbeddings(collectionDimension: number): Promise<OllamaEmbeddings> {
    const requestedDimension = getModelDimension(this.requestedEmbeddingModel);
    
    console.log(`[Agentic RAG] 请求的 embedding 模型: ${this.requestedEmbeddingModel} (${requestedDimension}D)`);
    console.log(`[Agentic RAG] Milvus 集合维度: ${collectionDimension}D`);
    
    // 如果维度匹配，使用请求的模型
    if (requestedDimension === collectionDimension) {
      console.log(`[Agentic RAG] 维度匹配，使用请求的模型: ${this.requestedEmbeddingModel}`);
      return this.embeddings;
    }
    
    // 维度不匹配，自动选择匹配的模型
    const matchingModel = selectModelByDimension(collectionDimension);
    console.log(`[Agentic RAG] 维度不匹配，自动选择模型: ${matchingModel} (${collectionDimension}D)`);
    
    return new OllamaEmbeddings({
      baseUrl: this.ollamaBaseUrl,
      model: matchingModel,
    });
  }

  // ==================== 节点实现 ====================

  /** 1. 查询分析与优化节点 */
  private async analyzeAndOptimizeQuery(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '查询分析与优化';
    
    try {
      const prompt = ChatPromptTemplate.fromTemplate(`
你是一个智能查询分析器，专门为 RAG（检索增强生成）系统优化查询。

用户查询: {query}

请以JSON格式返回分析结果（不要包含markdown代码块标记）:
{{
  "originalQuery": "原始查询",
  "rewrittenQuery": "优化后的查询（保持原意，适合向量检索）",
  "intent": "查询意图: factual/exploratory/comparison/procedural/unknown",
  "complexity": "复杂度: simple/moderate/complex",
  "needsRetrieval": true,
  "keywords": ["关键词1", "关键词2"],
  "confidence": 0.0-1.0
}}

重要规则:
1. needsRetrieval 应该几乎始终为 true，因为这是一个知识库问答系统
2. 只有纯粹的闲聊问候（如"你好"、"谢谢"）才设为 false
3. 任何涉及知识、事实、概念、定义、方法、技术的问题都必须设为 true
4. rewrittenQuery 应该保持原查询的核心含义，但更适合语义检索
5. 不要过度改写查询，保留原始关键词
`);

      const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
      const result = await chain.invoke({ query: state.query });
      
      let analysis: QueryAnalysis;
      try {
        // 清理可能的 markdown 代码块标记
        const cleanedResult = result.replace(/```json\n?|\n?```/g, '').trim();
        analysis = JSON.parse(cleanedResult);
        
        // 重要：强制使用用户原始查询，防止 LLM 错误修改
        // LLM 可能会错误地解析原始查询（例如把 "你好" 解析成 "你奶"）
        analysis.originalQuery = state.query;
        
        // 检测简单问候语 - 这类问题不需要检索知识库
        const greetingPatterns = [
          /^(你好|您好|hi|hello|hey|嗨|哈喽)[\s!！。.]*$/i,
          /^(谢谢|感谢|thanks|thank you|thx)[\s!！。.]*$/i,
          /^(再见|拜拜|bye|goodbye|see you)[\s!！。.]*$/i,
          /^(早上好|下午好|晚上好|早安|晚安)[\s!！。.]*$/i,
          /^(好的|ok|okay|没问题|收到)[\s!！。.]*$/i,
        ];
        const isSimpleGreeting = greetingPatterns.some(pattern => pattern.test(state.query.trim()));
        
        if (isSimpleGreeting) {
          // 简单问候不需要检索
          analysis.needsRetrieval = false;
          analysis.intent = 'greeting' as any;
          console.log('[Agentic RAG] 检测到简单问候，跳过检索');
        } else {
          // 其他情况强制检索
          analysis.needsRetrieval = true;
        }
        
        // 确保 keywords 包含原始查询的字符（如果 LLM 返回的关键词有误）
        if (!analysis.keywords || analysis.keywords.length === 0) {
          analysis.keywords = state.query.split(/\s+/).filter(w => w.length > 0);
        }
        // 如果关键词中没有原始查询的关键字符，添加进去
        const queryChars = state.query.replace(/\s+/g, '');
        if (queryChars.length <= 4 && !analysis.keywords.some(kw => queryChars.includes(kw) || kw.includes(queryChars))) {
          analysis.keywords = [state.query.trim(), ...analysis.keywords];
        }
        
        console.log(`[Agentic RAG] 查询分析结果:`, {
          original: analysis.originalQuery,
          rewritten: analysis.rewrittenQuery,
          needsRetrieval: analysis.needsRetrieval,
          intent: analysis.intent,
          keywords: analysis.keywords
        });
      } catch {
        // 解析失败时使用默认值，强制进行检索
        console.log('[Agentic RAG] 查询分析解析失败，使用默认值');
        analysis = {
          originalQuery: state.query,
          rewrittenQuery: state.query,
          intent: 'unknown',
          complexity: 'moderate',
          needsRetrieval: true, // 默认进行检索
          keywords: state.query.split(/\s+/).filter(w => w.length > 0),
          confidence: 0.5,
        };
      }

      const stepEnd = Date.now();
      
      trace('analyze_query_result', {
        original: analysis.originalQuery,
        rewritten: analysis.rewrittenQuery,
        needsRetrieval: analysis.needsRetrieval,
        intent: analysis.intent,
      });
      
      return {
        queryAnalysis: analysis,
        processedQuery: analysis.rewrittenQuery,  // 更新处理后的查询
        shouldRetrieve: analysis.needsRetrieval,
        currentStep: 'query_analyzed',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: stepEnd,
          duration: stepEnd - stepStart,
          input: { query: state.query },
          output: analysis,
        }],
      };
    } catch (error) {
      return {
        currentStep: 'query_analyzed',
        processedQuery: state.query,  // 失败时保持原始查询
        shouldRetrieve: true,
        workflowSteps: [{
          step: stepName,
          status: 'error',
          startTime: stepStart,
          endTime: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        }],
      };
    }
  }

  /** 2. 检索节点 - 支持双查询检索（原始+处理后） */
  private async retrieve(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '文档检索';

    // 如果是简单问候类问题，跳过检索
    if (!state.shouldRetrieve) {
      console.log('[Agentic RAG] 跳过检索（shouldRetrieve=false，可能是简单问候）');
      return {
        currentStep: 'retrieved',
        retrievedDocuments: [],
        originalQueryResults: [],
        processedQueryResults: [],
        workflowSteps: [{
          step: stepName,
          status: 'skipped',
          startTime: stepStart,
          endTime: Date.now(),
          input: { reason: '查询不需要检索（简单问候或闲聊）' },
        }],
      };
    }

    try {
      // 透传原始问题：同时使用原始查询和处理后的查询
      const originalQuery = state.originalQuery || state.query;
      const processedQuery = state.processedQuery || state.queryAnalysis?.rewrittenQuery || state.query;
      const usesDualQuery = originalQuery !== processedQuery;
      
      trace('retrieve_start', {
        originalQuery,
        processedQuery,
        usesDualQuery,
        topK: state.topK,
        similarityThreshold: state.similarityThreshold,
      });
      
      // 连接 Milvus
      const milvus = getMilvusInstance(this.milvusConfig);
      await milvus.connect();
      await milvus.initializeCollection();

      // 获取集合统计信息以确定维度
      const stats = await milvus.getCollectionStats();
      const collectionDimension = stats?.embeddingDimension || this.milvusConfig.embeddingDimension || 768;
      
      console.log(`[Agentic RAG] Milvus 集合维度: ${collectionDimension}D, 文档数: ${stats?.rowCount || 0}`);

      // 根据集合维度选择匹配的 embedding 模型
      const matchingEmbeddings = await this.getMatchingEmbeddings(collectionDimension);
      const embeddingModelName = matchingEmbeddings.model || this.requestedEmbeddingModel;

      // === 双查询检索策略 ===
      let originalQueryResults: RetrievedDocument[] = [];
      let processedQueryResults: RetrievedDocument[] = [];
      let debugInfo: AgentState['debugInfo'] = {
        embeddingModel: embeddingModelName,
        collectionDimension,
      };

      // 1. 使用原始查询检索
      console.log(`[Agentic RAG] 原始查询检索: "${originalQuery}"`);
      const originalEmbedding = await matchingEmbeddings.embedQuery(originalQuery);
      debugInfo.milvusQueryVector = originalEmbedding.slice(0, 10); // 只记录前10维用于调试
      
      trace('original_query_embedding', {
        query: originalQuery,
        vectorDimension: originalEmbedding.length,
        vectorPreview: originalEmbedding.slice(0, 5),
      });

      if (originalEmbedding.length !== collectionDimension) {
        throw new Error(`向量维度不匹配: 查询=${originalEmbedding.length}D, 集合=${collectionDimension}D`);
      }

      const originalResults = await milvus.search(
        originalEmbedding,
        state.topK,
        state.similarityThreshold
      );
      
      originalQueryResults = originalResults.map(r => ({
        content: r.content,
        metadata: { ...r.metadata, querySource: 'original' },
        score: r.score,
      }));
      
      debugInfo.milvusRawScores = originalResults.map(r => r.score);
      
      trace('original_query_results', {
        resultCount: originalQueryResults.length,
        scores: originalQueryResults.map(r => r.score),
        topContent: originalQueryResults[0]?.content?.substring(0, 100),
      });

      // 2. 如果有处理后的查询且与原始查询不同，也进行检索
      if (usesDualQuery) {
        console.log(`[Agentic RAG] 处理后查询检索: "${processedQuery}"`);
        const processedEmbedding = await matchingEmbeddings.embedQuery(processedQuery);
        
        trace('processed_query_embedding', {
          query: processedQuery,
          vectorDimension: processedEmbedding.length,
        });

        const processedResults = await milvus.search(
          processedEmbedding,
          state.topK,
          state.similarityThreshold
        );
        
        processedQueryResults = processedResults.map(r => ({
          content: r.content,
          metadata: { ...r.metadata, querySource: 'processed' },
          score: r.score,
        }));
        
        trace('processed_query_results', {
          resultCount: processedQueryResults.length,
          scores: processedQueryResults.map(r => r.score),
        });
      }

      // 3. 合并结果（加权去重）
      const mergedResults = this.mergeRetrievalResults(
        originalQueryResults,
        processedQueryResults,
        0.6 // 原始查询权重略高
      );
      
      console.log(`[Agentic RAG] 合并后检索结果: ${mergedResults.length} 个文档`);
      console.log(`[Agentic RAG]   - 原始查询结果: ${originalQueryResults.length} 个`);
      console.log(`[Agentic RAG]   - 处理后查询结果: ${processedQueryResults.length} 个`);

      trace('merged_results', {
        totalCount: mergedResults.length,
        originalCount: originalQueryResults.length,
        processedCount: processedQueryResults.length,
        topScores: mergedResults.slice(0, 3).map(r => r.score),
      });

      const stepEnd = Date.now();
      return {
        retrievedDocuments: mergedResults,
        originalQueryResults,
        processedQueryResults,
        currentStep: 'retrieved',
        debugInfo,
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: stepEnd,
          duration: stepEnd - stepStart,
          input: { 
            originalQuery, 
            processedQuery, 
            usesDualQuery,
            topK: state.topK, 
            dimension: collectionDimension,
            embeddingModel: embeddingModelName,
          },
          output: { 
            documentCount: mergedResults.length,
            originalResultCount: originalQueryResults.length,
            processedResultCount: processedQueryResults.length,
            rawScores: debugInfo.milvusRawScores,
          },
        }],
      };
    } catch (error) {
      console.error('[Agentic RAG] 检索错误:', error);
      trace('retrieve_error', { error: error instanceof Error ? error.message : String(error) });
      return {
        retrievedDocuments: [],
        originalQueryResults: [],
        processedQueryResults: [],
        currentStep: 'retrieved',
        workflowSteps: [{
          step: stepName,
          status: 'error',
          startTime: stepStart,
          endTime: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        }],
      };
    }
  }

  /** 合并两个检索结果集（加权去重） */
  private mergeRetrievalResults(
    originalResults: RetrievedDocument[],
    processedResults: RetrievedDocument[],
    originalWeight: number = 0.6
  ): RetrievedDocument[] {
    const processedWeight = 1 - originalWeight;
    const contentMap = new Map<string, RetrievedDocument>();
    
    // 添加原始查询结果
    for (const doc of originalResults) {
      const key = doc.content.substring(0, 200); // 使用内容前200字符作为唯一标识
      const existing = contentMap.get(key);
      if (existing) {
        // 如果已存在，更新分数（加权平均）
        existing.score = Math.max(existing.score, doc.score * originalWeight);
      } else {
        contentMap.set(key, { ...doc, score: doc.score * originalWeight });
      }
    }
    
    // 添加处理后查询结果
    for (const doc of processedResults) {
      const key = doc.content.substring(0, 200);
      const existing = contentMap.get(key);
      if (existing) {
        // 如果已存在，融合分数
        existing.score = (existing.score + doc.score * processedWeight) / 2;
        existing.metadata.querySource = 'both';
      } else {
        contentMap.set(key, { ...doc, score: doc.score * processedWeight });
      }
    }
    
    // 按分数排序返回
    return Array.from(contentMap.values())
      .sort((a, b) => b.score - a.score);
  }

  /** 3. 自省评分节点 */
  private async selfReflect(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '自省评分';

    if (!this.config.enableSelfReflection || state.retrievedDocuments.length === 0) {
      return {
        currentStep: 'reflected',
        workflowSteps: [{
          step: stepName,
          status: 'skipped',
          startTime: stepStart,
          endTime: Date.now(),
        }],
      };
    }

    try {
      const docsForPrompt = state.retrievedDocuments
        .map((doc, i) => `[文档${i + 1}] 相似度: ${(doc.score * 100).toFixed(1)}%\n${doc.content}`)
        .join('\n\n');

      const prompt = ChatPromptTemplate.fromTemplate(`
你是一个严格的检索质量评估专家。请对以下检索结果进行自省评分。

用户查询: {query}

检索到的文档:
{documents}

请以JSON格式评估每个文档（不要包含markdown代码块标记）:
{{
  "documentScores": [
    {{
      "index": 1,
      "relevance": 0.0-1.0,
      "usefulness": 0.0-1.0,
      "factuality": 0.0-1.0,
      "overall": 0.0-1.0,
      "reasoning": "评分理由"
    }}
  ],
  "queryAlignmentScore": 0.0-1.0,
  "contextCompleteness": 0.0-1.0,
  "recommendation": "use/expand/rewrite/skip"
}}

评分标准:
- relevance: 与查询的相关程度
- usefulness: 对回答问题的帮助程度
- factuality: 信息的准确性和可信度
- queryAlignmentScore: 检索结果与查询意图的对齐程度
- contextCompleteness: 上下文是否足够完整回答问题

recommendation 说明:
- use: 结果质量好，可直接使用
- expand: 需要扩展检索范围
- rewrite: 需要重写查询
- skip: 结果太差，建议跳过检索直接回答
`);

      const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
      const result = await chain.invoke({
        query: state.query,
        documents: docsForPrompt,
      });

      let selfReflection: SelfReflectionScore;
      try {
        const cleanedResult = result.replace(/```json\n?|\n?```/g, '').trim();
        selfReflection = JSON.parse(cleanedResult);
      } catch {
        selfReflection = {
          documentScores: state.retrievedDocuments.map((_, i) => ({
            index: i + 1,
            relevance: 0.5,
            usefulness: 0.5,
            factuality: 0.5,
            overall: 0.5,
            reasoning: '解析失败，使用默认评分',
          })),
          queryAlignmentScore: 0.5,
          contextCompleteness: 0.5,
          recommendation: 'use',
        };
      }

      // 根据自省结果更新文档评分
      const updatedDocs = state.retrievedDocuments.map((doc, i) => ({
        ...doc,
        relevanceScore: selfReflection.documentScores[i]?.relevance || doc.score,
        factualScore: selfReflection.documentScores[i]?.factuality || 0.5,
      }));

      // 决定是否需要重试
      const shouldRewrite = selfReflection.recommendation === 'rewrite' && state.retryCount < state.maxRetries;

      const stepEnd = Date.now();
      return {
        retrievedDocuments: updatedDocs,
        selfReflection,
        shouldRewrite,
        retryCount: shouldRewrite ? state.retryCount + 1 : state.retryCount,
        currentStep: 'reflected',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: stepEnd,
          duration: stepEnd - stepStart,
          output: selfReflection,
        }],
      };
    } catch (error) {
      return {
        currentStep: 'reflected',
        workflowSteps: [{
          step: stepName,
          status: 'error',
          startTime: stepStart,
          endTime: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        }],
      };
    }
  }

  /** 3.5 检索评估节点（Retrieval Grader） - 快速判断检索质量 */
  private async gradeRetrieval(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '检索评估';

    if (state.retrievedDocuments.length === 0) {
      trace('grade_retrieval_empty', { reason: '没有检索结果' });
      return {
        retrievalGrade: {
          isRelevant: false,
          score: 0,
          keywordMatchScore: 0,
          semanticScore: 0,
          hasAnswerSignals: false,
          reasoning: '没有检索到任何文档',
          documentGrades: [],
        },
        shouldRewrite: state.retryCount < state.maxRetries,
        currentStep: 'graded',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: Date.now(),
          output: { isRelevant: false, reason: 'no_results' },
        }],
      };
    }

    try {
      // 提取查询关键词
      const queryKeywords = this.extractKeywords(state.originalQuery || state.query);
      console.log(`[Agentic RAG] 检索评估 - 查询关键词: ${queryKeywords.join(', ')}`);
      
      trace('grade_retrieval_start', {
        queryKeywords,
        documentCount: state.retrievedDocuments.length,
      });

      const documentGrades: RetrievalGradeResult['documentGrades'] = [];
      let totalKeywordScore = 0;
      let totalSemanticScore = 0;
      let answerSignalCount = 0;

      // 快速评估每个文档
      for (let i = 0; i < state.retrievedDocuments.length; i++) {
        const doc = state.retrievedDocuments[i];
        const content = doc.content.toLowerCase();
        
        // 关键词匹配评分
        const matchedKeywords = queryKeywords.filter(kw => content.includes(kw.toLowerCase()));
        const keywordScore = queryKeywords.length > 0 
          ? matchedKeywords.length / queryKeywords.length 
          : 0;
        
        // 语义相关性（使用已有的向量相似度分数）
        const semanticScore = doc.score;
        
        // 答案信号检测（检查是否包含答案模式）
        const answerPatterns = [
          /是.{1,20}的/, /指的是/, /可以.{1,20}来/, /通过.{1,20}实现/,
          /包括.{1,30}等/, /主要.{1,20}有/, /\d+[种个类]/, /步骤.{0,5}[:：]/,
          /方法.{0,5}[:：]/, /原因.{0,5}[:：]/, /定义.{0,5}[:：]/,
        ];
        const hasAnswerSignal = answerPatterns.some(pattern => pattern.test(doc.content));
        if (hasAnswerSignal) answerSignalCount++;
        
        // 综合评分
        const docScore = keywordScore * 0.4 + semanticScore * 0.5 + (hasAnswerSignal ? 0.1 : 0);
        const isRelevant = docScore >= (state.gradePassThreshold || 0.5);
        
        documentGrades.push({
          index: i,
          isRelevant,
          score: docScore,
          matchedKeywords,
          reasoning: `关键词匹配: ${matchedKeywords.length}/${queryKeywords.length}, 语义分数: ${(semanticScore * 100).toFixed(1)}%, 答案信号: ${hasAnswerSignal ? '有' : '无'}`,
        });
        
        totalKeywordScore += keywordScore;
        totalSemanticScore += semanticScore;
      }

      const avgKeywordScore = totalKeywordScore / state.retrievedDocuments.length;
      const avgSemanticScore = totalSemanticScore / state.retrievedDocuments.length;
      const hasAnswerSignals = answerSignalCount > 0;
      
      // 综合评分
      const overallScore = avgKeywordScore * 0.35 + avgSemanticScore * 0.5 + (hasAnswerSignals ? 0.15 : 0);
      const isRelevant = overallScore >= (state.gradePassThreshold || 0.5);
      
      // 决定是否需要重写查询
      const shouldRewrite = !isRelevant && state.retryCount < state.maxRetries;

      const retrievalGrade: RetrievalGradeResult = {
        isRelevant,
        score: overallScore,
        keywordMatchScore: avgKeywordScore,
        semanticScore: avgSemanticScore,
        hasAnswerSignals,
        reasoning: isRelevant 
          ? `检索结果质量良好 (综合评分: ${(overallScore * 100).toFixed(1)}%)`
          : `检索结果质量不足 (综合评分: ${(overallScore * 100).toFixed(1)}%)，${shouldRewrite ? '将重写查询' : '已达最大重试次数'}`,
        documentGrades,
      };

      trace('grade_retrieval_result', {
        isRelevant,
        overallScore,
        avgKeywordScore,
        avgSemanticScore,
        hasAnswerSignals,
        shouldRewrite,
        relevantDocCount: documentGrades.filter(d => d.isRelevant).length,
      });

      console.log(`[Agentic RAG] 检索评估结果: ${isRelevant ? '✅ 通过' : '❌ 未通过'} (评分: ${(overallScore * 100).toFixed(1)}%)`);

      const stepEnd = Date.now();
      return {
        retrievalGrade,
        shouldRewrite,
        retryCount: shouldRewrite ? state.retryCount + 1 : state.retryCount,
        currentStep: 'graded',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: stepEnd,
          duration: stepEnd - stepStart,
          input: { queryKeywords, documentCount: state.retrievedDocuments.length },
          output: retrievalGrade,
        }],
      };
    } catch (error) {
      console.error('[Agentic RAG] 检索评估错误:', error);
      return {
        currentStep: 'graded',
        workflowSteps: [{
          step: stepName,
          status: 'error',
          startTime: stepStart,
          endTime: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        }],
      };
    }
  }

  /** 从查询中提取关键词 */
  private extractKeywords(query: string): string[] {
    // 移除常见停用词
    const stopWords = new Set([
      '的', '是', '在', '有', '和', '与', '或', '了', '这', '那', '什么', '怎么', '如何',
      '为什么', '哪些', '哪个', '吗', '呢', '啊', '吧', '嘛', '呀', '哦', '哈',
      'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'which',
      'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will', 'to', 'of', 'in',
    ]);
    
    // 分词（简单实现：按空格和标点分割）
    const words = query
      .toLowerCase()
      .split(/[\s,，。？！、；：""''（）【】\[\]{}]+/)
      .filter(w => w.length >= 2 && !stopWords.has(w));
    
    // 去重并返回
    return [...new Set(words)];
  }

  /** 4. 质量评估节点 */
  private async evaluateQuality(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '检索质量评估';

    const docs = state.retrievedDocuments;
    
    if (docs.length === 0) {
      return {
        retrievalQuality: {
          overallScore: 0,
          relevanceScore: 0,
          coverageScore: 0,
          diversityScore: 0,
          isAcceptable: false,
          suggestions: ['没有检索到任何文档，建议重写查询'],
        },
        currentStep: 'quality_evaluated',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: Date.now(),
        }],
      };
    }

    // 计算各项评分
    const relevanceScore = docs.reduce((sum, d) => sum + (d.relevanceScore || d.score), 0) / docs.length;
    
    // 覆盖度：检查文档来源多样性
    const sources = new Set(docs.map(d => d.metadata?.source || 'unknown'));
    const coverageScore = Math.min(sources.size / Math.min(3, state.topK), 1);
    
    // 多样性：计算内容长度的标准差归一化
    const lengths = docs.map(d => d.content.length);
    const avgLen = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const variance = lengths.reduce((sum, l) => sum + Math.pow(l - avgLen, 2), 0) / lengths.length;
    const diversityScore = Math.min(Math.sqrt(variance) / avgLen, 1);

    const overallScore = (relevanceScore * 0.5 + coverageScore * 0.3 + diversityScore * 0.2);
    const isAcceptable = overallScore >= 0.4;

    const suggestions: string[] = [];
    if (relevanceScore < 0.5) suggestions.push('相关性较低，建议优化查询关键词');
    if (coverageScore < 0.5) suggestions.push('覆盖度不足，建议增加检索数量');
    if (diversityScore < 0.3) suggestions.push('结果单一，建议扩展查询范围');

    const retrievalQuality: RetrievalQuality = {
      overallScore,
      relevanceScore,
      coverageScore,
      diversityScore,
      isAcceptable,
      suggestions,
    };

    const stepEnd = Date.now();
    return {
      retrievalQuality,
      currentStep: 'quality_evaluated',
      workflowSteps: [{
        step: stepName,
        status: 'completed',
        startTime: stepStart,
        endTime: stepEnd,
        duration: stepEnd - stepStart,
        output: retrievalQuality,
      }],
    };
  }

  /** 5. 生成答案节点 */
  private async generate(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '答案生成';

    try {
      // 检查是否是简单问候（跳过了检索）
      const isGreeting = !state.shouldRetrieve && state.retrievedDocuments.length === 0;
      
      if (isGreeting) {
        // 对简单问候直接生成友好回复，不使用知识库
        console.log('[Agentic RAG] 简单问候，生成友好回复');
        
        const greetingPrompt = ChatPromptTemplate.fromTemplate(`
你是一个友好的AI助手。用户向你打招呼或进行简单的闲聊。
请用自然、友好的方式回应，不需要提及知识库。

用户说: {question}

请直接回应用户（简短、友好、自然）:
`);

        const chain = greetingPrompt.pipe(this.llm).pipe(new StringOutputParser());
        const answer = await chain.invoke({
          question: state.query,
        });

        const stepEnd = Date.now();
        return {
          context: '（简单问候，无需检索知识库）',
          answer,
          currentStep: 'generated',
          workflowSteps: [{
            step: stepName,
            status: 'completed',
            startTime: stepStart,
            endTime: stepEnd,
            duration: stepEnd - stepStart,
            input: { type: 'greeting', query: state.query },
            output: { answerLength: answer.length },
          }],
        };
      }

      // 正常的知识库问答流程
      const context = state.retrievedDocuments.length > 0
        ? state.retrievedDocuments
            .map((doc, i) => {
              const relevance = doc.relevanceScore ? ` (相关性: ${(doc.relevanceScore * 100).toFixed(0)}%)` : '';
              return `[文档${i + 1}]${relevance}\n${doc.content}`;
            })
            .join('\n\n---\n\n')
        : '没有找到相关文档。';

      const prompt = ChatPromptTemplate.fromTemplate(`
你是一个专业的知识库助手。请根据下方提供的上下文信息来回答用户的问题。

【上下文信息】：
{context}

【用户问题】：
{question}

回答要求：
1. 如果上下文信息中不包含答案，请明确说明"根据现有知识库，我无法找到相关信息"
2. 不要编造不在上下文中的信息
3. 回答要简洁明了，使用中文
4. 如果可能，引用文档来源
`);

      const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
      const answer = await chain.invoke({
        context,
        question: state.query,
      });

      const stepEnd = Date.now();
      return {
        context,
        answer,
        currentStep: 'generated',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: stepEnd,
          duration: stepEnd - stepStart,
          input: { contextLength: context.length },
          output: { answerLength: answer.length },
        }],
      };
    } catch (error) {
      return {
        answer: '抱歉，生成答案时发生错误。',
        currentStep: 'generated',
        workflowSteps: [{
          step: stepName,
          status: 'error',
          startTime: stepStart,
          endTime: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        }],
      };
    }
  }

  /** 6. 幻觉检查节点 */
  private async checkHallucination(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '幻觉检查';

    // 跳过简单问候的幻觉检查（因为没有使用知识库）
    const isGreeting = !state.shouldRetrieve && state.retrievedDocuments.length === 0;
    
    if (!this.config.enableHallucinationCheck || !state.answer || isGreeting) {
      return {
        currentStep: 'hallucination_checked',
        hallucinationCheck: isGreeting ? {
          hasHallucination: false,
          confidence: 1.0,
          problematicClaims: [],
          supportedClaims: ['简单问候回复，无需检查'],
          overallFactualScore: 1.0,
        } : undefined,
        workflowSteps: [{
          step: stepName,
          status: 'skipped',
          startTime: stepStart,
          endTime: Date.now(),
          input: isGreeting ? { reason: '简单问候，无需幻觉检查' } : undefined,
        }],
      };
    }

    try {
      const prompt = ChatPromptTemplate.fromTemplate(`
你是一个事实核查专家。请检查以下答案是否与提供的上下文信息一致。

【上下文信息】：
{context}

【生成的答案】：
{answer}

请以JSON格式返回检查结果（不要包含markdown代码块标记）:
{{
  "hasHallucination": true/false,
  "confidence": 0.0-1.0,
  "problematicClaims": ["有问题的声明1", "有问题的声明2"],
  "supportedClaims": ["有据可查的声明1", "有据可查的声明2"],
  "overallFactualScore": 0.0-1.0
}}

检查标准:
1. 答案中的每个事实陈述是否都能在上下文中找到支持
2. 是否有超出上下文的推断或臆测
3. 数字、日期、名称等是否准确
`);

      const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
      const result = await chain.invoke({
        context: state.context,
        answer: state.answer,
      });

      let hallucinationCheck: HallucinationCheck;
      try {
        const cleanedResult = result.replace(/```json\n?|\n?```/g, '').trim();
        hallucinationCheck = JSON.parse(cleanedResult);
      } catch {
        hallucinationCheck = {
          hasHallucination: false,
          confidence: 0.5,
          problematicClaims: [],
          supportedClaims: [],
          overallFactualScore: 0.5,
        };
      }

      const stepEnd = Date.now();
      return {
        hallucinationCheck,
        currentStep: 'hallucination_checked',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: stepEnd,
          duration: stepEnd - stepStart,
          output: hallucinationCheck,
        }],
      };
    } catch (error) {
      return {
        currentStep: 'hallucination_checked',
        workflowSteps: [{
          step: stepName,
          status: 'error',
          startTime: stepStart,
          endTime: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        }],
      };
    }
  }

  /** 7. 查询重写节点（用于重试） */
  private async rewriteQuery(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const stepStart = Date.now();
    const stepName = '查询重写';

    try {
      const prompt = ChatPromptTemplate.fromTemplate(`
你是一个查询优化专家。原始查询的检索结果不理想，请重写查询以获得更好的结果。

原始查询: {query}
之前的优化查询: {previousQuery}
检索质量评估: {qualityFeedback}

请直接返回重写后的查询（不要包含任何解释或标记）:
`);

      const qualityFeedback = state.retrievalQuality
        ? `相关性: ${(state.retrievalQuality.relevanceScore * 100).toFixed(0)}%, 建议: ${state.retrievalQuality.suggestions.join('; ')}`
        : '无';

      const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
      const rewrittenQuery = await chain.invoke({
        query: state.query,
        previousQuery: state.queryAnalysis?.rewrittenQuery || state.query,
        qualityFeedback,
      });

      const stepEnd = Date.now();
      return {
        queryAnalysis: {
          ...state.queryAnalysis!,
          rewrittenQuery: rewrittenQuery.trim(),
        },
        shouldRewrite: false,
        currentStep: 'query_rewritten',
        workflowSteps: [{
          step: stepName,
          status: 'completed',
          startTime: stepStart,
          endTime: stepEnd,
          duration: stepEnd - stepStart,
          input: { originalQuery: state.query },
          output: { rewrittenQuery: rewrittenQuery.trim() },
        }],
      };
    } catch (error) {
      return {
        shouldRewrite: false,
        currentStep: 'query_rewritten',
        workflowSteps: [{
          step: stepName,
          status: 'error',
          startTime: stepStart,
          endTime: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        }],
      };
    }
  }

  /** 8. 结束节点 */
  private async finalize(state: typeof AgentStateAnnotation.State): Promise<Partial<typeof AgentStateAnnotation.State>> {
    const endTime = Date.now();
    return {
      endTime,
      totalDuration: endTime - state.startTime,
      currentStep: 'completed',
    };
  }

  // ==================== 图构建 ====================

  /**
   * 工作流图结构：
   * 
   *   START
   *     ↓
   *   analyze_query ─────────→ (不需要检索) → generate
   *     ↓ (需要检索)
   *   retrieve (双查询: 原始 + 处理后)
   *     ↓
   *   grade_retrieval (快速评估检索质量)
   *     ↓ (评分低) → rewrite_query → retrieve
   *     ↓ (评分高)
   *   self_reflect (深度自省评分)
   *     ↓
   *   evaluate_quality
   *     ↓
   *   generate
   *     ↓
   *   check_hallucination
   *     ↓
   *   finalize → END
   */
  private buildGraph() {
    const workflow = new StateGraph(AgentStateAnnotation)
      // 添加节点
      .addNode('analyze_query', this.analyzeAndOptimizeQuery.bind(this))
      .addNode('retrieve', this.retrieve.bind(this))
      .addNode('grade_retrieval', this.gradeRetrieval.bind(this))  // 新增：检索评估节点
      .addNode('self_reflect', this.selfReflect.bind(this))
      .addNode('evaluate_quality', this.evaluateQuality.bind(this))
      .addNode('generate', this.generate.bind(this))
      .addNode('check_hallucination', this.checkHallucination.bind(this))
      .addNode('rewrite_query', this.rewriteQuery.bind(this))
      .addNode('finalize', this.finalize.bind(this))
      
      // 定义边
      .addEdge(START, 'analyze_query')
      
      // 分析查询后：决定是否检索
      .addConditionalEdges('analyze_query', (state) => {
        trace('edge_after_analyze', { shouldRetrieve: state.shouldRetrieve });
        return state.shouldRetrieve ? 'retrieve' : 'generate';
      })
      
      // 检索后：进入评估节点
      .addEdge('retrieve', 'grade_retrieval')
      
      // 检索评估后：根据评分决定下一步
      .addConditionalEdges('grade_retrieval', (state) => {
        const grade = state.retrievalGrade;
        trace('edge_after_grade', { 
          isRelevant: grade?.isRelevant, 
          score: grade?.score,
          shouldRewrite: state.shouldRewrite,
          retryCount: state.retryCount,
          maxRetries: state.maxRetries,
        });
        
        // 如果评分不通过且未超过重试次数，重写查询
        if (!grade?.isRelevant && state.shouldRewrite && state.retryCount <= state.maxRetries) {
          console.log(`[Agentic RAG] 检索评估未通过，重写查询 (重试 ${state.retryCount}/${state.maxRetries})`);
          return 'rewrite_query';
        }
        
        // 否则进入自省评分（深度评估）
        return 'self_reflect';
      })
      
      // 自省评分后：进入质量评估
      .addEdge('self_reflect', 'evaluate_quality')
      
      // 重写查询后：重新检索
      .addEdge('rewrite_query', 'retrieve')
      
      // 质量评估后：生成答案
      .addEdge('evaluate_quality', 'generate')
      
      // 生成后：幻觉检查
      .addEdge('generate', 'check_hallucination')
      
      // 幻觉检查后：结束
      .addEdge('check_hallucination', 'finalize')
      .addEdge('finalize', END);

    return workflow.compile();
  }

  // ==================== 公共接口 ====================

  /** 执行 Agentic RAG 查询 */
  async query(
    question: string,
    options: {
      topK?: number;
      similarityThreshold?: number;
      maxRetries?: number;
      gradePassThreshold?: number;
    } = {}
  ): Promise<AgentState> {
    trace('query_start', { question, options });
    
    const initialState = {
      query: question,
      originalQuery: question,        // 透传原始问题
      processedQuery: question,       // 初始与原始相同
      topK: options.topK || 5,
      similarityThreshold: options.similarityThreshold || 0.3,
      maxRetries: options.maxRetries || 2,
      gradePassThreshold: options.gradePassThreshold || 0.5,  // 检索评分通过阈值
      retrievedDocuments: [],
      originalQueryResults: [],
      processedQueryResults: [],
      context: '',
      answer: '',
      currentStep: 'start',
      retryCount: 0,
      shouldRewrite: false,
      shouldRetrieve: true,
      workflowSteps: [],
      startTime: Date.now(),
    };

    try {
      console.log(`[Agentic RAG] 开始查询: "${question}"`);
      console.log(`[Agentic RAG] 配置: topK=${initialState.topK}, threshold=${initialState.similarityThreshold}, maxRetries=${initialState.maxRetries}`);
      
      const result = await this.graph.invoke(initialState);
      
      trace('query_complete', {
        answer: result.answer?.substring(0, 100),
        totalDuration: result.totalDuration,
        retryCount: result.retryCount,
        documentCount: result.retrievedDocuments?.length,
      });
      
      return result as AgentState;
    } catch (error) {
      trace('query_error', { error: error instanceof Error ? error.message : String(error) });
      return {
        ...initialState,
        error: error instanceof Error ? error.message : String(error),
        currentStep: 'error',
        endTime: Date.now(),
        totalDuration: Date.now() - initialState.startTime,
      } as AgentState;
    }
  }

  /** 流式执行（返回每个步骤的更新） */
  async *streamQuery(
    question: string,
    options: {
      topK?: number;
      similarityThreshold?: number;
      maxRetries?: number;
      gradePassThreshold?: number;
    } = {}
  ): AsyncGenerator<Partial<AgentState>> {
    const initialState = {
      query: question,
      originalQuery: question,
      processedQuery: question,
      topK: options.topK || 5,
      similarityThreshold: options.similarityThreshold || 0.3,
      maxRetries: options.maxRetries || 2,
      gradePassThreshold: options.gradePassThreshold || 0.5,
      retrievedDocuments: [],
      originalQueryResults: [],
      processedQueryResults: [],
      context: '',
      answer: '',
      currentStep: 'start',
      retryCount: 0,
      shouldRewrite: false,
      shouldRetrieve: true,
      workflowSteps: [],
      startTime: Date.now(),
    };

    try {
      for await (const chunk of await this.graph.stream(initialState)) {
        yield chunk;
      }
    } catch (error) {
      yield {
        error: error instanceof Error ? error.message : String(error),
        currentStep: 'error',
      };
    }
  }
}

// 导出工厂函数
export function createAgenticRAG(config?: AgenticRAGConfig): AgenticRAGSystem {
  return new AgenticRAGSystem(config);
}

export default AgenticRAGSystem;
