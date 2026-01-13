import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { ObservabilityEngine, type Trace } from "./observability";
import { 
  BPETokenizer, 
  type BPETokenizationResult, 
  type TokenInfo,
  type ProcessingStep,
  type VectorWeight,
  type DensityPoint
} from "./bpe-tokenizer";

// 导出类型供其他模块使用
export type { TokenInfo, ProcessingStep, VectorWeight, DensityPoint };

export interface VectorFeatures {
  techScore: number;
  businessScore: number;
  dailyScore: number;
  emotionScore: number;
  vectorMagnitude: number;
}

export interface SemanticAnalysis {
  context: string;
  semanticCategory: string;
  nearestConcepts: string[];
  confidence: number;
  vectorFeatures?: VectorFeatures;
}

export interface VectorizationProgress {
  current: number;
  total: number;
  filename: string;
  status: string;
  dimension?: number;
  timeTaken?: number;
}

export interface QueryVectorizationProgress {
  status: 'started' | 'tokenizing' | 'preprocessing' | 'embedding' | 'completed';
  message: string;
  timeTaken?: number;
  tokenization?: {
    originalText: string;
    tokenCount: number;
    tokens: TokenInfo[];
    processingTime: number;
    // BPE 可视化数据
    processingSteps?: ProcessingStep[];
    vectorWeights?: VectorWeight[];
    densityHeatmap?: DensityPoint[];
    statistics?: {
      totalTokens: number;
      uniqueTokens: number;
      subwordRatio: number;
      averageTokenLength: number;
      processingTime: number;
    };
    modelInfo?: {
      name: string;
      vocabSize: number;
      mergesCount: number;
    };
  };
  embedding?: {
    embedding: number[];
    embeddingDimension: number;
    semanticAnalysis: SemanticAnalysis;
    modelInfo: {
      name: string;
      vocabularySize?: number;
    };
  };
}

export interface SimilaritySearchResult {
  document: Document;
  similarity: number;
  index: number;
}

export interface RetrievalDetails {
  query: string;
  queryEmbedding: number[];
  queryVectorizationTime: number;
  topK: number;
  threshold: number;
  totalDocuments: number;
  searchTime: number;
  searchResults: SimilaritySearchResult[];
}

// 使用 BPE 算法的词元化器（基于 @xenova/transformers，支持多模型切换）
class SimpleTokenizer {
  private bpeTokenizer: BPETokenizer;
  private currentModel: string = 'Xenova/bert-base-multilingual-cased';

  constructor(private observabilityEngine?: ObservabilityEngine) {
    this.bpeTokenizer = new BPETokenizer(this.currentModel, observabilityEngine);
  }

  /**
   * 切换模型
   */
  async switchModel(modelName: string): Promise<void> {
    await this.bpeTokenizer.switchModel(modelName);
    this.currentModel = modelName;
  }

  /**
   * 获取当前模型
   */
  getCurrentModel(): string {
    return this.bpeTokenizer.getCurrentModel();
  }

  /**
   * 获取支持的模型列表
   */
  getSupportedModels(): string[] {
    return this.bpeTokenizer.getSupportedModels();
  }

  /**
   * 词元化文本（返回 TokenInfo 数组）
   */
  async tokenize(text: string, parentTraceId?: string): Promise<TokenInfo[]> {
    const result = await this.tokenizeWithDetails(text, parentTraceId);
    return result.tokens;
  }

  /**
   * 获取完整的词元化结果（包含可视化数据）
   */
  async tokenizeWithDetails(text: string, parentTraceId?: string): Promise<BPETokenizationResult> {
    // 确保 text 是字符串类型
    if (typeof text !== 'string') {
      text = String(text || '');
    }

    if (!text.trim()) {
      return {
        tokens: [],
        originalText: text,
        processingSteps: [],
        vectorWeights: [],
        densityHeatmap: [],
        statistics: {
          totalTokens: 0,
          uniqueTokens: 0,
          subwordRatio: 0,
          averageTokenLength: 0,
          processingTime: 0
        },
        modelInfo: {
          name: this.currentModel,
          vocabSize: 0,
          mergesCount: 0
        }
      };
    }

    return await this.bpeTokenizer.tokenize(text, true, parentTraceId);
  }
}

// 内存向量存储
class SimpleMemoryVectorStore {
  private documents: Document[] = [];
  private embeddings: number[][] = [];
  private tokenizer: SimpleTokenizer;

  constructor(
    private embeddingModel: OllamaEmbeddings,
    private onProgress?: (progress: VectorizationProgress) => void,
    private onQueryProgress?: (progress: QueryVectorizationProgress) => void
  ) {
    this.tokenizer = new SimpleTokenizer();
  }

  clear() {
    this.documents = [];
    this.embeddings = [];
  }

  async addDocuments(docs: Document[]) {
    const total = docs.length;
    
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const filename = doc.metadata?.source || `document-${i}`;
      
      this.onProgress?.({
        current: i + 1,
        total,
        filename,
        status: `正在向量化文档 ${i + 1}/${total}: ${filename}`
      });

      const startTime = Date.now();
      const embedding = await this.embeddingModel.embedQuery(doc.pageContent);
      const timeTaken = Date.now() - startTime;

      this.documents.push(doc);
      this.embeddings.push(embedding);

      this.onProgress?.({
        current: i + 1,
        total,
        filename,
        status: `文档 ${i + 1} 向量化完成，向量维度: ${embedding.length}`,
        dimension: embedding.length,
        timeTaken
      });
    }
  }

  async similaritySearchWithDetails(
    query: string,
    k: number,
    threshold: number,
    parentTraceId?: string,  // 主 Trace ID，用于关联 BPE tokenization
    onQueryProgress?: (progress: QueryVectorizationProgress) => void
  ): Promise<RetrievalDetails> {
    const startTime = Date.now();

    // 1. 词元化（使用 BPE tokenizer 获取完整结果）
    onQueryProgress?.({
      status: 'tokenizing',
      message: '正在进行词元化...'
    });

    // 使用 tokenizeWithDetails 获取完整的 BPE 结果（包含可视化数据）
    const tokenizationResult = await (this.tokenizer as any).tokenizeWithDetails(query, parentTraceId);
    const tokenizationTime = Date.now() - startTime;

    onQueryProgress?.({
      status: 'tokenizing',
      message: '词元化完成',
      timeTaken: tokenizationTime,
      tokenization: {
        originalText: query,
        tokenCount: tokenizationResult.tokens.length,
        tokens: tokenizationResult.tokens,
        processingTime: tokenizationTime,
        // BPE 可视化数据
        processingSteps: tokenizationResult.processingSteps,
        vectorWeights: tokenizationResult.vectorWeights,
        densityHeatmap: tokenizationResult.densityHeatmap,
        statistics: tokenizationResult.statistics,
        modelInfo: tokenizationResult.modelInfo
      }
    });

    // 2. 向量化
    onQueryProgress?.({
      status: 'embedding',
      message: '正在生成查询向量...'
    });

    const embeddingStartTime = Date.now();
    const queryEmbedding = await this.embeddingModel.embedQuery(query);
    const embeddingTime = Date.now() - embeddingStartTime;

    // 简化的语义分析
    const semanticAnalysis = this.analyzeSemantics(query, queryEmbedding);

    onQueryProgress?.({
      status: 'completed',
      message: '查询向量化完成',
      timeTaken: Date.now() - startTime,
      embedding: {
        embedding: queryEmbedding,
        embeddingDimension: queryEmbedding.length,
        semanticAnalysis,
        modelInfo: {
          name: 'nomic-embed-text'
        }
      }
    });

    // 3. 相似度搜索
    const searchStartTime = Date.now();
    const similarities = this.embeddings.map((embedding, index) => ({
      index,
      similarity: this.cosineSimilarity(queryEmbedding, embedding)
    }));

    // 过滤和排序
    const filteredResults = similarities
      .filter(result => result.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);

    const searchResults: SimilaritySearchResult[] = filteredResults.map(result => ({
      document: this.documents[result.index],
      similarity: result.similarity,
      index: result.index
    }));

    const searchTime = Date.now() - searchStartTime;

    return {
      query,
      queryEmbedding,
      queryVectorizationTime: embeddingTime,
      topK: k,
      threshold,
      totalDocuments: this.documents.length,
      searchTime,
      searchResults
    };
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }

  private analyzeSemantics(text: string, embedding: number[]): SemanticAnalysis {
    // 简化的语义分析
    const lowerText = text.toLowerCase();
    let context = '';
    let semanticCategory = '';
    let nearestConcepts: string[] = [];
    let confidence = 0.8;

    if (lowerText.includes('智能') || lowerText.includes('AI') || lowerText.includes('人工智能')) {
      context = '人工智能语境';
      semanticCategory = 'AI技术';
      nearestConcepts = ['人工智能', '机器学习', '深度学习', '算法'];
      confidence = 0.9;
    } else if (lowerText.includes('手机') || lowerText.includes('苹果')) {
      context = '科技产品语境';
      semanticCategory = '电子设备';
      nearestConcepts = ['智能手机', '电子设备', '科技产品'];
      confidence = 0.85;
    } else {
      context = '通用语境';
      semanticCategory = '一般';
      nearestConcepts = ['文本', '信息', '内容'];
      confidence = 0.7;
    }

    return {
      context,
      semanticCategory,
      nearestConcepts,
      confidence
    };
  }

  getDocumentCount(): number {
    return this.documents.length;
  }

  getEmbeddingDimension(): number {
    return this.embeddings.length > 0 ? this.embeddings[0].length : 0;
  }
}

// 主要的 RAG 系统类
export class LocalRAGSystem {
  private llm: Ollama;
  private embeddings: OllamaEmbeddings;
  private vectorStore!: SimpleMemoryVectorStore;
  private isInitialized = false;
  private observabilityEngine: ObservabilityEngine;

  constructor(
    private config: {
      ollamaBaseUrl?: string;
      llmModel?: string;
      embeddingModel?: string;
      onVectorizationProgress?: (progress: VectorizationProgress) => void;
      onRetrievalDetails?: (details: RetrievalDetails) => void;
      onQueryVectorizationProgress?: (progress: QueryVectorizationProgress) => void;
      onTraceUpdate?: (trace: Trace) => void;
    } = {}
  ) {
    const {
      ollamaBaseUrl = "http://localhost:11434",
      llmModel = "llama3.1",
      embeddingModel = "nomic-embed-text",
    } = config;

    this.llm = new Ollama({
      baseUrl: ollamaBaseUrl,
      model: llmModel,
      temperature: 0,
    });

    this.embeddings = new OllamaEmbeddings({
      baseUrl: ollamaBaseUrl,
      model: embeddingModel,
    });

    this.observabilityEngine = new ObservabilityEngine({
      onTraceUpdate: config.onTraceUpdate,
    });

    this.vectorStore = new SimpleMemoryVectorStore(
      this.embeddings,
      config.onVectorizationProgress,
      config.onQueryVectorizationProgress
    );
  }

  async initializeDatabase(docsPath?: string): Promise<void> {
    console.log("正在初始化 RAG 系统...");
    console.log("--- 正在初始化向量数据库 ---");

    // 创建示例文档
    const documents = [
      new Document({
        pageContent: "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、问题解决、感知和语言理解。",
        metadata: { source: "ai-intro.txt" }
      }),
      new Document({
        pageContent: "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。它基于算法，这些算法可以从数据中学习并做出预测或决策。",
        metadata: { source: "ml-intro.txt" }
      }),
      new Document({
        pageContent: "深度学习是机器学习的一个子领域，它使用具有多层的神经网络来模拟人脑的工作方式。这种方法在图像识别、自然语言处理和语音识别等领域取得了显著成功。",
        metadata: { source: "dl-intro.txt" }
      }),
      new Document({
        pageContent: "智能手机是一种功能强大的移动设备，集成了计算、通信和娱乐功能。现代智能手机配备了先进的处理器、高分辨率显示屏和多种传感器。",
        metadata: { source: "smartphone-intro.txt" }
      }),
      new Document({
        pageContent: "苹果公司是一家美国跨国科技公司，以设计、开发和销售消费电子产品、计算机软件和在线服务而闻名。其产品包括iPhone、iPad、Mac电脑等。",
        metadata: { source: "apple-intro.txt" }
      })
    ];

    // 文本分割
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const splitDocs = await textSplitter.splitDocuments(documents);
    console.log(`切分为 ${splitDocs.length} 个文本块`);

    // 向量化
    await this.vectorStore.addDocuments(splitDocs);

    this.isInitialized = true;
    console.log("向量数据库初始化完成。");
    console.log("RAG 系统初始化完成！");
  }

  async reinitialize(documents: string[]): Promise<void> {
    console.log("正在重新初始化 RAG 系统...");
    
    // 清空现有数据
    this.vectorStore.clear();
    
    // 将字符串数组转换为 Document 对象
    const docs = documents.map((content, index) => 
      new Document({
        pageContent: content,
        metadata: { source: `document-${index}.txt` }
      })
    );
    
    if (docs.length === 0) {
      this.isInitialized = false;
      console.log("系统已清空");
      return;
    }
    
    // 文本分割
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const splitDocs = await textSplitter.splitDocuments(docs);
    console.log(`切分为 ${splitDocs.length} 个文本块`);

    // 向量化
    await this.vectorStore.addDocuments(splitDocs);

    this.isInitialized = true;
    console.log("向量数据库重新初始化完成。");
    console.log("RAG 系统重新初始化完成！");
  }

  async askWithDetails(
    question: string,
    options: {
      topK?: number;
      similarityThreshold?: number;
      userId?: string;
      sessionId?: string;
      tokenizerModel?: string;  // 允许指定 tokenizer 模型
    } = {}
  ): Promise<{
    answer: string;
    retrievalDetails: RetrievalDetails;
    context: string;
    traceId: string;
    queryAnalysis?: {
      tokenization?: any;
      embedding?: any;
    };
  }> {
    if (!this.isInitialized) {
      throw new Error("RAG 系统尚未初始化，请先调用 initializeDatabase()");
    }

    const { topK = 3, similarityThreshold = 0.0, userId, sessionId, tokenizerModel } = options;
    
    // 如果指定了 tokenizer 模型，切换模型
    if (tokenizerModel) {
      try {
        await (this.vectorStore as any).tokenizer.switchModel(tokenizerModel);
        console.log(`[RAG System] 已切换 tokenizer 模型到: ${tokenizerModel}`);
      } catch (error) {
        console.error(`[RAG System] 切换 tokenizer 模型失败:`, error);
      }
    }

    // 创建 Trace
    const traceId = this.observabilityEngine.createTrace({
      name: 'RAG Query',
      userId,
      sessionId,
      input: { question, topK, similarityThreshold },
      metadata: {
        model: this.llm.model,
        embeddingModel: this.embeddings.model,
        timestamp: new Date().toISOString()
      },
      tags: ['rag', 'question-answering']
    });

    try {
      // 查询理解与向量化 Span
      const querySpanId = this.observabilityEngine.createSpan({
        traceId,
        name: 'Query Understanding & Vectorization',
        input: { question },
        metadata: { stage: 'query_processing' }
      });

      // 向量检索 Span
      const retrievalSpanId = this.observabilityEngine.createSpan({
        traceId,
        name: 'Vector Retrieval',
        parentObservationId: querySpanId,
        input: { question, topK, similarityThreshold },
        metadata: { stage: 'retrieval' }
      });

      // 收集 tokenization 和 embedding 数据
      let tokenizationData: any = null;
      let embeddingData: any = null;
      
      // 创建包装的进度回调
      const wrappedProgressCallback = (progress: QueryVectorizationProgress) => {
        // 调用原始回调
        this.config.onQueryVectorizationProgress?.(progress);
        // 同时保存数据
        if (progress.tokenization) {
          tokenizationData = progress.tokenization;
        }
        if (progress.embedding) {
          embeddingData = progress.embedding;
        }
      };
      
      // 执行检索（带进度回调和主 Trace ID）
      const retrievalDetails = await this.vectorStore.similaritySearchWithDetails(
        question,
        topK,
        similarityThreshold,
        traceId,  // 传入主 Trace ID，让 BPE tokenizer 创建 Span 而不是独立的 Trace
        wrappedProgressCallback
      );

      // 更新检索 Span
      this.observabilityEngine.updateObservation(retrievalSpanId, {
        output: {
          totalDocuments: retrievalDetails.totalDocuments,
          matchedDocuments: retrievalDetails.searchResults.length,
          searchTime: retrievalDetails.searchTime,
        },
        endTime: new Date(),
      });

      // 发送检索详情
      this.config.onRetrievalDetails?.(retrievalDetails);

      // 构造上下文
      const context = retrievalDetails.searchResults
        .map((result, index) =>
          `[文档${index + 1}] (相似度: ${result.similarity.toFixed(4)}) (来源: ${result.document.metadata?.source || 'Unknown'})\n${result.document.pageContent}`
        )
        .join("\n---\n");

      // LLM 生成 Generation
      const generationId = this.observabilityEngine.createGeneration({
        traceId,
        name: 'Answer Generation',
        input: { question, context },
        model: this.llm.model,
        modelParameters: { temperature: 0 },
        metadata: { stage: 'generation' }
      });

      // 构造 Prompt
      const prompt = ChatPromptTemplate.fromTemplate(`
        你是一个专业的知识库助手。请根据下方提供的上下文信息来回答用户的问题。
        
        【上下文信息】：
        {context}
        
        【用户问题】：
        {question}
        
        如果上下文信息中不包含答案，请礼貌地说明你不知道，不要胡乱编造。
        请使用中文回答，回答要简洁明了。
      `);

      const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
      const result = await chain.invoke({
        context: context,
        question: question,
      });

      // 更新 Generation
      this.observabilityEngine.updateObservation(generationId, {
        output: result,
        endTime: new Date(),
        usage: {
          promptTokens: Math.ceil(context.length / 4),
          completionTokens: Math.ceil(result.length / 4),
          totalTokens: Math.ceil((context.length + result.length) / 4)
        }
      });

      // 完成 Trace
      this.observabilityEngine.updateTrace(traceId, {
        output: { answer: result, context },
        status: 'SUCCESS',
        endTime: new Date(),
      });

      return {
        answer: result,
        retrievalDetails,
        context,
        traceId,
        queryAnalysis: {
          tokenization: tokenizationData,
          embedding: embeddingData
        }
      };

    } catch (error) {
      this.observabilityEngine.updateTrace(traceId, {
        status: 'ERROR',
        endTime: new Date(),
        metadata: { error: error instanceof Error ? error.message : String(error) }
      });
      throw error;
    }
  }

  // 可观测性方法
  getObservabilityData() {
    return {
      traces: this.observabilityEngine.getAllTraces(),
      stats: this.observabilityEngine.getTraceStats()
    };
  }

  getTrace(traceId: string) {
    return this.observabilityEngine.getTrace(traceId);
  }

  addUserFeedback(traceId: string, score: number | boolean, comment?: string) {
    return this.observabilityEngine.addScore({
      traceId,
      name: 'user_feedback',
      value: score,
      source: 'USER',
      comment
    });
  }

  clearObservabilityData() {
    this.observabilityEngine.clear();
  }

  getStatus() {
    return {
      initialized: this.isInitialized,
      documentCount: this.vectorStore?.getDocumentCount() || 0,
      embeddingDimension: this.vectorStore?.getEmbeddingDimension() || 0
    };
  }
}