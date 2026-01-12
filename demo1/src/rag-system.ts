import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import fs from "fs";
import path from "path";
import { ObservabilityEngine, type Trace, type Observation } from "./observability";

// Token 信息接口
interface TokenInfo {
  token: string;
  tokenId: number;
  position: number;
  type: 'chinese' | 'english' | 'number' | 'punctuation' | 'special';
}

// 向量特征接口
interface VectorFeatures {
  techScore: number;
  businessScore: number;
  dailyScore: number;
  emotionScore: number;
  vectorMagnitude: number;
}

// 语义分析结果接口
interface SemanticAnalysis {
  context: string;
  semanticCategory: string;
  nearestConcepts: string[];
  confidence: number;
  vectorFeatures?: VectorFeatures;
}

// 增强的词元化器类（模拟 BPE 算法）
class EnhancedTokenizer {
  private readonly maxTokens = 8192;
  private readonly vocabulary: Map<string, number>;
  private readonly reverseVocab: Map<number, string>;
  
  constructor() {
    // 模拟词汇表（简化的 BPE 词汇表）
    this.vocabulary = new Map();
    this.reverseVocab = new Map();
    this.initializeVocabulary();
  }
  
  private initializeVocabulary() {
    // 更真实的词汇表，包含更多词汇和更合理的ID分配
    const vocabularyData = [
      // 特殊token (1-10)
      { token: '<|pad|>', id: 0 },
      { token: '<|unk|>', id: 1 },
      { token: '<|start|>', id: 2 },
      { token: '<|end|>', id: 3 },
      
      // 常用标点 (10-30)
      { token: '.', id: 10 }, { token: ',', id: 11 }, { token: '?', id: 12 },
      { token: '!', id: 13 }, { token: ':', id: 14 }, { token: ';', id: 15 },
      { token: '(', id: 16 }, { token: ')', id: 17 }, { token: '"', id: 18 },
      { token: "'", id: 19 }, { token: '-', id: 20 }, { token: '/', id: 21 },
      
      // 常用英文词汇 (100-200)
      { token: 'the', id: 100 }, { token: 'and', id: 101 }, { token: 'or', id: 102 },
      { token: 'in', id: 103 }, { token: 'on', id: 104 }, { token: 'at', id: 105 },
      { token: 'to', id: 106 }, { token: 'for', id: 107 }, { token: 'with', id: 108 },
      { token: 'is', id: 109 }, { token: 'are', id: 110 }, { token: 'was', id: 111 },
      { token: 'be', id: 112 }, { token: 'have', id: 113 }, { token: 'has', id: 114 },
      
      // 技术相关英文词汇 (200-300)
      { token: 'AI', id: 200 }, { token: 'machine', id: 201 }, { token: 'learn', id: 202 },
      { token: 'ing', id: 203 }, { token: 'deep', id: 204 }, { token: 'neural', id: 205 },
      { token: 'network', id: 206 }, { token: 'data', id: 207 }, { token: 'algorithm', id: 208 },
      { token: 'model', id: 209 }, { token: 'train', id: 210 }, { token: 'test', id: 211 },
      { token: 'app', id: 212 }, { token: 'le', id: 213 }, { token: 'phone', id: 214 },
      { token: 'tech', id: 215 }, { token: 'comp', id: 216 }, { token: 'any', id: 217 },
      
      // 常用中文字符 (1000-1200)
      { token: '的', id: 1000 }, { token: '是', id: 1001 }, { token: '在', id: 1002 },
      { token: '有', id: 1003 }, { token: '和', id: 1004 }, { token: '与', id: 1005 },
      { token: '或', id: 1006 }, { token: '但', id: 1007 }, { token: '而', id: 1008 },
      { token: '了', id: 1009 }, { token: '着', id: 1010 }, { token: '过', id: 1011 },
      
      // 技术相关中文词汇 (2000-2200)
      { token: '智', id: 2000 }, { token: '能', id: 2001 }, { token: '人', id: 2002 },
      { token: '工', id: 2003 }, { token: '智能', id: 2004 }, { token: '人工', id: 2005 },
      { token: '人工智能', id: 2006 }, { token: '机', id: 2007 }, { token: '器', id: 2008 },
      { token: '机器', id: 2009 }, { token: '学', id: 2010 }, { token: '习', id: 2011 },
      { token: '学习', id: 2012 }, { token: '机器学习', id: 2013 },
      { token: '深', id: 2014 }, { token: '度', id: 2015 }, { token: '深度', id: 2016 },
      { token: '深度学习', id: 2017 }, { token: '神', id: 2018 }, { token: '经', id: 2019 },
      { token: '神经', id: 2020 }, { token: '网', id: 2021 }, { token: '络', id: 2022 },
      { token: '网络', id: 2023 }, { token: '神经网络', id: 2024 },
      
      // 其他常用词汇 (3000-3200)
      { token: '手', id: 3000 }, { token: '机', id: 3001 }, { token: '手机', id: 3002 },
      { token: '苹', id: 3003 }, { token: '果', id: 3004 }, { token: '苹果', id: 3005 },
      { token: '公', id: 3006 }, { token: '司', id: 3007 }, { token: '公司', id: 3008 },
      { token: '技', id: 3009 }, { token: '术', id: 3010 }, { token: '技术', id: 3011 },
      { token: '系', id: 3012 }, { token: '统', id: 3013 }, { token: '系统', id: 3014 },
      { token: '数', id: 3015 }, { token: '据', id: 3016 }, { token: '数据', id: 3017 },
      { token: '分', id: 3018 }, { token: '析', id: 3019 }, { token: '分析', id: 3020 },
      { token: '算', id: 3021 }, { token: '法', id: 3022 }, { token: '算法', id: 3023 },
      
      // 问题相关词汇 (4000-4100)
      { token: '什', id: 4000 }, { token: '么', id: 4001 }, { token: '什么', id: 4002 },
      { token: '怎', id: 4003 }, { token: '样', id: 4004 }, { token: '怎样', id: 4005 },
      { token: '如', id: 4006 }, { token: '何', id: 4007 }, { token: '如何', id: 4008 },
      { token: '为', id: 4009 }, { token: '为什么', id: 4010 }, { token: '哪', id: 4011 },
      { token: '里', id: 4012 }, { token: '哪里', id: 4013 }, { token: '谁', id: 4014 },
      { token: '哪个', id: 4015 }, { token: '多少', id: 4016 }, { token: '几', id: 4017 },
    ];
    
    vocabularyData.forEach(({ token, id }) => {
      this.vocabulary.set(token, id);
      this.reverseVocab.set(id, token);
    });
  }
  
  tokenize(text: string): TokenizationResult {
    const startTime = Date.now();
    const originalText = text;
    
    // 1. 预处理
    let processedText = text.trim();
    
    // 2. 模拟 BPE 分词过程
    const tokenInfos: TokenInfo[] = [];
    let position = 0;
    
    // 改进的分词逻辑（更真实的 BPE 模拟）
    let i = 0;
    
    while (i < processedText.length) {
      let bestMatch = '';
      let bestMatchId = 1; // 默认为 <|unk|>
      
      // 尝试匹配最长的词汇（从长到短）
      for (let len = Math.min(8, processedText.length - i); len >= 1; len--) {
        const candidate = processedText.substring(i, i + len);
        if (this.vocabulary.has(candidate)) {
          bestMatch = candidate;
          bestMatchId = this.vocabulary.get(candidate)!;
          break;
        }
      }
      
      // 如果没有匹配到任何词汇，使用单个字符
      if (!bestMatch) {
        bestMatch = processedText[i];
        // 为未知字符生成一个基于字符码的ID
        bestMatchId = this.vocabulary.get(bestMatch) || (5000 + bestMatch.charCodeAt(0) % 1000);
      }
      
      // 确定token类型
      let tokenType: TokenInfo['type'] = 'special';
      if (/[\u4e00-\u9fa5]/.test(bestMatch)) {
        tokenType = 'chinese';
      } else if (/[a-zA-Z]/.test(bestMatch)) {
        tokenType = 'english';
      } else if (/[0-9]/.test(bestMatch)) {
        tokenType = 'number';
      } else if (/[.,!?;:]/.test(bestMatch)) {
        tokenType = 'punctuation';
      }
      
      tokenInfos.push({
        token: bestMatch,
        tokenId: bestMatchId,
        position: position++,
        type: tokenType
      });
      
      i += bestMatch.length;
    }
    
    const processingTime = Date.now() - startTime;
    
    return {
      originalText,
      tokens: tokenInfos.map(t => t.token),
      tokenIds: tokenInfos.map(t => t.tokenId),
      tokenInfos: tokenInfos,
      tokenCount: tokenInfos.length,
      processedText,
      processingTime
    };
  }
  
  // 基于向量的语义分析
  analyzeSemantics(text: string, embedding: number[]): SemanticAnalysis {
    // 基于向量的特征分析
    const vectorFeatures = this.analyzeVectorFeatures(embedding);
    
    // 基于文本内容和向量特征的综合分析
    const textFeatures = this.analyzeTextFeatures(text);
    
    // 综合分析结果
    let context = '';
    let semanticCategory = '';
    let nearestConcepts: string[] = [];
    let confidence = 0;
    
    // 基于向量特征的主要维度分析
    if (vectorFeatures.techScore > 0.6) {
      if (textFeatures.hasAI || textFeatures.hasTech) {
        context = '人工智能/技术语境';
        semanticCategory = 'AI技术';
        nearestConcepts = ['人工智能', '机器学习', '深度学习', '算法', '神经网络'];
        confidence = 0.85 + vectorFeatures.techScore * 0.1;
      } else if (textFeatures.hasDevice) {
        context = '电子设备语境';
        semanticCategory = '智能设备';
        nearestConcepts = ['智能手机', '电子设备', '移动技术', '通讯'];
        confidence = 0.80 + vectorFeatures.techScore * 0.1;
      } else {
        context = '技术相关语境';
        semanticCategory = '技术';
        nearestConcepts = ['技术', '创新', '系统', '开发'];
        confidence = 0.75 + vectorFeatures.techScore * 0.1;
      }
    } else if (vectorFeatures.businessScore > 0.5) {
      context = '商业/公司语境';
      semanticCategory = '商业';
      nearestConcepts = ['公司', '企业', '商业', '市场', '产品'];
      confidence = 0.70 + vectorFeatures.businessScore * 0.15;
    } else if (vectorFeatures.dailyScore > 0.5) {
      if (textFeatures.hasFood) {
        context = '日常生活/食物语境';
        semanticCategory = '生活';
        nearestConcepts = ['食物', '生活', '日常', '健康'];
        confidence = 0.75 + vectorFeatures.dailyScore * 0.1;
      } else {
        context = '日常对话语境';
        semanticCategory = '日常';
        nearestConcepts = ['日常', '对话', '交流', '生活'];
        confidence = 0.65 + vectorFeatures.dailyScore * 0.1;
      }
    } else {
      context = '通用语境';
      semanticCategory = '一般';
      nearestConcepts = ['文本', '信息', '内容', '查询'];
      confidence = 0.60;
    }
    
    // 确保置信度在合理范围内
    confidence = Math.min(0.95, Math.max(0.50, confidence));
    
    return {
      context,
      semanticCategory,
      nearestConcepts,
      confidence,
      vectorFeatures // 添加向量特征信息
    };
  }
  
  // 分析向量特征
  private analyzeVectorFeatures(embedding: number[]) {
    // 基于向量的不同维度计算特征分数
    const dim = embedding.length;
    
    // 技术相关特征（前1/4维度）
    const techDims = embedding.slice(0, Math.floor(dim / 4));
    const techScore = Math.abs(techDims.reduce((sum, val) => sum + val, 0)) / techDims.length;
    
    // 商业相关特征（中间1/4维度）
    const businessDims = embedding.slice(Math.floor(dim / 4), Math.floor(dim / 2));
    const businessScore = Math.abs(businessDims.reduce((sum, val) => sum + val, 0)) / businessDims.length;
    
    // 日常生活特征（后1/4维度）
    const dailyDims = embedding.slice(Math.floor(3 * dim / 4));
    const dailyScore = Math.abs(dailyDims.reduce((sum, val) => sum + val, 0)) / dailyDims.length;
    
    // 情感特征（特定维度范围）
    const emotionDims = embedding.slice(Math.floor(dim / 2), Math.floor(3 * dim / 4));
    const emotionScore = emotionDims.reduce((sum, val) => sum + val, 0) / emotionDims.length;
    
    return {
      techScore: Math.min(1, techScore),
      businessScore: Math.min(1, businessScore),
      dailyScore: Math.min(1, dailyScore),
      emotionScore: Math.tanh(emotionScore), // 使用tanh归一化到[-1,1]
      vectorMagnitude: Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0))
    };
  }
  
  // 分析文本特征
  private analyzeTextFeatures(text: string) {
    const lowerText = text.toLowerCase();
    
    return {
      hasAI: /智能|AI|人工智能|机器学习|深度学习|神经网络|算法/.test(text),
      hasTech: /技术|系统|开发|编程|代码|软件|硬件/.test(text),
      hasDevice: /手机|电脑|设备|iPhone|Android|电子/.test(text),
      hasFood: /吃|食物|水果|苹果|食品|美食|营养/.test(text) && !/公司|iPhone|手机/.test(text),
      hasBusiness: /公司|企业|商业|市场|产品|服务|客户/.test(text),
      hasQuestion: /什么|怎么|如何|为什么|哪里|谁|多少/.test(text)
    };
  }
  
  getModelInfo() {
    return {
      name: "Enhanced BPE Tokenizer",
      maxTokens: this.maxTokens,
      vocabularySize: this.vocabulary.size
    };
  }
}

// 向量化进度回调接口
interface VectorizationProgress {
  current: number;
  total: number;
  document: string;
  embedding?: number[];
  timestamp: string;
}

// 相似度搜索结果接口
interface SimilaritySearchResult {
  document: Document;
  similarity: number;
  index: number;
  embedding: number[];
}

// 文本处理步骤接口
interface TextProcessingStep {
  step: 'tokenization' | 'preprocessing' | 'embedding' | 'completed';
  status: 'starting' | 'processing' | 'completed';
  data?: any;
  timestamp: string;
}

// 词元化结果接口
interface TokenizationResult {
  originalText: string;
  tokens: string[];
  tokenIds: number[];
  tokenInfos: TokenInfo[];
  tokenCount: number;
  processedText: string;
  processingTime: number;
}

// 嵌入过程接口
interface EmbeddingProcess {
  inputTokens: string[];
  inputTokenIds: number[];
  embeddingDimension: number;
  embedding: number[];
  semanticAnalysis: SemanticAnalysis;
  processingTime: number;
  modelInfo: {
    name: string;
    maxTokens: number;
    vocabularySize?: number;
  };
}

// 查询向量化进度接口（增强版）
interface QueryVectorizationProgress {
  query: string;
  status: 'starting' | 'tokenizing' | 'preprocessing' | 'embedding' | 'completed';
  tokenization?: TokenizationResult;
  embedding?: EmbeddingProcess;
  totalTime?: number;
  timestamp: string;
}

// 检索过程详情接口
interface RetrievalDetails {
  query: string;
  queryEmbedding: number[];
  searchResults: SimilaritySearchResult[];
  threshold: number;
  topK: number;
  totalDocuments: number;
  searchTime: number;
  queryVectorizationTime: number;
  timestamp: string;
}

// 增强的内存向量存储实现
class SimpleMemoryVectorStore {
  private documents: Document[] = [];
  private embeddings: number[][] = [];
  private onProgress?: (progress: VectorizationProgress) => void;
  private onQueryProgress?: (progress: QueryVectorizationProgress) => void;
  private tokenizer: EnhancedTokenizer;
  
  constructor(
    private embeddingModel: OllamaEmbeddings,
    progressCallback?: (progress: VectorizationProgress) => void,
    queryProgressCallback?: (progress: QueryVectorizationProgress) => void
  ) {
    this.onProgress = progressCallback;
    this.onQueryProgress = queryProgressCallback;
    this.tokenizer = new EnhancedTokenizer();
  }
  
  async addDocuments(docs: Document[]) {
    const startIndex = this.documents.length;
    this.documents.push(...docs);
    
    // 为每个文档生成嵌入，带进度回调
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const currentIndex = startIndex + i;
      
      // 发送进度更新
      if (this.onProgress) {
        this.onProgress({
          current: i + 1,
          total: docs.length,
          document: doc.metadata?.source || `Document ${currentIndex + 1}`,
          timestamp: new Date().toISOString()
        });
      }
      
      console.log(`正在向量化文档 ${i + 1}/${docs.length}: ${doc.metadata?.source || 'Unknown'}`);
      
      const embedding = await this.embeddingModel.embedQuery(doc.pageContent);
      this.embeddings.push(embedding);
      
      // 发送完成的进度更新（包含嵌入向量）
      if (this.onProgress) {
        this.onProgress({
          current: i + 1,
          total: docs.length,
          document: doc.metadata?.source || `Document ${currentIndex + 1}`,
          embedding: embedding,
          timestamp: new Date().toISOString()
        });
      }
      
      console.log(`文档 ${i + 1} 向量化完成，向量维度: ${embedding.length}`);
    }
  }
  
  async similaritySearchWithDetails(
    query: string, 
    k: number = 3, 
    threshold: number = 0.0
  ): Promise<RetrievalDetails> {
    const startTime = Date.now();
    
    if (this.documents.length === 0) {
      return {
        query,
        queryEmbedding: [],
        searchResults: [],
        threshold,
        topK: k,
        totalDocuments: 0,
        searchTime: 0,
        timestamp: new Date().toISOString()
      };
    }
    
    console.log(`开始相似度搜索: "${query}"`);
    console.log(`参数: Top-K=${k}, 阈值=${threshold}, 总文档数=${this.documents.length}`);
    
    // 生成查询的嵌入（带详细进度监控）
    const queryVectorizationStart = Date.now();
    
    // 1. 发送查询向量化开始事件
    if (this.onQueryProgress) {
      this.onQueryProgress({
        query,
        status: 'starting',
        timestamp: new Date().toISOString()
      });
    }
    
    // 2. 词元化阶段
    if (this.onQueryProgress) {
      this.onQueryProgress({
        query,
        status: 'tokenizing',
        timestamp: new Date().toISOString()
      });
    }
    
    const tokenizationResult = this.tokenizer.tokenize(query);
    console.log(`查询词元化完成: ${tokenizationResult.tokenCount} 个词元, 耗时: ${tokenizationResult.processingTime}ms`);
    console.log(`词元: [${tokenizationResult.tokens.join(', ')}]`);
    
    // 3. 预处理阶段
    if (this.onQueryProgress) {
      this.onQueryProgress({
        query,
        status: 'preprocessing',
        tokenization: tokenizationResult,
        timestamp: new Date().toISOString()
      });
    }
    
    // 模拟预处理延迟
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // 4. 嵌入阶段
    if (this.onQueryProgress) {
      this.onQueryProgress({
        query,
        status: 'embedding',
        tokenization: tokenizationResult,
        timestamp: new Date().toISOString()
      });
    }
    
    const embeddingStart = Date.now();
    const queryEmbedding = await this.embeddingModel.embedQuery(query);
    const embeddingTime = Date.now() - embeddingStart;
    const queryVectorizationTime = Date.now() - queryVectorizationStart;
    
    console.log(`查询嵌入完成，维度: ${queryEmbedding.length}, 嵌入耗时: ${embeddingTime}ms, 总耗时: ${queryVectorizationTime}ms`);
    
    // 5. 语义分析
    const semanticAnalysis = this.tokenizer.analyzeSemantics(query, queryEmbedding);
    console.log(`语义分析完成: ${semanticAnalysis.context} (置信度: ${semanticAnalysis.confidence})`);
    
    // 6. 完成事件
    const embeddingProcess: EmbeddingProcess = {
      inputTokens: tokenizationResult.tokens,
      inputTokenIds: tokenizationResult.tokenIds,
      embeddingDimension: queryEmbedding.length,
      embedding: queryEmbedding,
      semanticAnalysis: semanticAnalysis,
      processingTime: embeddingTime,
      modelInfo: this.tokenizer.getModelInfo()
    };
    
    if (this.onQueryProgress) {
      this.onQueryProgress({
        query,
        status: 'completed',
        tokenization: tokenizationResult,
        embedding: embeddingProcess,
        totalTime: queryVectorizationTime,
        timestamp: new Date().toISOString()
      });
    }
    
    // 计算所有文档的相似度
    const similarities: SimilaritySearchResult[] = this.embeddings.map((embedding, index) => {
      const similarity = this.cosineSimilarity(queryEmbedding, embedding);
      return {
        document: this.documents[index],
        similarity,
        index,
        embedding
      };
    });
    
    console.log(`相似度计算完成，所有相似度分数:`);
    similarities.forEach((result, i) => {
      console.log(`  文档 ${i + 1} (${result.document.metadata?.source}): ${result.similarity.toFixed(4)}`);
    });
    
    // 按相似度排序
    similarities.sort((a, b) => b.similarity - a.similarity);
    
    // 应用阈值过滤
    const filteredResults = similarities.filter(result => result.similarity >= threshold);
    console.log(`阈值过滤后剩余文档数: ${filteredResults.length}`);
    
    // 取前K个结果
    const topKResults = filteredResults.slice(0, k);
    console.log(`最终返回文档数: ${topKResults.length}`);
    
    const searchTime = Date.now() - startTime;
    console.log(`搜索完成，耗时: ${searchTime}ms`);
    
    return {
      query,
      queryEmbedding,
      searchResults: topKResults,
      threshold,
      topK: k,
      totalDocuments: this.documents.length,
      searchTime,
      queryVectorizationTime,
      timestamp: new Date().toISOString()
    };
  }
  
  async similaritySearch(query: string, k: number = 3): Promise<Document[]> {
    const details = await this.similaritySearchWithDetails(query, k);
    return details.searchResults.map(result => result.document);
  }
  
  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }
  
  getDocumentCount(): number {
    return this.documents.length;
  }
  
  getEmbeddingDimension(): number {
    return this.embeddings.length > 0 ? this.embeddings[0].length : 0;
  }
  
  getAllDocuments(): Document[] {
    return [...this.documents];
  }
  
  clear() {
    this.documents = [];
    this.embeddings = [];
  }
}

export class LocalRAGSystem {
  private llm: Ollama;
  private embeddings: OllamaEmbeddings;
  private vectorStore!: SimpleMemoryVectorStore;
  private isInitialized = false;
  private onVectorizationProgress?: (progress: VectorizationProgress) => void;
  private onRetrievalDetails?: (details: RetrievalDetails) => void;
  private onQueryVectorizationProgress?: (progress: QueryVectorizationProgress) => void;
  private observabilityEngine: ObservabilityEngine;
  private onTraceUpdate?: (trace: Trace) => void;

  constructor(
    private config: {
      ollamaBaseUrl?: string;
      llmModel?: string;
      embeddingModel?: string;
      docsPath?: string;
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
      onVectorizationProgress,
      onRetrievalDetails,
      onQueryVectorizationProgress,
      onTraceUpdate,
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

    this.onVectorizationProgress = onVectorizationProgress;
    this.onRetrievalDetails = onRetrievalDetails;
    this.onQueryVectorizationProgress = onQueryVectorizationProgress;
    this.onTraceUpdate = onTraceUpdate;
    
    // 初始化可观测性引擎
    this.observabilityEngine = new ObservabilityEngine({
      onTraceUpdate: this.onTraceUpdate,
      onObservationUpdate: (observation) => {
        console.log('Observation updated:', observation.name);
      },
      onScoreUpdate: (score) => {
        console.log('Score updated:', score.name, score.value);
      }
    });

    this.vectorStore = new SimpleMemoryVectorStore(
      this.embeddings, 
      this.onVectorizationProgress,
      this.onQueryVectorizationProgress
    );
  }

  /**
   * 初始化数据库，加载文档
   */
  async initializeDatabase(docsPath: string = "./data"): Promise<void> {
    console.log("--- 正在初始化向量数据库 ---");

    // 检查数据目录是否存在
    if (!fs.existsSync(docsPath)) {
      console.log(`数据目录 ${docsPath} 不存在，创建示例文档...`);
      fs.mkdirSync(docsPath, { recursive: true });
      
      // 创建示例文档
      const sampleDoc = `
这是一个示例知识库文档。

核心技术架构：
- 使用 LangChain 构建 RAG 系统
- Ollama 作为本地 LLM 服务
- 向量数据库存储文档嵌入
- 支持多种文档格式

项目负责人：
- 姓名：张三
- 邮箱：zhangsan@example.com
- 电话：138-0000-0000

技术特点：
1. 本地部署，数据安全
2. 支持中文问答
3. 可扩展的文档格式支持
4. 高效的向量检索

系统功能：
- 文档上传和处理
- 智能问答
- 相似度搜索
- Web 界面交互
      `;
      
      fs.writeFileSync(path.join(docsPath, "sample.txt"), sampleDoc, "utf8");
      console.log("已创建示例文档");
    }

    await this.loadDocumentsFromDirectory(docsPath);
    this.isInitialized = true;
    console.log("向量数据库初始化完成。");
  }

  /**
   * 从目录加载文档
   */
  async loadDocumentsFromDirectory(docsPath: string): Promise<void> {
    const documents: Document[] = [];
    
    // 加载文本文件
    const files = fs.readdirSync(docsPath).filter(file => file.endsWith('.txt'));
    
    for (const file of files) {
      const filePath = path.join(docsPath, file);
      const content = fs.readFileSync(filePath, "utf8");
      documents.push(new Document({
        pageContent: content,
        metadata: { source: file, path: filePath }
      }));
    }

    console.log(`已加载 ${documents.length} 个文档`);

    if (documents.length > 0) {
      // 文本切分
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 600,
        chunkOverlap: 100,
      });

      const splitDocs = await splitter.splitDocuments(documents);
      console.log(`切分为 ${splitDocs.length} 个文本块`);

      // 添加到向量存储
      await this.vectorStore.addDocuments(splitDocs);
    }
  }

  /**
   * 添加单个文档
   */
  async addDocument(content: string, metadata: any = {}): Promise<void> {
    const doc = new Document({
      pageContent: content,
      metadata
    });

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600,
      chunkOverlap: 100,
    });

    const splitDocs = await splitter.splitDocuments([doc]);
    await this.vectorStore.addDocuments(splitDocs);
  }

  /**
   * 保存上传的文件到数据目录
   */
  async saveUploadedFile(filename: string, content: string, docsPath: string = "./data"): Promise<void> {
    if (!fs.existsSync(docsPath)) {
      fs.mkdirSync(docsPath, { recursive: true });
    }
    
    const filePath = path.join(docsPath, filename);
    fs.writeFileSync(filePath, content, "utf8");
    
    // 重新加载文档
    await this.addDocument(content, { source: filename, path: filePath });
  }

  /**
   * 执行问答（带详细检索信息）
   */
  async askWithDetails(
    question: string, 
    options: {
      topK?: number;
      similarityThreshold?: number;
      userId?: string;
      sessionId?: string;
    } = {}
  ): Promise<{
    answer: string;
    retrievalDetails: RetrievalDetails;
    context: string;
    traceId: string;
  }> {
    if (!this.isInitialized) {
      throw new Error("RAG 系统尚未初始化，请先调用 initializeDatabase()");
    }

    const { topK = 3, similarityThreshold = 0.0, userId, sessionId } = options;

    // 🎯 创建 Trace（Langfuse 风格）
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

    console.log(`🔍 [Trace ${traceId}] 用户问题: ${question}`);
    console.log(`📊 检索参数: Top-K=${topK}, 相似度阈值=${similarityThreshold}`);

    try {
      // 🔍 阶段1: 查询理解与向量化 (Span)
      const querySpanId = this.observabilityEngine.createSpan({
        traceId,
        name: 'Query Understanding & Vectorization',
        input: { question },
        metadata: { stage: 'query_processing' }
      });

      // 🔍 阶段2: 向量检索 (Span)
      const retrievalSpanId = this.observabilityEngine.createSpan({
        traceId,
        name: 'Vector Retrieval',
        parentObservationId: querySpanId,
        input: { question, topK, similarityThreshold },
        metadata: { stage: 'retrieval' }
      });

      // 执行检索
      const retrievalDetails = await this.vectorStore.similaritySearchWithDetails(
        question, 
        topK, 
        similarityThreshold,
        (progress) => {
          // 查询向量化进度事件
          this.observabilityEngine.createEvent({
            traceId,
            parentObservationId: querySpanId,
            name: 'Query Vectorization Progress',
            input: { progress },
            metadata: { stage: progress.status }
          });
          
          if (this.onQueryVectorizationProgress) {
            this.onQueryVectorizationProgress(progress);
          }
        }
      );

      // 更新检索 Span
      this.observabilityEngine.updateObservation(retrievalSpanId, {
        output: {
          totalDocuments: retrievalDetails.totalDocuments,
          matchedDocuments: retrievalDetails.searchResults.length,
          searchTime: retrievalDetails.searchTime,
          topResults: retrievalDetails.searchResults.map(r => ({
            similarity: r.similarity,
            source: r.document.metadata?.source,
            contentPreview: r.document.pageContent.substring(0, 100) + '...'
          }))
        },
        endTime: new Date(),
        metadata: { 
          stage: 'retrieval',
          performance: {
            searchTime: retrievalDetails.searchTime,
            documentsScanned: retrievalDetails.totalDocuments,
            documentsMatched: retrievalDetails.searchResults.length
          }
        }
      });

      // 发送检索详情到回调
      if (this.onRetrievalDetails) {
        this.onRetrievalDetails(retrievalDetails);
      }

      // 构造上下文
      const context = retrievalDetails.searchResults
        .map((result, index) => 
          `[文档${index + 1}] (相似度: ${result.similarity.toFixed(4)}) (来源: ${result.document.metadata?.source || 'Unknown'})\n${result.document.pageContent}`
        )
        .join("\n---\n");

      console.log(`📚 检索到 ${retrievalDetails.searchResults.length} 个相关文档`);

      // 🤖 阶段3: LLM 生成 (Generation)
      const generationId = this.observabilityEngine.createGeneration({
        traceId,
        name: 'Answer Generation',
        input: { question, context },
        model: this.llm.model,
        modelParameters: {
          temperature: 0,
        },
        metadata: { stage: 'generation' }
      });

      // 构造 Prompt 模板
      const prompt = ChatPromptTemplate.fromTemplate(`
        你是一个专业的知识库助手。请根据下方提供的上下文信息来回答用户的问题。
        
        【上下文信息】：
        {context}
        
        【用户问题】：
        {question}
        
        如果上下文信息中不包含答案，请礼貌地说明你不知道，不要胡乱编造。
        请使用中文回答，回答要简洁明了。
      `);

      // 运行链式调用
      const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
      const startTime = Date.now();
      
      const result = await chain.invoke({
        context: context,
        question: question,
      });

      const endTime = Date.now();
      const generationTime = endTime - startTime;

      // 更新 Generation
      this.observabilityEngine.updateObservation(generationId, {
        output: result,
        endTime: new Date(),
        usage: {
          promptTokens: Math.ceil(context.length / 4), // 估算
          completionTokens: Math.ceil(result.length / 4), // 估算
          totalTokens: Math.ceil((context.length + result.length) / 4)
        },
        metadata: {
          stage: 'generation',
          performance: {
            generationTime,
            contextLength: context.length,
            responseLength: result.length
          }
        }
      });

      // 🎯 完成 Trace
      this.observabilityEngine.updateTrace(traceId, {
        output: { answer: result, context },
        status: 'SUCCESS',
        endTime: new Date(),
        metadata: {
          totalTime: Date.now() - new Date(this.observabilityEngine.getTrace(traceId)!.startTime).getTime(),
          retrievedDocuments: retrievalDetails.searchResults.length,
          performance: {
            queryTime: retrievalDetails.queryVectorizationTime,
            searchTime: retrievalDetails.searchTime,
            generationTime
          }
        }
      });

      console.log(`🤖 AI 回答: \n${result}\n`);
      
      return {
        answer: result,
        retrievalDetails,
        context,
        traceId
      };

    } catch (error) {
      // 错误处理
      this.observabilityEngine.updateTrace(traceId, {
        status: 'ERROR',
        endTime: new Date(),
        metadata: { error: error instanceof Error ? error.message : String(error) }
      });
      
      console.error(`❌ [Trace ${traceId}] 错误:`, error);
      throw error;
    }
  }

  /**
   * 执行问答（简化版，保持向后兼容）
   */
  async ask(question: string): Promise<string> {
    const result = await this.askWithDetails(question);
    return result.answer;
  }

  /**
   * 获取系统状态
   */
  getStatus(): { 
    initialized: boolean; 
    documentCount: number;
    embeddingDimension: number;
  } {
    return {
      initialized: this.isInitialized,
      documentCount: this.vectorStore.getDocumentCount(),
      embeddingDimension: this.vectorStore.getEmbeddingDimension()
    };
  }

  /**
   * 获取可观测性数据
   */
  getObservabilityData() {
    return {
      traces: this.observabilityEngine.getAllTraces(),
      stats: this.observabilityEngine.getTraceStats()
    };
  }

  /**
   * 获取特定 Trace
   */
  getTrace(traceId: string) {
    return this.observabilityEngine.getTrace(traceId);
  }

  /**
   * 添加用户反馈评分
   */
  addUserFeedback(traceId: string, score: number | boolean, comment?: string) {
    return this.observabilityEngine.addScore({
      traceId,
      name: 'user_feedback',
      value: score,
      source: 'USER',
      comment
    });
  }

  /**
   * 清除可观测性数据
   */
  clearObservabilityData() {
    this.observabilityEngine.clear();
  }

  /**
   * 获取所有文档信息
   */
  getAllDocuments(): Document[] {
    return this.vectorStore.getAllDocuments();
  }

  /**
   * 设置进度回调
   */
  setProgressCallbacks(
    onVectorizationProgress?: (progress: VectorizationProgress) => void,
    onRetrievalDetails?: (details: RetrievalDetails) => void
  ) {
    this.onVectorizationProgress = onVectorizationProgress;
    this.onRetrievalDetails = onRetrievalDetails;
    
    // 重新创建向量存储以使用新的回调
    const oldDocuments = this.vectorStore.getAllDocuments();
    this.vectorStore = new SimpleMemoryVectorStore(
      this.embeddings,
      this.onVectorizationProgress
    );
    
    // 如果有旧文档，重新添加它们
    if (oldDocuments.length > 0) {
      this.vectorStore.addDocuments(oldDocuments);
    }
  }

  /**
   * 清空向量存储
   */
  clearDatabase(): void {
    this.vectorStore.clear();
    this.isInitialized = false;
  }
}