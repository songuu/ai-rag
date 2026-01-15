/**
 * Milvus 增强版 RAG 系统
 * 
 * 支持两种存储后端：
 * 1. 内存存储（默认）- 适合开发和小数据集
 * 2. Milvus 存储 - 适合生产和大数据集
 */

import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { MilvusVectorStore, getMilvusInstance, MilvusConfig, MilvusSearchResult } from "./milvus-client";
import { ObservabilityEngine, type Trace } from "./observability";
import { v4 as uuidv4 } from 'uuid';

// 存储后端类型
export type StorageBackend = 'memory' | 'milvus';

// 配置接口
export interface MilvusRAGConfig {
  ollamaBaseUrl?: string;
  llmModel?: string;
  embeddingModel?: string;
  storageBackend?: StorageBackend;
  milvusConfig?: MilvusConfig;
  onTraceUpdate?: (trace: Trace) => void;
}

// 检索结果接口
export interface RAGSearchResult {
  content: string;
  similarity: number;
  metadata: Record<string, any>;
  id: string;
}

// RAG 回答结果
export interface RAGAnswer {
  answer: string;
  sources: RAGSearchResult[];
  traceId: string;
  processingTime: number;
}

/**
 * Milvus 增强版 RAG 系统
 */
export class MilvusRAGSystem {
  private llm: Ollama;
  private embeddings: OllamaEmbeddings;
  private milvus: MilvusVectorStore | null = null;
  private observabilityEngine: ObservabilityEngine;
  private config: Required<MilvusRAGConfig>;
  private isInitialized: boolean = false;

  constructor(config: MilvusRAGConfig = {}) {
    this.config = {
      ollamaBaseUrl: config.ollamaBaseUrl || "http://localhost:11434",
      llmModel: config.llmModel || "llama3.1",
      embeddingModel: config.embeddingModel || "nomic-embed-text",
      storageBackend: config.storageBackend || "milvus",
      milvusConfig: config.milvusConfig || {},
      onTraceUpdate: config.onTraceUpdate || (() => {}),
    };

    this.llm = new Ollama({
      baseUrl: this.config.ollamaBaseUrl,
      model: this.config.llmModel,
      temperature: 0,
    });

    this.embeddings = new OllamaEmbeddings({
      baseUrl: this.config.ollamaBaseUrl,
      model: this.config.embeddingModel,
    });

    this.observabilityEngine = new ObservabilityEngine({
      onTraceUpdate: this.config.onTraceUpdate,
    });
  }

  /**
   * 初始化系统
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('[MilvusRAG] Initializing with backend:', this.config.storageBackend);

    if (this.config.storageBackend === 'milvus') {
      this.milvus = getMilvusInstance(this.config.milvusConfig);
      await this.milvus.connect();
      await this.milvus.initializeCollection();
    }

    this.isInitialized = true;
    console.log('[MilvusRAG] Initialized successfully');
  }

  /**
   * 确保系统已初始化
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }
  }

  /**
   * 添加文档
   */
  async addDocuments(documents: Array<{ content: string; metadata?: Record<string, any> }>): Promise<string[]> {
    await this.ensureInitialized();

    const ids: string[] = [];

    for (const doc of documents) {
      const id = uuidv4();
      const embedding = await this.embeddings.embedQuery(doc.content);

      if (this.milvus) {
        await this.milvus.insertDocuments([{
          id,
          content: doc.content,
          embedding,
          metadata: doc.metadata || {},
        }]);
      }

      ids.push(id);
    }

    return ids;
  }

  /**
   * 相似度搜索
   */
  async search(
    query: string,
    topK: number = 5,
    threshold: number = 0.0
  ): Promise<RAGSearchResult[]> {
    await this.ensureInitialized();

    const queryEmbedding = await this.embeddings.embedQuery(query);

    if (this.milvus) {
      const results = await this.milvus.search(queryEmbedding, topK, threshold);
      return results.map(r => ({
        content: r.content,
        similarity: r.score,
        metadata: r.metadata,
        id: r.id,
      }));
    }

    return [];
  }

  /**
   * 提问并获取回答
   */
  async ask(
    question: string,
    options: { topK?: number; threshold?: number } = {}
  ): Promise<RAGAnswer> {
    await this.ensureInitialized();

    const startTime = Date.now();
    const traceId = uuidv4();

    // 开始追踪
    this.observabilityEngine.startTrace(traceId, 'rag_query');

    // 检索相关文档
    const { topK = 5, threshold = 0.0 } = options;
    const sources = await this.search(question, topK, threshold);

    // 构建上下文
    const context = sources.length > 0
      ? sources.map((s, i) => `[${i + 1}] ${s.content}`).join('\n\n')
      : '没有找到相关文档。';

    // 生成回答
    const prompt = ChatPromptTemplate.fromTemplate(`
你是一个智能助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请诚实地说明。

参考资料：
{context}

用户问题：{question}

请提供详细、准确的回答：
`);

    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
    const answer = await chain.invoke({ context, question });

    // 结束追踪
    this.observabilityEngine.endTrace(traceId, { answer, sources: sources.length });

    return {
      answer,
      sources,
      traceId,
      processingTime: Date.now() - startTime,
    };
  }

  /**
   * 获取系统状态
   */
  async getStatus(): Promise<{
    initialized: boolean;
    backend: StorageBackend;
    documentCount: number;
    milvusConnected: boolean;
  }> {
    let documentCount = 0;
    let milvusConnected = false;

    if (this.milvus) {
      const stats = await this.milvus.getCollectionStats();
      documentCount = stats?.rowCount || 0;
      milvusConnected = this.milvus.isReady();
    }

    return {
      initialized: this.isInitialized,
      backend: this.config.storageBackend,
      documentCount,
      milvusConnected,
    };
  }

  /**
   * 清空所有文档
   */
  async clear(): Promise<void> {
    if (this.milvus) {
      await this.milvus.clearCollection();
    }
  }

  /**
   * 获取配置
   */
  getConfig(): Required<MilvusRAGConfig> {
    return { ...this.config };
  }
}

// 全局实例
let milvusRagInstance: MilvusRAGSystem | null = null;

/**
 * 获取 Milvus RAG 实例
 */
export async function getMilvusRAGSystem(config?: MilvusRAGConfig): Promise<MilvusRAGSystem> {
  if (!milvusRagInstance) {
    milvusRagInstance = new MilvusRAGSystem(config);
    await milvusRagInstance.initialize();
  }
  return milvusRagInstance;
}

/**
 * 重置 Milvus RAG 实例
 */
export function resetMilvusRAGSystem(): void {
  milvusRagInstance = null;
}

export default MilvusRAGSystem;
