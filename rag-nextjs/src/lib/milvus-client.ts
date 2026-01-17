/**
 * Milvus 向量数据库客户端管理器
 * 提供连接管理、集合操作、向量存储等功能
 */

import { MilvusClient, DataType, MetricType, InsertReq, SearchReq } from '@zilliz/milvus2-sdk-node';

// Milvus 配置接口
export interface MilvusConfig {
  address?: string;          // Milvus 服务地址 (如: localhost:19530)
  username?: string;         // 用户名（可选）
  password?: string;         // 密码（可选）
  ssl?: boolean;             // 是否使用 SSL
  database?: string;         // 数据库名（默认: default）
  collectionName?: string;   // 集合名称（默认: rag_documents）
  embeddingDimension?: number; // 向量维度（默认: 768）
  indexType?: 'IVF_FLAT' | 'IVF_SQ8' | 'IVF_PQ' | 'HNSW' | 'ANNOY' | 'FLAT'; // 索引类型
  metricType?: 'L2' | 'IP' | 'COSINE'; // 距离度量类型
}

// 文档接口
export interface MilvusDocument {
  id: string;
  content: string;
  embedding: number[];
  metadata: Record<string, any>;
}

// 搜索结果接口
export interface MilvusSearchResult {
  id: string;
  content: string;
  metadata: Record<string, any>;
  score: number;
  distance: number;
}

// 集合统计信息
export interface CollectionStats {
  name: string;
  rowCount: number;
  embeddingDimension: number;
  indexType: string;
  metricType: string;
  loaded: boolean;
}

// 默认配置
const DEFAULT_CONFIG: Required<MilvusConfig> = {
  address: 'localhost:19530',
  username: '',
  password: '',
  ssl: false,
  database: 'default',
  collectionName: 'rag_documents',
  embeddingDimension: 768,
  indexType: 'IVF_FLAT',
  metricType: 'COSINE'
};

/**
 * Milvus 向量存储类
 */
export class MilvusVectorStore {
  private client: MilvusClient | null = null;
  private config: Required<MilvusConfig>;
  private isConnected: boolean = false;
  private isInitialized: boolean = false;

  constructor(config: MilvusConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * 获取当前配置
   */
  getConfig(): Required<MilvusConfig> {
    return { ...this.config };
  }

  /**
   * 连接到 Milvus 服务
   */
  async connect(): Promise<void> {
    if (this.isConnected && this.client) {
      console.log('[Milvus] Already connected');
      return;
    }

    try {
      console.log(`[Milvus] Connecting to ${this.config.address}...`);
      
      this.client = new MilvusClient({
        address: this.config.address,
        username: this.config.username || undefined,
        password: this.config.password || undefined,
        ssl: this.config.ssl,
      });

      // 检查连接
      const health = await this.client.checkHealth();
      if (!health.isHealthy) {
        throw new Error('Milvus service is not healthy');
      }

      this.isConnected = true;
      console.log('[Milvus] Connected successfully');

      // 使用指定数据库
      if (this.config.database !== 'default') {
        await this.client.useDatabase({ db_name: this.config.database });
      }
    } catch (error) {
      this.isConnected = false;
      this.client = null;
      throw new Error(`Failed to connect to Milvus: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * 断开连接
   */
  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.closeConnection();
      this.client = null;
      this.isConnected = false;
      this.isInitialized = false;
      console.log('[Milvus] Disconnected');
    }
  }

  /**
   * 确保连接已建立
   */
  private async ensureConnected(): Promise<MilvusClient> {
    if (!this.isConnected || !this.client) {
      await this.connect();
    }
    return this.client!;
  }

  /**
   * 初始化集合（创建 Schema 和索引）
   */
  async initializeCollection(): Promise<void> {
    const client = await this.ensureConnected();
    const collectionName = this.config.collectionName;

    try {
      // 检查集合是否存在
      const hasCollection = await client.hasCollection({ collection_name: collectionName });

      if (hasCollection.value) {
        console.log(`[Milvus] Collection '${collectionName}' already exists`);
        
        // 加载集合到内存
        await this.loadCollection();
        this.isInitialized = true;
        return;
      }

      console.log(`[Milvus] Creating collection '${collectionName}'...`);

      // 创建集合
      await client.createCollection({
        collection_name: collectionName,
        fields: [
          {
            name: 'id',
            description: 'Primary key',
            data_type: DataType.VarChar,
            is_primary_key: true,
            max_length: 256,
          },
          {
            name: 'content',
            description: 'Document content',
            data_type: DataType.VarChar,
            max_length: 65535,
          },
          {
            name: 'embedding',
            description: 'Vector embedding',
            data_type: DataType.FloatVector,
            dim: this.config.embeddingDimension,
          },
          {
            name: 'source',
            description: 'Document source',
            data_type: DataType.VarChar,
            max_length: 1024,
          },
          {
            name: 'metadata_json',
            description: 'Metadata as JSON string',
            data_type: DataType.VarChar,
            max_length: 65535,
          },
          {
            name: 'created_at',
            description: 'Creation timestamp',
            data_type: DataType.Int64,
          }
        ],
      });

      console.log(`[Milvus] Collection '${collectionName}' created`);

      // 创建向量索引
      await this.createIndex();

      // 加载集合
      await this.loadCollection();

      this.isInitialized = true;
      console.log(`[Milvus] Collection '${collectionName}' initialized successfully`);
    } catch (error) {
      throw new Error(`Failed to initialize collection: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * 创建向量索引
   */
  private async createIndex(): Promise<void> {
    const client = await this.ensureConnected();
    const collectionName = this.config.collectionName;

    console.log(`[Milvus] Creating index for collection '${collectionName}'...`);

    // 根据索引类型设置参数
    let indexParams: any = {};
    switch (this.config.indexType) {
      case 'IVF_FLAT':
        indexParams = { nlist: 128 };
        break;
      case 'IVF_SQ8':
        indexParams = { nlist: 128 };
        break;
      case 'IVF_PQ':
        indexParams = { nlist: 128, m: 8, nbits: 8 };
        break;
      case 'HNSW':
        indexParams = { M: 16, efConstruction: 256 };
        break;
      case 'ANNOY':
        indexParams = { n_trees: 8 };
        break;
      case 'FLAT':
      default:
        indexParams = {};
    }

    await client.createIndex({
      collection_name: collectionName,
      field_name: 'embedding',
      index_type: this.config.indexType,
      metric_type: this.config.metricType as MetricType,
      params: indexParams,
    });

    console.log(`[Milvus] Index created: ${this.config.indexType} with ${this.config.metricType}`);
  }

  /**
   * 加载集合到内存
   */
  async loadCollection(): Promise<void> {
    const client = await this.ensureConnected();
    const collectionName = this.config.collectionName;

    try {
      const loadState = await client.getLoadState({ collection_name: collectionName });
      
      if (loadState.state !== 'LoadStateLoaded') {
        console.log(`[Milvus] Loading collection '${collectionName}'...`);
        await client.loadCollection({ collection_name: collectionName });
        console.log(`[Milvus] Collection '${collectionName}' loaded`);
      }
    } catch (error) {
      console.warn(`[Milvus] Warning loading collection: ${error}`);
    }
  }

  /**
   * 释放集合
   */
  async releaseCollection(): Promise<void> {
    const client = await this.ensureConnected();
    await client.releaseCollection({ collection_name: this.config.collectionName });
    console.log(`[Milvus] Collection '${this.config.collectionName}' released`);
  }

  /**
   * 插入文档
   */
  async insertDocuments(documents: MilvusDocument[]): Promise<string[]> {
    if (!this.isInitialized) {
      await this.initializeCollection();
    }

    const client = await this.ensureConnected();
    const collectionName = this.config.collectionName;

    console.log(`[Milvus] Preparing to insert ${documents.length} documents...`);

    // 验证所有文档的 embedding 维度一致
    const firstDimension = documents[0]?.embedding?.length;
    if (!firstDimension) {
      throw new Error('First document has no embedding or invalid embedding');
    }

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      if (!doc.embedding || !Array.isArray(doc.embedding)) {
        console.error(`[Milvus] Document ${i} (id: ${doc.id}) has invalid embedding:`, doc.embedding);
        throw new Error(`Document ${i} (id: ${doc.id}) has invalid embedding`);
      }
      if (doc.embedding.length !== firstDimension) {
        console.error(`[Milvus] Dimension mismatch at document ${i}:`, {
          docId: doc.id,
          expected: firstDimension,
          actual: doc.embedding.length
        });
        throw new Error(`Document ${i} (id: ${doc.id}) has mismatched embedding dimension: expected ${firstDimension}D, got ${doc.embedding.length}D`);
      }
    }

    console.log(`[Milvus] ✅ All ${documents.length} documents have consistent embedding dimension: ${firstDimension}D`);

    // 使用简单的对象数组格式 (推荐格式)
    const data = documents.map((doc) => ({
      id: doc.id,
      content: doc.content.substring(0, 65000), // 限制长度
      embedding: doc.embedding,
      source: doc.metadata?.source || 'unknown',
      metadata_json: JSON.stringify(doc.metadata || {}).substring(0, 65000),
      created_at: Date.now(),
    }));

    console.log(`[Milvus] Inserting ${documents.length} documents...`);
    console.log(`[Milvus] Sample document structure:`, {
      id: data[0].id,
      contentLength: data[0].content.length,
      embeddingDimension: data[0].embedding.length,
      source: data[0].source
    });

    const insertReq: InsertReq = {
      collection_name: collectionName,
      data: data,
    };

    const result = await client.insert(insertReq);
    
    if (result.status.error_code !== 'Success') {
      throw new Error(`Insert failed: ${result.status.reason}`);
    }

    console.log(`[Milvus] Inserted ${result.insert_cnt} documents`);
    
    // 刷新数据确保持久化
    console.log(`[Milvus] Flushing data...`);
    await client.flushSync({ collection_names: [collectionName] });
    
    // 重新加载集合以确保新数据可被搜索
    console.log(`[Milvus] Reloading collection to make new data searchable...`);
    try {
      await client.releaseCollection({ collection_name: collectionName });
      await client.loadCollection({ collection_name: collectionName });
      console.log(`[Milvus] Collection reloaded successfully`);
    } catch (reloadError) {
      console.warn(`[Milvus] Reload warning (may be OK):`, reloadError);
    }

    // 返回所有文档的 ID
    return data.map(d => d.id);
  }

  /**
   * 相似度搜索
   */
  async search(
    queryEmbedding: number[],
    topK: number = 5,
    threshold: number = 0.0,
    filter?: string
  ): Promise<MilvusSearchResult[]> {
    if (!this.isInitialized) {
      await this.initializeCollection();
    }

    const client = await this.ensureConnected();
    const collectionName = this.config.collectionName;

    // 根据索引类型设置搜索参数
    let searchParams: any = {};
    switch (this.config.indexType) {
      case 'IVF_FLAT':
      case 'IVF_SQ8':
      case 'IVF_PQ':
        searchParams = { nprobe: 16 };
        break;
      case 'HNSW':
        searchParams = { ef: 64 };
        break;
      case 'ANNOY':
        searchParams = { search_k: -1 };
        break;
      default:
        searchParams = {};
    }

    const searchReq = {
      collection_name: collectionName,
      data: [queryEmbedding],
      anns_field: 'embedding',
      limit: topK,
      output_fields: ['id', 'content', 'source', 'metadata_json', 'created_at'],
      params: searchParams,
      filter: filter,
    } as any; // 使用 any 绕过类型检查，因为 SDK 类型定义可能不完整

    const results = await client.search(searchReq);

    console.log('[Milvus] Search response status:', results.status);
    console.log('[Milvus] Search results type:', typeof results.results, Array.isArray(results.results));
    console.log('[Milvus] Search results length:', results.results?.length);

    if (results.status.error_code !== 'Success') {
      throw new Error(`Search failed: ${results.status.reason}`);
    }

    // 转换结果
    const searchResults: MilvusSearchResult[] = [];
    
    // Milvus SDK 2.x 返回的 results.results 直接是数组
    // 但如果是多向量查询，可能是嵌套数组
    let hits: any[] = [];
    
    if (Array.isArray(results.results)) {
      if (results.results.length > 0) {
        // 检查是否是嵌套数组（多向量查询）
        if (Array.isArray(results.results[0])) {
          hits = results.results[0];
        } else {
          // 单向量查询，直接使用
          hits = results.results;
        }
      }
    }
    
    console.log('[Milvus] Parsed hits count:', hits.length);
    if (hits.length > 0) {
      console.log('[Milvus] First hit sample:', JSON.stringify(hits[0]).substring(0, 200));
    }
    
    if (hits.length === 0) {
      console.warn('[Milvus] No search results returned');
      return [];
    }
    
    for (const hit of hits) {
      // 计算相似度 (根据度量类型转换)
      let similarity: number;
      const distance = (hit as any).score || (hit as any).distance || 0;
      
      switch (this.config.metricType) {
        case 'COSINE':
          // Milvus COSINE 返回的是 1 - cosine_similarity
          similarity = 1 - distance;
          break;
        case 'IP':
          // Inner Product，越大越相似
          similarity = distance;
          break;
        case 'L2':
          // L2 距离，越小越相似，转换为相似度
          similarity = 1 / (1 + distance);
          break;
        default:
          similarity = distance;
      }

      // 应用阈值过滤
      if (similarity < threshold) {
        continue;
      }

      const hitData = hit as any;
      let metadata = {};
      try {
        metadata = JSON.parse(hitData.metadata_json || '{}');
      } catch {
        metadata = { source: hitData.source };
      }

      searchResults.push({
        id: hitData.id || '',
        content: hitData.content || '',
        metadata: { ...metadata, source: hitData.source },
        score: similarity,
        distance: distance,
      });
    }

    return searchResults;
  }

  /**
   * 删除文档
   */
  async deleteDocuments(ids: string[]): Promise<void> {
    const client = await this.ensureConnected();
    const collectionName = this.config.collectionName;

    const expr = `id in [${ids.map(id => `"${id}"`).join(',')}]`;
    
    await client.delete({
      collection_name: collectionName,
      filter: expr,
    });

    console.log(`[Milvus] Deleted ${ids.length} documents`);
  }

  /**
   * 清空集合
   */
  async clearCollection(): Promise<void> {
    const client = await this.ensureConnected();
    const collectionName = this.config.collectionName;

    // 删除并重建集合
    const hasCollection = await client.hasCollection({ collection_name: collectionName });
    
    if (hasCollection.value) {
      await client.dropCollection({ collection_name: collectionName });
      console.log(`[Milvus] Collection '${collectionName}' dropped`);
    }

    this.isInitialized = false;
    await this.initializeCollection();
  }

  /**
   * 获取集合统计信息
   */
  async getCollectionStats(): Promise<CollectionStats | null> {
    try {
      const client = await this.ensureConnected();
      const collectionName = this.config.collectionName;

      const hasCollection = await client.hasCollection({ collection_name: collectionName });
      if (!hasCollection.value) {
        return null;
      }

      const stats = await client.getCollectionStatistics({ collection_name: collectionName });
      const loadState = await client.getLoadState({ collection_name: collectionName });
      
      // 从集合 schema 获取实际的向量维度
      let actualDimension = this.config.embeddingDimension;
      try {
        const collectionInfo = await client.describeCollection({ collection_name: collectionName });
        const embeddingField = collectionInfo.schema?.fields?.find(
          (f: any) => f.name === 'embedding' && f.type_params
        );
        if (embeddingField?.type_params) {
          const dimParam = embeddingField.type_params.find((p: any) => p.key === 'dim');
          if (dimParam?.value) {
            actualDimension = parseInt(dimParam.value);
            console.log(`[Milvus] Actual collection dimension from schema: ${actualDimension}D`);
          }
        }
      } catch (schemaError) {
        console.warn('[Milvus] Could not get schema dimension, using config:', schemaError);
      }

      return {
        name: collectionName,
        rowCount: parseInt(stats.data.row_count || '0'),
        embeddingDimension: actualDimension,
        indexType: this.config.indexType,
        metricType: this.config.metricType,
        loaded: loadState.state === 'LoadStateLoaded',
      };
    } catch (error) {
      console.error('[Milvus] Error getting stats:', error);
      return null;
    }
  }

  /**
   * 检查健康状态
   */
  async checkHealth(): Promise<{ healthy: boolean; message: string }> {
    try {
      const client = await this.ensureConnected();
      const health = await client.checkHealth();
      
      return {
        healthy: health.isHealthy,
        message: health.isHealthy ? 'Milvus is healthy' : 'Milvus is not healthy',
      };
    } catch (error) {
      return {
        healthy: false,
        message: `Connection failed: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * 获取文档数量
   */
  async getDocumentCount(): Promise<number> {
    const stats = await this.getCollectionStats();
    return stats?.rowCount || 0;
  }

  /**
   * 获取配置
   */
  getConfig(): Required<MilvusConfig> {
    return { ...this.config };
  }

  /**
   * 更新配置（需要重新连接）
   */
  async updateConfig(newConfig: Partial<MilvusConfig>): Promise<void> {
    await this.disconnect();
    this.config = { ...this.config, ...newConfig };
    await this.connect();
  }

  /**
   * 获取连接状态
   */
  isReady(): boolean {
    return this.isConnected && this.isInitialized;
  }
}

// 全局单例
let milvusInstance: MilvusVectorStore | null = null;

/**
 * 获取 Milvus 实例（单例模式，支持配置更新）
 */
export function getMilvusInstance(config?: MilvusConfig): MilvusVectorStore {
  if (!milvusInstance) {
    milvusInstance = new MilvusVectorStore(config);
    console.log('[Milvus] Created new instance with config:', config?.collectionName, config?.embeddingDimension);
  } else if (config) {
    // 检查关键配置是否变化，如果变化则重建实例
    const currentConfig = milvusInstance.getConfig();
    if (currentConfig.embeddingDimension !== config.embeddingDimension ||
        currentConfig.collectionName !== config.collectionName) {
      console.log('[Milvus] Config changed, recreating instance...');
      console.log('[Milvus] Old:', currentConfig.embeddingDimension, 'New:', config.embeddingDimension);
      // 断开旧连接
      milvusInstance.disconnect().catch(() => {});
      milvusInstance = new MilvusVectorStore(config);
    }
  }
  return milvusInstance;
}

/**
 * 重置 Milvus 实例
 */
export async function resetMilvusInstance(): Promise<void> {
  if (milvusInstance) {
    await milvusInstance.disconnect();
    milvusInstance = null;
  }
}

// Embedding 模型维度映射
const MODEL_DIMENSIONS: Record<string, number> = {
  'nomic-embed-text': 768,
  'nomic-embed-text-v2-moe': 768,
  'mxbai-embed-large': 1024,
  'bge-large': 1024,
  'bge-m3': 1024,
  'snowflake-arctic-embed': 1024,
  'e5-large': 1024,
  'gte-large': 1024,
  'all-minilm': 384,
  'paraphrase-multilingual': 768,
};

/**
 * 获取模型的向量维度
 */
export function getModelDimension(modelName: string): number {
  // 移除 :latest 后缀
  const baseName = modelName.split(':')[0].toLowerCase();
  
  console.log(`[getModelDimension] Input: "${modelName}", BaseName: "${baseName}"`);
  
  // 精确匹配
  if (MODEL_DIMENSIONS[baseName]) {
    console.log(`[getModelDimension] Exact match: ${baseName} → ${MODEL_DIMENSIONS[baseName]}D`);
    return MODEL_DIMENSIONS[baseName];
  }
  
  // 部分匹配
  for (const [key, dim] of Object.entries(MODEL_DIMENSIONS)) {
    if (baseName.includes(key) || key.includes(baseName)) {
      console.log(`[getModelDimension] Partial match: ${key} → ${dim}D`);
      return dim;
    }
  }
  
  // 默认维度
  console.log(`[getModelDimension] No match, using default: 768D`);
  return 768;
}

/**
 * 根据维度选择合适的 embedding 模型
 */
export function selectModelByDimension(dimension: number): string {
  console.log(`[selectModelByDimension] Looking for model with dimension: ${dimension}D`);
  
  // 按维度分组的模型列表（优先使用的模型在前）
  const modelsByDimension: Record<number, string[]> = {
    384: ['all-minilm'],
    768: ['nomic-embed-text', 'nomic-embed-text-v2-moe', 'paraphrase-multilingual'],
    1024: ['bge-m3', 'bge-large', 'mxbai-embed-large', 'snowflake-arctic-embed', 'e5-large', 'gte-large'],
  };
  
  const candidates = modelsByDimension[dimension];
  
  if (candidates && candidates.length > 0) {
    const selected = candidates[0];
    console.log(`[selectModelByDimension] Selected: ${selected} (${dimension}D)`);
    return selected;
  }
  
  // 如果没有精确匹配，选择最接近的
  const availableDimensions = Object.keys(modelsByDimension).map(Number);
  const closest = availableDimensions.reduce((prev, curr) =>
    Math.abs(curr - dimension) < Math.abs(prev - dimension) ? curr : prev
  );
  
  const fallback = modelsByDimension[closest][0];
  console.log(`[selectModelByDimension] No exact match, using closest: ${fallback} (${closest}D)`);
  return fallback;
}

export default MilvusVectorStore;
