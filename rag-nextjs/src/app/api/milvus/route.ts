import { NextRequest, NextResponse } from 'next/server';
import { MilvusVectorStore, MilvusConfig, getMilvusInstance, resetMilvusInstance } from '@/lib/milvus-client';
import { OllamaEmbeddings } from '@langchain/ollama';
import { v4 as uuidv4 } from 'uuid';

// 环境变量配置
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS || 'localhost:19530';
const MILVUS_USERNAME = process.env.MILVUS_USERNAME || '';
const MILVUS_PASSWORD = process.env.MILVUS_PASSWORD || '';
const MILVUS_DATABASE = process.env.MILVUS_DATABASE || 'default';
const MILVUS_COLLECTION = process.env.MILVUS_COLLECTION || 'rag_documents';
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'nomic-embed-text';
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';

// 默认配置
const defaultConfig: MilvusConfig = {
  address: MILVUS_ADDRESS,
  username: MILVUS_USERNAME,
  password: MILVUS_PASSWORD,
  database: MILVUS_DATABASE,
  collectionName: MILVUS_COLLECTION,
  embeddingDimension: 768,
  indexType: 'IVF_FLAT',
  metricType: 'COSINE',
};

// 模型维度映射
const MODEL_DIMENSION_MAP: Record<string, number> = {
  'nomic-embed-text': 768,
  'mxbai-embed-large': 1024,
  'bge-large': 1024,
  'bge-m3': 1024,
  'snowflake-arctic-embed': 1024,
  'e5-large': 1024,
  'gte-large': 1024,
};

// 根据模型名推断维度
function getModelDimension(modelName: string): number {
  const baseName = modelName.split(':')[0].toLowerCase().trim();
  
  console.log(`[getModelDimension] Input: "${modelName}", BaseName: "${baseName}"`);
  
  // 精确匹配
  if (MODEL_DIMENSION_MAP[baseName]) {
    console.log(`[getModelDimension] Exact match: ${baseName} → ${MODEL_DIMENSION_MAP[baseName]}D`);
    return MODEL_DIMENSION_MAP[baseName];
  }
  
  // 模糊匹配
  for (const [name, dim] of Object.entries(MODEL_DIMENSION_MAP)) {
    if (baseName.includes(name) || name.includes(baseName)) {
      console.log(`[getModelDimension] Fuzzy match: "${baseName}" ↔ "${name}" → ${dim}D`);
      return dim;
    }
  }
  
  // 根据名称模式推断（按优先级）
  if (baseName.includes('bge-m3')) {
    console.log(`[getModelDimension] Pattern: bge-m3 → 1024D`);
    return 1024;
  }
  if (baseName.includes('bge') && (baseName.includes('large') || baseName.includes('base'))) {
    console.log(`[getModelDimension] Pattern: bge-large/base → 1024D`);
    return 1024;
  }
  if (baseName.includes('nomic') || baseName.includes('embed-text')) {
    console.log(`[getModelDimension] Pattern: nomic → 768D`);
    return 768;
  }
  if (baseName.includes('mxbai') || baseName.includes('snowflake')) {
    console.log(`[getModelDimension] Pattern: mxbai/snowflake → 1024D`);
    return 1024;
  }
  
  console.warn(`[getModelDimension] No match for "${baseName}", defaulting to 768D`);
  return 768; // 默认
}

// 根据维度选择合适的模型
function selectModelByDimension(dimension: number, preferredModel?: string): string {
  if (preferredModel) {
    const modelDim = getModelDimension(preferredModel);
    if (modelDim === dimension) {
      return preferredModel; // 保留完整模型名称（包括版本标签）
    }
  }
  
  // 根据维度返回推荐模型
  return dimension === 768 ? 'nomic-embed-text' : 'mxbai-embed-large';
}

// 获取 Embedding 模型
function getEmbeddingModel(modelName?: string) {
  const actualModelName = modelName || EMBEDDING_MODEL;
  console.log(`[Milvus API] Creating embedding model: ${actualModelName}`);
  
  return new OllamaEmbeddings({
    baseUrl: OLLAMA_BASE_URL,
    model: actualModelName,
  });
}

// POST: 执行 Milvus 操作
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, ...params } = body;

    switch (action) {
      // 连接到 Milvus
      case 'connect': {
        const config: MilvusConfig = {
          ...defaultConfig,
          ...params.config,
        };
        
        const milvus = getMilvusInstance(config);
        await milvus.connect();
        await milvus.initializeCollection();
        
        const stats = await milvus.getCollectionStats();
        
        return NextResponse.json({
          success: true,
          message: 'Connected to Milvus',
          stats,
        });
      }

      // 断开连接
      case 'disconnect': {
        await resetMilvusInstance();
        return NextResponse.json({
          success: true,
          message: 'Disconnected from Milvus',
        });
      }

      // 检查健康状态
      case 'health': {
        const milvus = getMilvusInstance(defaultConfig);
        const health = await milvus.checkHealth();
        return NextResponse.json({
          success: true,
          ...health,
        });
      }

      // 获取集合统计信息
      case 'stats': {
        const milvus = getMilvusInstance(defaultConfig);
        const stats = await milvus.getCollectionStats();
        return NextResponse.json({
          success: true,
          stats,
        });
      }

      // 插入文档
      case 'insert': {
        const { documents, embeddingModel } = params;
        
        console.log(`[Milvus Insert] ========== 开始导入 ==========`);
        console.log(`[Milvus Insert] Documents count: ${documents?.length || 0}`);
        console.log(`[Milvus Insert] Requested embedding model: "${embeddingModel || 'default'}"`);
        
        if (!documents || !Array.isArray(documents) || documents.length === 0) {
          return NextResponse.json({
            success: false,
            error: '请提供有效的文档列表',
          }, { status: 400 });
        }

        const milvus = getMilvusInstance(defaultConfig);
        await milvus.connect();
        await milvus.initializeCollection();

        // 获取集合的向量维度
        const stats = await milvus.getCollectionStats();
        const collectionDimension = stats?.embeddingDimension || 768;
        console.log(`[Milvus Insert] Collection dimension: ${collectionDimension}D`);

        // 获取模型维度信息
        const actualModelName = embeddingModel || EMBEDDING_MODEL;
        const modelDimension = getModelDimension(actualModelName);
        
        console.log(`[Milvus Insert] Using model: "${actualModelName}" (${modelDimension}D)`);
        
        // 检查维度是否匹配
        if (modelDimension !== collectionDimension) {
          console.warn(`[Milvus Insert] ⚠️ 维度不匹配警告: 模型 ${modelDimension}D vs 集合 ${collectionDimension}D`);
          console.warn(`[Milvus Insert] 这可能会导致插入失败！`);
        }
        
        const embeddings = getEmbeddingModel(actualModelName);
        
        // 为每个文档生成向量
        console.log(`[Milvus Insert] Generating embeddings for ${documents.length} documents...`);
        const milvusDocs = await Promise.all(documents.map(async (doc: any) => {
          const embedding = await embeddings.embedQuery(doc.content);
          return {
            id: doc.id || uuidv4(),
            content: doc.content,
            embedding,
            metadata: doc.metadata || {},
          };
        }));

        // 验证生成的向量维度
        const actualDimension = milvusDocs[0]?.embedding?.length || 0;
        console.log(`[Milvus Insert] Generated embedding dimension: ${actualDimension}D`);
        
        if (actualDimension !== collectionDimension) {
          console.error(`[Milvus Insert] ❌ 维度不匹配! 生成: ${actualDimension}D, 集合: ${collectionDimension}D`);
          return NextResponse.json({
            success: false,
            error: `向量维度不匹配！生成的向量: ${actualDimension}维, 集合要求: ${collectionDimension}维。`,
            generatedDimension: actualDimension,
            collectionDimension,
            usedModel: actualModelName,
          }, { status: 400 });
        }

        console.log(`[Milvus Insert] ✅ 维度匹配，开始插入...`);
        const ids = await milvus.insertDocuments(milvusDocs);
        console.log(`[Milvus Insert] ✅ 成功插入 ${ids.length} 个文档`);
        console.log(`[Milvus Insert] ========== 导入完成 ==========`);
        
        return NextResponse.json({
          success: true,
          message: `Inserted ${ids.length} documents`,
          ids,
          embeddingModel: actualModelName,
          dimension: actualDimension,
          collectionDimension,
        });
      }

      // 相似度搜索
      case 'search': {
        const { query, topK = 5, threshold = 0.0, filter, embeddingModel } = params;
        
        console.log(`[Milvus Search] ========== 开始搜索 ==========`);
        console.log(`[Milvus Search] Query: "${query}"`);
        console.log(`[Milvus Search] Requested embedding model: "${embeddingModel || 'default'}"`);
        
        if (!query || typeof query !== 'string') {
          return NextResponse.json({
            success: false,
            error: '请提供有效的查询文本',
          }, { status: 400 });
        }

        const milvus = getMilvusInstance(defaultConfig);
        await milvus.connect();
        await milvus.initializeCollection();

        // 获取集合的向量维度
        const stats = await milvus.getCollectionStats();
        const collectionDimension = stats?.embeddingDimension || 768;
        console.log(`[Milvus Search] Collection dimension: ${collectionDimension}D`);

        // 自动选择与集合维度匹配的模型
        const actualModel = selectModelByDimension(collectionDimension, embeddingModel);
        console.log(`[Milvus Search] Auto-selected model: "${actualModel}"`);
        
        const embeddings = getEmbeddingModel(actualModel);
        
        const queryEmbedding = await embeddings.embedQuery(query);
        const queryDimension = queryEmbedding.length;
        console.log(`[Milvus Search] Generated query embedding dimension: ${queryDimension}D`);
        
        // 检查维度是否匹配
        if (queryDimension !== collectionDimension) {
          console.error(`[Milvus Search] ❌ 维度不匹配! Collection: ${collectionDimension}D, Query: ${queryDimension}D`);
          return NextResponse.json({
            success: false,
            error: `向量维度不匹配！集合维度: ${collectionDimension}, 查询向量维度: ${queryDimension}。请使用与导入文档时相同维度的 Embedding 模型，或清空集合后使用新模型重新导入。`,
            collectionDimension,
            queryDimension,
            requestedModel: embeddingModel,
            actualModel,
            suggestion: collectionDimension === 768 
              ? '建议使用 nomic-embed-text 模型 (768维)' 
              : collectionDimension === 1024 
                ? '建议使用 bge-m3 或 mxbai-embed-large 模型 (1024维)'
                : `需要 ${collectionDimension} 维的模型`,
          }, { status: 400 });
        }
        
        console.log(`[Milvus Search] ✅ 维度匹配，开始搜索...`);
        const results = await milvus.search(queryEmbedding, topK, threshold, filter);
        console.log(`[Milvus Search] ✅ 找到 ${results.length} 个结果`);
        console.log(`[Milvus Search] ========== 搜索完成 ==========`);
        
        return NextResponse.json({
          success: true,
          query,
          results,
          count: results.length,
          embeddingModel: actualModel,
          dimension: queryDimension,
          collectionDimension,
        });
      }

      // 删除文档
      case 'delete': {
        const { ids } = params;
        
        if (!ids || !Array.isArray(ids) || ids.length === 0) {
          return NextResponse.json({
            success: false,
            error: '请提供有效的文档 ID 列表',
          }, { status: 400 });
        }

        const milvus = getMilvusInstance(defaultConfig);
        await milvus.deleteDocuments(ids);
        
        return NextResponse.json({
          success: true,
          message: `Deleted ${ids.length} documents`,
        });
      }

      // 清空集合
      case 'clear': {
        const milvus = getMilvusInstance(defaultConfig);
        await milvus.clearCollection();
        
        return NextResponse.json({
          success: true,
          message: 'Collection cleared',
        });
      }

      // 从文件导入文档
      case 'import-files': {
        const { files } = params;
        
        if (!files || !Array.isArray(files) || files.length === 0) {
          return NextResponse.json({
            success: false,
            error: '请提供有效的文件列表',
          }, { status: 400 });
        }

        const milvus = getMilvusInstance(defaultConfig);
        await milvus.connect();
        await milvus.initializeCollection();

        const embeddings = getEmbeddingModel();
        const { RecursiveCharacterTextSplitter } = await import('@langchain/textsplitters');
        
        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 500,
          chunkOverlap: 50,
        });

        const allDocs: any[] = [];
        
        for (const file of files) {
          const { content, filename } = file;
          const chunks = await splitter.splitText(content);
          
          for (let i = 0; i < chunks.length; i++) {
            const embedding = await embeddings.embedQuery(chunks[i]);
            allDocs.push({
              id: `${filename}-chunk-${i}-${uuidv4().slice(0, 8)}`,
              content: chunks[i],
              embedding,
              metadata: {
                source: filename,
                chunkIndex: i,
                totalChunks: chunks.length,
              },
            });
          }
        }

        const ids = await milvus.insertDocuments(allDocs);
        
        return NextResponse.json({
          success: true,
          message: `Imported ${files.length} files as ${ids.length} chunks`,
          files: files.map((f: any) => f.filename),
          chunkCount: ids.length,
        });
      }

      // 重建索引
      case 'rebuild-index': {
        const { indexType = 'IVF_FLAT', metricType = 'COSINE' } = params;
        
        const milvus = getMilvusInstance(defaultConfig);
        await milvus.updateConfig({ indexType, metricType });
        await milvus.clearCollection();
        
        return NextResponse.json({
          success: true,
          message: `Index rebuilt with ${indexType} and ${metricType}`,
        });
      }

      // 更新配置
      case 'update-config': {
        const { config } = params;
        
        if (!config) {
          return NextResponse.json({
            success: false,
            error: '请提供配置参数',
          }, { status: 400 });
        }

        const milvus = getMilvusInstance(defaultConfig);
        await milvus.updateConfig(config);
        
        return NextResponse.json({
          success: true,
          message: 'Configuration updated',
          config: milvus.getConfig(),
        });
      }

      default:
        return NextResponse.json({
          success: false,
          error: `Unknown action: ${action}`,
        }, { status: 400 });
    }
  } catch (error) {
    console.error('[Milvus API] Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}

// GET: 获取 Milvus 状态和信息
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get('action') || 'status';

  try {
    switch (action) {
      case 'status': {
        const milvus = getMilvusInstance(defaultConfig);
        const health = await milvus.checkHealth();
        const stats = health.healthy ? await milvus.getCollectionStats() : null;
        
        return NextResponse.json({
          success: true,
          connected: health.healthy,
          health,
          stats,
          config: {
            address: defaultConfig.address,
            database: defaultConfig.database,
            collectionName: defaultConfig.collectionName,
            embeddingDimension: defaultConfig.embeddingDimension,
            indexType: defaultConfig.indexType,
            metricType: defaultConfig.metricType,
          },
        });
      }

      case 'config': {
        return NextResponse.json({
          success: true,
          config: {
            address: MILVUS_ADDRESS,
            database: MILVUS_DATABASE,
            collectionName: MILVUS_COLLECTION,
            embeddingModel: EMBEDDING_MODEL,
            supportedIndexTypes: ['FLAT', 'IVF_FLAT', 'IVF_SQ8', 'IVF_PQ', 'HNSW', 'ANNOY'],
            supportedMetricTypes: ['L2', 'IP', 'COSINE'],
          },
        });
      }

      default:
        return NextResponse.json({
          success: false,
          error: `Unknown action: ${action}`,
        }, { status: 400 });
    }
  } catch (error) {
    console.error('[Milvus API] Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
