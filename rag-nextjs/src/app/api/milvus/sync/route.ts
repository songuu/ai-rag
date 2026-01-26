import { NextRequest, NextResponse } from 'next/server';
import { getMilvusInstance, MilvusConfig } from '@/lib/milvus-client';
import { getRagSystem } from '@/lib/rag-instance';
import { createEmbedding, getModelFactory } from '@/lib/model-config';
import fs from 'fs';
import path from 'path';

// 环境变量配置
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS || 'localhost:19530';
const MILVUS_COLLECTION = process.env.MILVUS_COLLECTION || 'rag_documents';
const factory = getModelFactory();
const envConfig = factory.getEnvConfig();
const EMBEDDING_MODEL = envConfig.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text';
const UPLOADS_DIR = path.join(process.cwd(), 'uploads');

const defaultConfig: MilvusConfig = {
  address: MILVUS_ADDRESS,
  collectionName: MILVUS_COLLECTION,
  embeddingDimension: 768,
  indexType: 'IVF_FLAT',
  metricType: 'COSINE',
};

// 模型维度映射
const MODEL_DIMENSIONS: Record<string, number> = {
  'nomic-embed-text': 768,
  'mxbai-embed-large': 1024,
  'bge-large': 1024,
  'bge-m3': 1024,
};

function getModelDimension(model: string): number {
  const baseName = model.split(':')[0].toLowerCase();
  return MODEL_DIMENSIONS[baseName] || 768;
}

// POST: 同步文档到 Milvus
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action = 'sync-from-uploads', embeddingModel = EMBEDDING_MODEL } = body;

    console.log(`[Milvus Sync] Action: ${action}, Model: ${embeddingModel}`);

    switch (action) {
      // 从 uploads 目录同步文档到 Milvus
      case 'sync-from-uploads': {
        // 检查 uploads 目录
        if (!fs.existsSync(UPLOADS_DIR)) {
          return NextResponse.json({
            success: false,
            error: 'uploads 目录不存在',
          }, { status: 400 });
        }

        const files = fs.readdirSync(UPLOADS_DIR).filter(f => 
          f.endsWith('.txt') || f.endsWith('.md') || f.endsWith('.json')
        );

        if (files.length === 0) {
          return NextResponse.json({
            success: false,
            error: '没有找到可同步的文档文件',
          }, { status: 400 });
        }

        // 获取模型维度并更新配置
        const dimension = getModelDimension(embeddingModel);
        const config = { ...defaultConfig, embeddingDimension: dimension };

        // 连接 Milvus
        const milvus = getMilvusInstance(config);
        await milvus.connect();

        // 检查是否需要重建集合（维度变化）
        const stats = await milvus.getCollectionStats();
        if (stats && stats.embeddingDimension !== dimension) {
          console.log(`[Milvus Sync] 维度变化 (${stats.embeddingDimension} -> ${dimension})，重建集合...`);
          await milvus.clearCollection();
        }

        await milvus.initializeCollection();

        // 使用统一配置系统创建 Embedding 模型
        const embeddings = createEmbedding(embeddingModel);

        // 读取并处理每个文件
        const results: Array<{ filename: string; chunks: number; success: boolean; error?: string }> = [];
        let totalChunks = 0;

        for (const filename of files) {
          try {
            const filePath = path.join(UPLOADS_DIR, filename);
            const content = fs.readFileSync(filePath, 'utf-8');

            // 分块处理
            const chunks = splitTextIntoChunks(content, 500, 50);
            
            // 生成嵌入向量
            const chunkTexts = chunks.map(c => c.text);
            const vectors = await embeddings.embedDocuments(chunkTexts);

            // 插入 Milvus
            const documents = chunks.map((chunk, i) => ({
              id: `${filename.replace(/[^a-zA-Z0-9]/g, '_')}_${i}_${Date.now()}`,
              content: chunk.text,
              embedding: vectors[i],
              metadata: {
                source: filename,
                chunkIndex: i,
                totalChunks: chunks.length,
              },
            }));

            await milvus.insertDocuments(documents);
            
            results.push({ filename, chunks: chunks.length, success: true });
            totalChunks += chunks.length;
            console.log(`[Milvus Sync] ✅ ${filename}: ${chunks.length} chunks`);
          } catch (error) {
            console.error(`[Milvus Sync] ❌ ${filename}:`, error);
            results.push({ 
              filename, 
              chunks: 0, 
              success: false, 
              error: error instanceof Error ? error.message : 'Unknown error' 
            });
          }
        }

        // 获取最新统计
        const newStats = await milvus.getCollectionStats();

        return NextResponse.json({
          success: true,
          message: `成功同步 ${results.filter(r => r.success).length}/${files.length} 个文件`,
          totalChunks,
          results,
          stats: newStats,
          embeddingModel,
          dimension,
        });
      }

      // 从内存 RAG 系统同步 (实际上也是从 uploads 目录读取，因为内存数据来自 uploads)
      case 'sync-from-memory': {
        // 由于无法直接访问内存中的文档对象，我们从 uploads 目录重新读取
        // 这与 sync-from-uploads 功能相同
        console.log('[Milvus Sync] sync-from-memory redirecting to sync-from-uploads');
        
        // 检查 uploads 目录
        if (!fs.existsSync(UPLOADS_DIR)) {
          return NextResponse.json({
            success: false,
            error: 'uploads 目录不存在',
          }, { status: 400 });
        }

        const files = fs.readdirSync(UPLOADS_DIR).filter(f => 
          f.endsWith('.txt') || f.endsWith('.md') || f.endsWith('.json')
        );

        if (files.length === 0) {
          return NextResponse.json({
            success: false,
            error: '没有找到可同步的文档文件',
          }, { status: 400 });
        }

        // 获取模型维度并更新配置
        const dimension = getModelDimension(embeddingModel);
        const config = { ...defaultConfig, embeddingDimension: dimension };

        // 连接 Milvus
        const milvus = getMilvusInstance(config);
        await milvus.connect();

        // 检查是否需要重建集合（维度变化）
        const stats = await milvus.getCollectionStats();
        if (stats && stats.embeddingDimension !== dimension) {
          console.log(`[Milvus Sync] 维度变化 (${stats.embeddingDimension} -> ${dimension})，重建集合...`);
          await milvus.clearCollection();
        }

        await milvus.initializeCollection();

        // 使用统一配置系统创建 Embedding 模型
        const embeddings = createEmbedding(embeddingModel);

        // 读取并处理每个文件
        const results: Array<{ filename: string; chunks: number; success: boolean; error?: string }> = [];
        let totalChunks = 0;

        for (const filename of files) {
          try {
            const filePath = path.join(UPLOADS_DIR, filename);
            const content = fs.readFileSync(filePath, 'utf-8');

            // 分块处理
            const chunks = splitTextIntoChunks(content, 500, 50);
            
            // 生成嵌入向量
            const chunkTexts = chunks.map(c => c.text);
            const vectors = await embeddings.embedDocuments(chunkTexts);

            // 插入 Milvus
            const documents = chunks.map((chunk, i) => ({
              id: `${filename.replace(/[^a-zA-Z0-9]/g, '_')}_${i}_${Date.now()}`,
              content: chunk.text,
              embedding: vectors[i],
              metadata: {
                source: filename,
                chunkIndex: i,
                totalChunks: chunks.length,
              },
            }));

            await milvus.insertDocuments(documents);
            
            results.push({ filename, chunks: chunks.length, success: true });
            totalChunks += chunks.length;
            console.log(`[Milvus Sync] ✅ ${filename}: ${chunks.length} chunks`);
          } catch (error) {
            console.error(`[Milvus Sync] ❌ ${filename}:`, error);
            results.push({ 
              filename, 
              chunks: 0, 
              success: false, 
              error: error instanceof Error ? error.message : 'Unknown error' 
            });
          }
        }

        // 获取最新统计
        const newStats = await milvus.getCollectionStats();

        return NextResponse.json({
          success: true,
          message: `成功同步 ${results.filter(r => r.success).length}/${files.length} 个文件`,
          totalChunks,
          results,
          stats: newStats,
          embeddingModel,
          dimension,
        });
      }

      default:
        return NextResponse.json({
          success: false,
          error: `未知操作: ${action}`,
        }, { status: 400 });
    }
  } catch (error) {
    console.error('[Milvus Sync] Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}

// 文本分块函数
function splitTextIntoChunks(text: string, chunkSize: number, overlap: number): Array<{ text: string; start: number; end: number }> {
  const chunks: Array<{ text: string; start: number; end: number }> = [];
  
  // 按段落分割
  const paragraphs = text.split(/\n\n+/);
  let currentChunk = '';
  let currentStart = 0;
  let position = 0;

  for (const para of paragraphs) {
    if (currentChunk.length + para.length > chunkSize && currentChunk.length > 0) {
      chunks.push({
        text: currentChunk.trim(),
        start: currentStart,
        end: position,
      });
      
      // 保留重叠
      const words = currentChunk.split(/\s+/);
      const overlapWords = words.slice(-Math.ceil(overlap / 5));
      currentChunk = overlapWords.join(' ') + '\n\n' + para;
      currentStart = position - overlapWords.join(' ').length;
    } else {
      currentChunk += (currentChunk ? '\n\n' : '') + para;
    }
    position += para.length + 2;
  }

  if (currentChunk.trim()) {
    chunks.push({
      text: currentChunk.trim(),
      start: currentStart,
      end: position,
    });
  }

  return chunks;
}

// GET: 获取同步状态
export async function GET() {
  try {
    // 检查 uploads 目录
    const uploadsExist = fs.existsSync(UPLOADS_DIR);
    const uploadFiles = uploadsExist 
      ? fs.readdirSync(UPLOADS_DIR).filter(f => f.endsWith('.txt') || f.endsWith('.md') || f.endsWith('.json'))
      : [];

    // 检查 Milvus 状态
    let milvusStats = null;
    try {
      const milvus = getMilvusInstance(defaultConfig);
      await milvus.connect();
      milvusStats = await milvus.getCollectionStats();
    } catch (e) {
      console.error('[Milvus Sync] Cannot get Milvus stats:', e);
    }

    // 检查内存 RAG
    let memoryDocCount = 0;
    try {
      const ragSystem = await getRagSystem();
      const status = ragSystem.getStatus();
      memoryDocCount = status?.documentCount || 0;
    } catch (e) {
      console.error('[Milvus Sync] Cannot get memory docs:', e);
    }

    return NextResponse.json({
      success: true,
      uploads: {
        exists: uploadsExist,
        files: uploadFiles,
        count: uploadFiles.length,
      },
      memory: {
        documentCount: memoryDocCount,
      },
      milvus: milvusStats ? {
        connected: true,
        rowCount: milvusStats.rowCount,
        dimension: milvusStats.embeddingDimension,
      } : {
        connected: false,
        rowCount: 0,
      },
      needsSync: memoryDocCount > 0 && (!milvusStats || milvusStats.rowCount === 0),
    });
  } catch (error) {
    console.error('[Milvus Sync] GET Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
