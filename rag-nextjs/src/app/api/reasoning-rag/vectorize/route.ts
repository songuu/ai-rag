/**
 * Reasoning RAG 独立向量化 API
 * 使用专用的 Milvus 集合 reasoning_rag_documents
 * 
 * 已更新为使用统一模型配置系统 (model-config.ts)
 */

import { NextRequest, NextResponse } from 'next/server';
import { readdir, readFile } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { MilvusVectorStore } from '@/lib/milvus-client';
import { createEmbedding, getModelFactory } from '@/lib/model-config';
import { getEmbeddingConfigSummary, getEmbeddingDimension, ALL_EMBEDDING_DIMENSIONS } from '@/lib/embedding-config';

// Reasoning RAG 专用配置
const REASONING_UPLOAD_DIR = path.join(process.cwd(), 'reasoning-uploads');
const REASONING_COLLECTION = 'reasoning_rag_documents';

// 使用独立的 Embedding 配置系统
const embeddingConfig = getEmbeddingConfigSummary();
const DEFAULT_EMBEDDING_MODEL = embeddingConfig.model;

// 模型维度映射 - 使用统一映射
const MODEL_DIMENSIONS = ALL_EMBEDDING_DIMENSIONS;

function getModelDimension(model: string): number {
  // 优先使用 embedding-config 的维度
  if (!model || model === DEFAULT_EMBEDDING_MODEL) {
    return getEmbeddingDimension();
  }
  const baseName = model.split(':')[0];
  return MODEL_DIMENSIONS[baseName] || MODEL_DIMENSIONS[model] || 768;
}

// 文本分块函数
function splitTextIntoChunks(
  text: string, 
  chunkSize: number = 500, 
  overlap: number = 50
): { text: string; startIndex: number; endIndex: number }[] {
  const chunks: { text: string; startIndex: number; endIndex: number }[] = [];
  let startIndex = 0;

  while (startIndex < text.length) {
    let endIndex = Math.min(startIndex + chunkSize, text.length);
    
    // 尝试在句子边界处分割
    if (endIndex < text.length) {
      const lastPeriod = text.lastIndexOf('。', endIndex);
      const lastQuestion = text.lastIndexOf('？', endIndex);
      const lastExclaim = text.lastIndexOf('！', endIndex);
      const lastNewline = text.lastIndexOf('\n', endIndex);
      
      const candidates = [lastPeriod, lastQuestion, lastExclaim, lastNewline]
        .filter(idx => idx > startIndex && idx <= endIndex);
      
      if (candidates.length > 0) {
        endIndex = Math.max(...candidates) + 1;
      }
    }
    
    const chunkText = text.slice(startIndex, endIndex).trim();
    if (chunkText.length > 0) {
      chunks.push({
        text: chunkText,
        startIndex,
        endIndex
      });
    }
    
    startIndex = endIndex - overlap;
    if (startIndex >= text.length - overlap) break;
  }

  return chunks;
}

/**
 * POST: 向量化 Reasoning RAG 专用目录中的文件
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      action = 'vectorize-all',
      embeddingModel = DEFAULT_EMBEDDING_MODEL,
      chunkSize = 500,
      chunkOverlap = 50,
      files: specificFiles  // 可选：指定要向量化的文件
    } = body;

    console.log(`[Reasoning Vectorize] ========================================`);
    console.log(`[Reasoning Vectorize] Action: ${action}`);
    console.log(`[Reasoning Vectorize] Model: ${embeddingModel}`);
    console.log(`[Reasoning Vectorize] Collection: ${REASONING_COLLECTION}`);
    console.log(`[Reasoning Vectorize] ========================================`);

    // 检查上传目录
    if (!existsSync(REASONING_UPLOAD_DIR)) {
      return NextResponse.json({
        success: false,
        error: '没有找到上传的文件，请先上传文件'
      }, { status: 400 });
    }

    // 获取文件列表
    const allFiles = await readdir(REASONING_UPLOAD_DIR);
    const textFiles = allFiles.filter(f => f.endsWith('_parsed.txt'));

    if (textFiles.length === 0) {
      return NextResponse.json({
        success: false,
        error: '没有找到可向量化的文本文件'
      }, { status: 400 });
    }

    // 如果指定了特定文件，进行过滤
    const filesToProcess = specificFiles 
      ? textFiles.filter(f => specificFiles.includes(f))
      : textFiles;

    if (filesToProcess.length === 0) {
      return NextResponse.json({
        success: false,
        error: '指定的文件不存在或不可向量化'
      }, { status: 400 });
    }

    // 获取模型维度
    const dimension = getModelDimension(embeddingModel);
    console.log(`[Reasoning Vectorize] Model dimension: ${dimension}D`);

    // 创建 Milvus 客户端 - 使用独立集合
    const milvus = new MilvusVectorStore({
      collectionName: REASONING_COLLECTION,
      embeddingDimension: dimension,
      metricType: 'COSINE'
    });

    // 连接并初始化集合
    await milvus.connect();
    
    // 检查集合维度是否匹配
    const stats = await milvus.getCollectionStats();
    if (stats && stats.embeddingDimension !== dimension) {
      console.log(`[Reasoning Vectorize] 维度变化，重建集合 (${stats.embeddingDimension}D -> ${dimension}D)`);
      await milvus.clearCollection();
    }
    
    await milvus.initializeCollection();

    // 使用统一配置系统创建 Embedding 模型
    const embeddings = createEmbedding(embeddingModel);

    // 处理每个文件
    const results: Array<{
      filename: string;
      chunks: number;
      success: boolean;
      error?: string;
    }> = [];
    let totalChunks = 0;
    let totalDocuments = 0;

    for (const filename of filesToProcess) {
      try {
        const filePath = path.join(REASONING_UPLOAD_DIR, filename);
        const content = await readFile(filePath, 'utf-8');

        if (!content.trim()) {
          results.push({ filename, chunks: 0, success: false, error: '文件内容为空' });
          continue;
        }

        // 分块处理
        const chunks = splitTextIntoChunks(content, chunkSize, chunkOverlap);
        console.log(`[Reasoning Vectorize] ${filename}: ${chunks.length} chunks`);

        // 生成嵌入向量
        const chunkTexts = chunks.map(c => c.text);
        const vectors = await embeddings.embedDocuments(chunkTexts);

        // 构建文档
        const documents = chunks.map((chunk, i) => ({
          id: `reasoning_${filename.replace(/[^a-zA-Z0-9]/g, '_')}_${i}_${Date.now()}`,
          content: chunk.text,
          embedding: vectors[i],
          metadata: {
            source: filename,
            chunkIndex: i,
            totalChunks: chunks.length,
            startIndex: chunk.startIndex,
            endIndex: chunk.endIndex,
            collection: REASONING_COLLECTION,
            timestamp: Date.now()
          }
        }));

        // 插入 Milvus
        await milvus.insertDocuments(documents);

        results.push({ filename, chunks: chunks.length, success: true });
        totalChunks += chunks.length;
        totalDocuments += 1;
        
        console.log(`[Reasoning Vectorize] ✅ ${filename}: ${chunks.length} chunks indexed`);

      } catch (error) {
        console.error(`[Reasoning Vectorize] ❌ ${filename}:`, error);
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
      message: `成功向量化 ${results.filter(r => r.success).length}/${filesToProcess.length} 个文件`,
      collection: REASONING_COLLECTION,
      embeddingModel,
      dimension,
      totalChunks,
      totalDocuments,
      results,
      stats: newStats
    });

  } catch (error) {
    console.error('[Reasoning Vectorize] Error:', error);
    return NextResponse.json({
      success: false,
      error: '向量化处理失败',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
}

/**
 * GET: 获取 Reasoning RAG 集合状态
 */
export async function GET() {
  try {
    const milvus = new MilvusVectorStore({
      collectionName: REASONING_COLLECTION
    });

    await milvus.connect();
    const stats = await milvus.getCollectionStats();

    // 获取上传目录文件统计
    let fileCount = 0;
    let textFileCount = 0;
    
    if (existsSync(REASONING_UPLOAD_DIR)) {
      const allFiles = await readdir(REASONING_UPLOAD_DIR);
      textFileCount = allFiles.filter(f => f.endsWith('_parsed.txt')).length;
      fileCount = allFiles.filter(f => !f.endsWith('_parsed.txt')).length;
    }

    return NextResponse.json({
      success: true,
      collection: REASONING_COLLECTION,
      collectionStats: stats || { rowCount: 0, name: REASONING_COLLECTION },
      fileStats: {
        uploadedFiles: fileCount,
        textFiles: textFileCount,
        uploadDir: 'reasoning-uploads'
      },
      isReady: stats && stats.rowCount > 0
    });

  } catch (error) {
    console.error('[Reasoning Vectorize] Get stats error:', error);
    return NextResponse.json({
      success: false,
      error: '获取状态失败',
      details: error instanceof Error ? error.message : String(error),
      collection: REASONING_COLLECTION,
      collectionStats: null,
      isReady: false
    }, { status: 500 });
  }
}

/**
 * DELETE: 清空 Reasoning RAG 集合
 */
export async function DELETE() {
  try {
    const milvus = new MilvusVectorStore({
      collectionName: REASONING_COLLECTION
    });

    await milvus.connect();
    await milvus.clearCollection();

    return NextResponse.json({
      success: true,
      message: `成功清空集合: ${REASONING_COLLECTION}`
    });

  } catch (error) {
    console.error('[Reasoning Vectorize] Clear collection error:', error);
    return NextResponse.json({
      success: false,
      error: '清空集合失败',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
}
