import { NextRequest, NextResponse } from 'next/server';
import { getRagSystem } from '@/lib/rag-instance';
import { analyzeQuery } from '@/lib/semantic-analyzer';
import { getMilvusInstance, MilvusConfig } from '@/lib/milvus-client';
import { OllamaEmbeddings } from '@langchain/ollama';
import { AgenticRAGSystem } from '@/lib/agentic-rag';

const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS || 'localhost:19530';
const MILVUS_COLLECTION = process.env.MILVUS_COLLECTION || 'rag_documents';
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';

// 默认 Milvus 配置
const defaultMilvusConfig: MilvusConfig = {
  address: MILVUS_ADDRESS,
  collectionName: MILVUS_COLLECTION,
  embeddingDimension: 768,
  indexType: 'IVF_FLAT',
  metricType: 'COSINE',
};

function getEmbeddingModel(modelName: string): OllamaEmbeddings {
  return new OllamaEmbeddings({
    model: modelName,
    baseUrl: OLLAMA_BASE_URL,
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      question, 
      topK = 3, 
      similarityThreshold = 0.0,
      llmModel = 'llama3.1',
      embeddingModel = 'nomic-embed-text',
      userId,
      sessionId,
      storageBackend = 'memory', // 存储后端选择
      useAgenticRAG = false,     // 是否使用 Agentic RAG 模式
      maxRetries = 2,            // Agentic RAG 最大重试次数
    } = body;

    if (!question || typeof question !== "string") {
      return NextResponse.json(
        { error: "请提供有效的问题" },
        { status: 400 }
      );
    }

    console.log(`[Ask API] 使用模型 - LLM: ${llmModel}, Embedding: ${embeddingModel}, 后端: ${storageBackend}, Agentic: ${useAgenticRAG}`);

    // 使用 Agentic RAG 模式
    if (useAgenticRAG && storageBackend === 'milvus') {
      return await handleAgenticQuery(question, {
        topK: parseInt(topK),
        similarityThreshold: parseFloat(similarityThreshold),
        llmModel,
        embeddingModel,
        maxRetries: parseInt(maxRetries),
      });
    }

    // 根据存储后端选择不同的检索方式
    if (storageBackend === 'milvus') {
      return await handleMilvusQuery(question, {
        topK: parseInt(topK),
        similarityThreshold: parseFloat(similarityThreshold),
        llmModel,
        embeddingModel,
        userId,
        sessionId
      });
    }

    // 默认使用内存存储
    const ragSystem = await getRagSystem();
    
    const result = await ragSystem.askWithDetails(question.trim(), {
      topK: parseInt(topK),
      similarityThreshold: parseFloat(similarityThreshold),
      llmModel,
      embeddingModel,
      userId,
      sessionId
    });

    // 使用语义分析器进行深度分析
    const queryEmbedding = result.retrievalDetails.queryEmbedding;
    const queryAnalysis = analyzeQuery(
      question,
      queryEmbedding,
      embeddingModel, // 使用实际选择的模型名称
      result.retrievalDetails.queryVectorizationTime || 0
    );

    return NextResponse.json({
      success: true,
      question,
      answer: result.answer,
      models: {
        llm: llmModel,
        embedding: embeddingModel
      },
      retrievalDetails: {
        searchResults: result.retrievalDetails.searchResults.map(r => ({
          document: {
            content: r.document.pageContent,
            metadata: r.document.metadata
          },
          similarity: r.similarity,
          index: r.index
        })),
        queryEmbedding: queryEmbedding.slice(0, 10),
        threshold: result.retrievalDetails.threshold,
        topK: result.retrievalDetails.topK,
        totalDocuments: result.retrievalDetails.totalDocuments,
        searchTime: result.retrievalDetails.searchTime
      },
      queryAnalysis,
      context: result.context,
      traceId: result.traceId,
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error("问答处理错误:", error);
    return NextResponse.json(
      { 
        error: "处理问题时发生错误",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

// Milvus 查询处理
async function handleMilvusQuery(
  question: string,
  options: {
    topK: number;
    similarityThreshold: number;
    llmModel: string;
    embeddingModel: string;
    userId?: string;
    sessionId?: string;
  }
) {
  const startTime = Date.now();
  const { topK, similarityThreshold, llmModel, embeddingModel } = options;

  try {
    // 1. 连接 Milvus (使用统一的全局实例)
    const milvus = getMilvusInstance(defaultMilvusConfig);
    await milvus.connect();
    await milvus.initializeCollection();

    // 2. 获取查询向量
    const embeddings = getEmbeddingModel(embeddingModel);
    const queryEmbedding = await embeddings.embedQuery(question);
    const vectorizationTime = Date.now() - startTime;

    // 3. 执行 Milvus 搜索
    const searchStart = Date.now();
    const searchResults = await milvus.search(queryEmbedding, topK, similarityThreshold);
    const searchTime = Date.now() - searchStart;

    // 4. 构建上下文
    const context = searchResults
      .map((r, i) => `[文档 ${i + 1}] (相似度: ${(r.score * 100).toFixed(1)}%)\n${r.content}`)
      .join('\n\n');

    // 5. 调用 LLM 生成回答
    const { Ollama } = await import('@langchain/ollama');
    const llm = new Ollama({
      model: llmModel,
      baseUrl: OLLAMA_BASE_URL,
    });

    const prompt = `基于以下上下文信息回答用户的问题。如果上下文中没有相关信息，请说明你无法从现有知识库中找到答案。

上下文信息:
${context}

用户问题: ${question}

请提供详细、准确的回答:`;

    const llmStart = Date.now();
    const answer = await llm.invoke(prompt);
    const llmTime = Date.now() - llmStart;

    // 6. 生成查询分析
    const queryAnalysis = analyzeQuery(
      question,
      queryEmbedding,
      embeddingModel,
      vectorizationTime
    );

    // 7. 获取集合统计
    const stats = await milvus.getCollectionStats();

    return NextResponse.json({
      success: true,
      question,
      answer,
      models: {
        llm: llmModel,
        embedding: embeddingModel
      },
      storageBackend: 'milvus',
      retrievalDetails: {
        searchResults: searchResults.map((r, i) => ({
          document: {
            content: r.content,
            metadata: r.metadata
          },
          similarity: r.score,
          distance: r.distance,
          index: i
        })),
        queryEmbedding: queryEmbedding.slice(0, 10),
        threshold: similarityThreshold,
        topK,
        totalDocuments: stats?.rowCount || 0,
        searchTime,
        vectorizationTime,
        llmTime,
        milvusStats: stats
      },
      queryAnalysis,
      context,
      traceId: `milvus-${Date.now()}`,
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('[Milvus Query Error]:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Milvus 查询失败',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

// Agentic RAG 查询处理
async function handleAgenticQuery(
  question: string,
  options: {
    topK: number;
    similarityThreshold: number;
    llmModel: string;
    embeddingModel: string;
    maxRetries: number;
  }
) {
  const { topK, similarityThreshold, llmModel, embeddingModel, maxRetries } = options;

  try {
    const agenticRAG = new AgenticRAGSystem({
      ollamaBaseUrl: OLLAMA_BASE_URL,
      llmModel,
      embeddingModel,
      milvusConfig: {
        address: MILVUS_ADDRESS,
        collectionName: MILVUS_COLLECTION,
      },
      enableHallucinationCheck: true,
      enableSelfReflection: true,
    });

    const result = await agenticRAG.query(question, {
      topK,
      similarityThreshold,
      maxRetries,
    });

    return NextResponse.json({
      success: !result.error,
      question,
      answer: result.answer,
      models: {
        llm: llmModel,
        embedding: embeddingModel,
      },
      storageBackend: 'milvus',
      agenticMode: true,
      
      // 工作流信息
      workflow: {
        steps: result.workflowSteps,
        totalDuration: result.totalDuration,
        retryCount: result.retryCount,
      },
      
      // 查询分析
      queryAnalysis: result.queryAnalysis,
      
      // 检索详情
      retrievalDetails: {
        searchResults: result.retrievedDocuments.map((doc, i) => ({
          document: {
            content: doc.content,
            metadata: doc.metadata,
          },
          similarity: doc.score,
          relevanceScore: doc.relevanceScore,
          factualScore: doc.factualScore,
          index: i,
        })),
        quality: result.retrievalQuality,
        selfReflection: result.selfReflection,
        totalDocuments: result.retrievedDocuments.length,
        // 添加标准字段以兼容前端显示
        threshold: similarityThreshold,
        topK: topK,
        searchTime: result.workflowSteps?.find((s: any) => s.step === '文档检索')?.duration || 0,
      },
      
      // 幻觉检查
      hallucinationCheck: result.hallucinationCheck,
      
      context: result.context,
      traceId: `agentic-${Date.now()}`,
      timestamp: new Date().toISOString(),
      error: result.error,
    });

  } catch (error) {
    console.error('[Agentic RAG Error]:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Agentic RAG 查询失败',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}