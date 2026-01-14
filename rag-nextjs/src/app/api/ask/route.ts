import { NextRequest, NextResponse } from 'next/server';
import { getRagSystem } from '@/lib/rag-instance';
import { analyzeQuery } from '@/lib/semantic-analyzer';

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
      sessionId 
    } = body;

    if (!question || typeof question !== "string") {
      return NextResponse.json(
        { error: "请提供有效的问题" },
        { status: 400 }
      );
    }

    console.log(`[Ask API] 使用模型 - LLM: ${llmModel}, Embedding: ${embeddingModel}`);

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