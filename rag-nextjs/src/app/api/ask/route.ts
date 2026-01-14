import { NextRequest, NextResponse } from 'next/server';
import { getRagSystem } from '@/lib/rag-instance';

// 计算向量特征
function calculateVectorFeatures(embedding: number[], text: string) {
  // 计算向量模长
  const vectorMagnitude = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
  
  // 基于文本内容估算各维度得分
  const hasTechKeywords = /人工智能|AI|机器学习|深度学习|神经网络|算法|模型|数据|编程|代码|技术|系统|软件|硬件|网络|云计算|大数据/i.test(text);
  const hasBusinessKeywords = /商业|市场|销售|客户|产品|服务|价格|成本|收入|利润|投资|管理|运营|战略/i.test(text);
  const hasDailyKeywords = /生活|日常|天气|吃饭|睡觉|运动|健康|旅游|购物|娱乐/i.test(text);
  const hasEmotionKeywords = /喜欢|讨厌|开心|难过|愤怒|恐惧|惊讶|感谢|抱歉/i.test(text);
  
  // 计算各维度分数（基于向量分布和关键词）
  const embeddingStats = {
    mean: embedding.reduce((a, b) => a + b, 0) / embedding.length,
    variance: 0,
    maxAbs: Math.max(...embedding.map(Math.abs))
  };
  embeddingStats.variance = embedding.reduce((sum, v) => sum + Math.pow(v - embeddingStats.mean, 2), 0) / embedding.length;
  
  // 基于向量统计特征和关键词计算各维度得分
  const techScore = Math.min(1, (hasTechKeywords ? 0.6 : 0.2) + embeddingStats.variance * 2);
  const businessScore = Math.min(1, (hasBusinessKeywords ? 0.5 : 0.15) + Math.abs(embeddingStats.mean) * 5);
  const dailyScore = Math.min(1, (hasDailyKeywords ? 0.5 : 0.2) + (1 - embeddingStats.variance) * 0.3);
  const emotionScore = Math.min(1, (hasEmotionKeywords ? 0.5 : 0.1) + embeddingStats.maxAbs * 0.3);
  
  return {
    techScore: parseFloat(techScore.toFixed(3)),
    businessScore: parseFloat(businessScore.toFixed(3)),
    dailyScore: parseFloat(dailyScore.toFixed(3)),
    emotionScore: parseFloat(emotionScore.toFixed(3)),
    vectorMagnitude: parseFloat(vectorMagnitude.toFixed(4))
  };
}

// 分析语义上下文
function analyzeSemanticContext(text: string, embedding: number[]) {
  const categories = [
    { name: 'AI技术', keywords: ['人工智能', 'AI', '机器学习', '深度学习', '神经网络', '算法', '模型'] },
    { name: '技术开发', keywords: ['编程', '代码', '软件', '系统', '开发', '框架', '工具'] },
    { name: '商业管理', keywords: ['商业', '市场', '销售', '管理', '运营', '战略'] },
    { name: '日常生活', keywords: ['生活', '日常', '健康', '运动', '娱乐'] },
    { name: '通用问答', keywords: ['什么', '如何', '为什么', '怎么'] }
  ];
  
  let bestCategory = '通用问答';
  let maxScore = 0;
  
  for (const cat of categories) {
    const score = cat.keywords.filter(kw => text.includes(kw)).length;
    if (score > maxScore) {
      maxScore = score;
      bestCategory = cat.name;
    }
  }
  
  // 基于向量计算置信度
  const vectorMagnitude = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
  const confidence = Math.min(0.95, 0.5 + maxScore * 0.1 + vectorMagnitude * 0.05);
  
  // 生成最近概念
  const nearestConcepts = categories
    .filter(cat => cat.name !== bestCategory)
    .flatMap(cat => cat.keywords.filter(kw => text.toLowerCase().includes(kw.toLowerCase())))
    .slice(0, 5);
  
  if (nearestConcepts.length === 0) {
    nearestConcepts.push('文本', '信息', '内容');
  }
  
  return {
    context: bestCategory === 'AI技术' ? '人工智能语境' : 
             bestCategory === '技术开发' ? '技术开发语境' : 
             bestCategory === '商业管理' ? '商业管理语境' : '通用语境',
    semanticCategory: bestCategory,
    confidence: parseFloat(confidence.toFixed(3)),
    nearestConcepts
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      question, 
      topK = 3, 
      similarityThreshold = 0.0,
      userId,
      sessionId 
    } = body;

    if (!question || typeof question !== "string") {
      return NextResponse.json(
        { error: "请提供有效的问题" },
        { status: 400 }
      );
    }

    const ragSystem = await getRagSystem();
    
    const result = await ragSystem.askWithDetails(question.trim(), {
      topK: parseInt(topK),
      similarityThreshold: parseFloat(similarityThreshold),
      userId,
      sessionId
    });

    // 计算真实的查询分析数据
    const queryEmbedding = result.retrievalDetails.queryEmbedding;
    const vectorFeatures = calculateVectorFeatures(queryEmbedding, question);
    const semanticAnalysis = analyzeSemanticContext(question, queryEmbedding);
    
    // 构建查询分析数据
    const queryAnalysis = {
      tokenization: {
        tokenCount: Math.ceil(question.length / 1.5), // 估算 token 数量
        processingTime: result.retrievalDetails.queryVectorizationTime || 0,
        originalText: question
      },
      embedding: {
        embedding: queryEmbedding.slice(0, 20), // 只返回前20维用于展示
        embeddingDimension: queryEmbedding.length,
        semanticAnalysis: {
          ...semanticAnalysis,
          vectorFeatures
        },
        modelInfo: {
          name: 'nomic-embed-text',
          vocabularySize: 50000 // 估算
        }
      }
    };

    return NextResponse.json({
      success: true,
      question,
      answer: result.answer,
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