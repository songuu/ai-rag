import { NextRequest, NextResponse } from 'next/server';

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';

// 已知的模型分类
const MODEL_CATEGORIES = {
  llm: {
    patterns: [
      'llama', 'mistral', 'mixtral', 'gemma', 'phi', 'qwen',
      'deepseek', 'yi', 'solar', 'vicuna', 'orca', 'starling',
      'openchat', 'neural', 'dolphin', 'wizard', 'falcon'
    ],
    exclude: ['embed', 'embedding']
  },
  embedding: {
    patterns: [
      'embed', 'embedding', 'bge', 'gte', 'jina', 'e5',
      'instructor', 'multilingual-e5'
    ],
    include: ['nomic-embed', 'mxbai-embed', 'snowflake-arctic-embed']
  }
};

// 推荐的模型配置
const RECOMMENDED_MODELS = {
  llm: [
    {
      name: 'llama3.1:latest',
      displayName: 'Llama 3.1',
      description: '最新的 Meta Llama 模型，性能优异',
      size: '4.7 GB',
      contextLength: 128000,
      recommended: true
    },
    {
      name: 'llama3.2:latest',
      displayName: 'Llama 3.2',
      description: 'Meta 最新版本，更快更准确',
      size: '2.0 GB',
      contextLength: 128000,
      recommended: true
    },
    {
      name: 'qwen2.5:latest',
      displayName: 'Qwen 2.5',
      description: '阿里通义千问，中文优化',
      size: '4.4 GB',
      contextLength: 32768,
      recommended: true
    },
    {
      name: 'mistral:latest',
      displayName: 'Mistral',
      description: '高性能开源模型',
      size: '4.1 GB',
      contextLength: 32768,
      recommended: false
    },
    {
      name: 'gemma2:latest',
      displayName: 'Gemma 2',
      description: 'Google 轻量级模型',
      size: '5.4 GB',
      contextLength: 8192,
      recommended: false
    }
  ],
  embedding: [
    {
      name: 'nomic-embed-text:latest',
      displayName: 'Nomic Embed Text',
      description: '高质量英文嵌入模型',
      dimension: 768,
      size: '274 MB',
      recommended: true
    },
    {
      name: 'mxbai-embed-large:latest',
      displayName: 'MixedBread AI Embed',
      description: '大型嵌入模型，性能优异',
      dimension: 1024,
      size: '669 MB',
      recommended: true
    },
    {
      name: 'bge-large:latest',
      displayName: 'BGE Large',
      description: 'BAAI 出品，中英文支持',
      dimension: 1024,
      size: '1.3 GB',
      recommended: false
    },
    {
      name: 'snowflake-arctic-embed:latest',
      displayName: 'Snowflake Arctic Embed',
      description: '多语言嵌入模型',
      dimension: 1024,
      size: '669 MB',
      recommended: false
    }
  ]
};

// 判断模型类型
function categorizeModel(modelName: string): 'llm' | 'embedding' | 'unknown' {
  const nameLower = modelName.toLowerCase();
  
  // 检查是否为 embedding 模型
  if (MODEL_CATEGORIES.embedding.include.some(pattern => nameLower.includes(pattern))) {
    return 'embedding';
  }
  
  if (MODEL_CATEGORIES.embedding.patterns.some(pattern => nameLower.includes(pattern))) {
    return 'embedding';
  }
  
  // 排除 embedding 后检查 LLM
  if (MODEL_CATEGORIES.llm.exclude.some(pattern => nameLower.includes(pattern))) {
    return 'unknown';
  }
  
  if (MODEL_CATEGORIES.llm.patterns.some(pattern => nameLower.includes(pattern))) {
    return 'llm';
  }
  
  return 'unknown';
}

// 获取模型详细信息
async function getModelDetails(modelName: string) {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/show`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName })
    });
    
    if (!response.ok) return null;
    
    const data = await response.json();
    return data;
  } catch (error) {
    return null;
  }
}

// GET: 获取本地模型列表
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const action = searchParams.get('action');
  
  try {
    // 检查 Ollama 状态
    const statusResponse = await fetch(`${OLLAMA_BASE_URL}/api/tags`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!statusResponse.ok) {
      return NextResponse.json({
        success: false,
        error: 'Ollama 服务未运行',
        code: 'OLLAMA_OFFLINE',
        suggestion: '请先启动 Ollama 服务: ollama serve'
      }, { status: 503 });
    }
    
    const data = await statusResponse.json();
    const allModels = data.models || [];
    
    // 如果没有任何模型
    if (allModels.length === 0) {
      return NextResponse.json({
        success: true,
        hasModels: false,
        llmModels: [],
        embeddingModels: [],
        allModels: [],
        recommended: RECOMMENDED_MODELS,
        message: '未检测到已安装的模型',
        suggestion: '请安装推荐的模型'
      });
    }
    
    // 分类模型
    const llmModels: any[] = [];
    const embeddingModels: any[] = [];
    const unknownModels: any[] = [];
    
    for (const model of allModels) {
      const modelName = model.name;
      const category = categorizeModel(modelName);
      
      const modelInfo = {
        name: modelName,
        displayName: modelName.split(':')[0],
        tag: modelName.split(':')[1] || 'latest',
        size: model.size,
        sizeFormatted: formatBytes(model.size),
        modifiedAt: model.modified_at,
        digest: model.digest,
        category
      };
      
      if (category === 'llm') {
        llmModels.push(modelInfo);
      } else if (category === 'embedding') {
        embeddingModels.push(modelInfo);
      } else {
        unknownModels.push(modelInfo);
      }
    }
    
    // 获取推荐模型状态
    const recommendedStatus = {
      llm: RECOMMENDED_MODELS.llm.map(rec => ({
        ...rec,
        installed: llmModels.some(m => m.name.includes(rec.name.split(':')[0]))
      })),
      embedding: RECOMMENDED_MODELS.embedding.map(rec => ({
        ...rec,
        installed: embeddingModels.some(m => m.name.includes(rec.name.split(':')[0]))
      }))
    };
    
    // 检查是否有推荐模型已安装
    const hasRecommendedLLM = recommendedStatus.llm.some(m => m.installed);
    const hasRecommendedEmbedding = recommendedStatus.embedding.some(m => m.installed);
    
    return NextResponse.json({
      success: true,
      hasModels: true,
      llmModels,
      embeddingModels,
      unknownModels,
      allModels,
      count: {
        total: allModels.length,
        llm: llmModels.length,
        embedding: embeddingModels.length,
        unknown: unknownModels.length
      },
      recommended: recommendedStatus,
      status: {
        hasRecommendedLLM,
        hasRecommendedEmbedding,
        ready: hasRecommendedLLM && hasRecommendedEmbedding
      },
      warnings: [
        ...(!hasRecommendedLLM ? ['未检测到推荐的 LLM 模型，建议安装 Llama 3.1 或 Qwen 2.5'] : []),
        ...(!hasRecommendedEmbedding ? ['未检测到推荐的 Embedding 模型，建议安装 nomic-embed-text'] : []),
        ...(unknownModels.length > 0 ? [`检测到 ${unknownModels.length} 个未分类的模型`] : [])
      ]
    });
    
  } catch (error) {
    console.error('Failed to fetch Ollama models:', error);
    return NextResponse.json({
      success: false,
      error: '无法连接到 Ollama 服务',
      code: 'CONNECTION_ERROR',
      suggestion: '请检查 Ollama 是否正在运行，默认地址: http://localhost:11434'
    }, { status: 500 });
  }
}

// POST: 模型操作（拉取、删除等）
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, modelName } = body;
    
    if (action === 'pull') {
      // 触发模型拉取（异步）
      const response = await fetch(`${OLLAMA_BASE_URL}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName })
      });
      
      if (!response.ok) {
        throw new Error('Failed to initiate model pull');
      }
      
      return NextResponse.json({
        success: true,
        message: `正在下载模型: ${modelName}`,
        note: '下载过程可能需要几分钟，请稍后刷新查看'
      });
    }
    
    if (action === 'delete') {
      const response = await fetch(`${OLLAMA_BASE_URL}/api/delete`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName })
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete model');
      }
      
      return NextResponse.json({
        success: true,
        message: `已删除模型: ${modelName}`
      });
    }
    
    if (action === 'validate') {
      // 验证模型是否可用
      const response = await fetch(`${OLLAMA_BASE_URL}/api/show`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName })
      });
      
      return NextResponse.json({
        success: response.ok,
        available: response.ok,
        modelName
      });
    }
    
    return NextResponse.json({
      success: false,
      error: 'Unknown action'
    }, { status: 400 });
    
  } catch (error) {
    console.error('Model operation error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Operation failed'
    }, { status: 500 });
  }
}

// 格式化字节大小
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
