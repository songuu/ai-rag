import { NextRequest, NextResponse } from 'next/server';
import {
  getModelFactory,
  getConfigSummary,
  ModelProvider,
  ModelType,
  DynamicModelEntry,
} from '@/lib/model-config';

/**
 * 获取当前模型配置
 */
export async function GET() {
  try {
    const factory = getModelFactory();
    const summary = getConfigSummary();
    const validation = factory.validateConfig();
    const registeredModels = factory.getRegisteredModels();

    return NextResponse.json({
      success: true,
      config: {
        ...summary,
        registeredModels: registeredModels.map(m => ({
          id: m.id,
          type: m.type,
          provider: m.config.provider,
          modelName: m.config.modelName,
          description: m.description,
          createdAt: m.createdAt,
        })),
      },
      validation,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('[Model Config API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: '获取配置失败',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

/**
 * 动态注册模型
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    const factory = getModelFactory();

    switch (action) {
      case 'register': {
        const { id, type, config, description } = body;

        if (!id || !type || !config) {
          return NextResponse.json(
            { success: false, error: '缺少必要参数: id, type, config' },
            { status: 400 }
          );
        }

        // 验证类型
        if (!['llm', 'embedding', 'reasoning'].includes(type)) {
          return NextResponse.json(
            { success: false, error: '无效的模型类型，必须是 llm, embedding 或 reasoning' },
            { status: 400 }
          );
        }

        // 验证提供商
        if (!['ollama', 'openai', 'azure', 'custom'].includes(config.provider)) {
          return NextResponse.json(
            { success: false, error: '无效的提供商，必须是 ollama, openai, azure 或 custom' },
            { status: 400 }
          );
        }

        factory.registerModel({
          id,
          type: type as ModelType,
          config: {
            provider: config.provider as ModelProvider,
            modelName: config.modelName,
            apiKey: config.apiKey,
            baseUrl: config.baseUrl,
            temperature: config.temperature,
            maxTokens: config.maxTokens,
            dimension: config.dimension,
            options: config.options,
          },
          description,
        });

        return NextResponse.json({
          success: true,
          message: `模型 ${id} 注册成功`,
          model: { id, type, config: { ...config, apiKey: config.apiKey ? '***' : undefined } },
        });
      }

      case 'unregister': {
        const { id } = body;
        if (!id) {
          return NextResponse.json(
            { success: false, error: '缺少模型 ID' },
            { status: 400 }
          );
        }

        const deleted = factory.getRegisteredModels().some(m => m.id === id);
        if (!deleted) {
          return NextResponse.json(
            { success: false, error: `模型 ${id} 不存在` },
            { status: 404 }
          );
        }

        // 注意：当前实现中 unregister 方法在 ModelRegistry 中是私有的
        // 这里我们返回一个提示
        return NextResponse.json({
          success: false,
          error: '暂不支持动态注销模型，请重启服务',
        });
      }

      case 'reload': {
        factory.reloadConfig();
        const newSummary = getConfigSummary();
        
        return NextResponse.json({
          success: true,
          message: '配置已重新加载',
          config: newSummary,
        });
      }

      case 'clear-cache': {
        factory.clearCache();
        
        return NextResponse.json({
          success: true,
          message: '模型缓存已清空',
        });
      }

      default:
        return NextResponse.json(
          { success: false, error: `未知操作: ${action}` },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('[Model Config API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: '操作失败',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
