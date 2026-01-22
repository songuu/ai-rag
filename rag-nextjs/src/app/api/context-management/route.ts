import { NextResponse } from 'next/server';
import { createContextManager, ContextManager, ContextManagerConfig } from '@/lib/context-management';

// 全局实例
let contextManager: ContextManager | null = null;

function getContextManager(config?: Partial<ContextManagerConfig>): ContextManager {
  if (!contextManager || config) {
    contextManager = createContextManager(config);
  }
  return contextManager;
}

/**
 * GET - 获取会话列表、详情、配置等
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const sessionId = searchParams.get('sessionId');
    
    const manager = getContextManager();
    
    switch (action) {
      case 'sessions': {
        const sessions = await manager.listSessions();
        return NextResponse.json({
          success: true,
          sessions,
          count: sessions.length,
        });
      }
      
      case 'session': {
        if (!sessionId) {
          return NextResponse.json({ success: false, error: '缺少 sessionId' }, { status: 400 });
        }
        
        const state = await manager.getSession(sessionId);
        if (!state) {
          return NextResponse.json({ success: false, error: '会话不存在' }, { status: 404 });
        }
        
        return NextResponse.json({
          success: true,
          session: state,
          tokenStats: manager.getTokenStats(state),
        });
      }
      
      case 'config': {
        return NextResponse.json({
          success: true,
          config: manager.getConfig(),
        });
      }
      
      case 'status': {
        const sessions = await manager.listSessions();
        return NextResponse.json({
          success: true,
          status: {
            totalSessions: sessions.length,
            activeSessions: sessions.filter(s => Date.now() - s.lastActiveAt < 3600000).length,
            config: manager.getConfig(),
          },
        });
      }
      
      default:
        return NextResponse.json({ success: false, error: '未知操作' }, { status: 400 });
    }
  } catch (error) {
    console.error('[ContextManagement API] GET error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : '未知错误' },
      { status: 500 }
    );
  }
}

/**
 * POST - 处理查询、创建会话等
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, ...params } = body;
    
    let manager = getContextManager();
    
    switch (action) {
      case 'query': {
        const {
          sessionId,
          question,
          userId,
          topK = 5,
          similarityThreshold = 0.3,
          llmModel,
          embeddingModel,
          windowStrategy,
          maxRounds,
          maxTokens,
          enableQueryRewriting = true,
        } = params;
        
        if (!sessionId || !question) {
          return NextResponse.json(
            { success: false, error: '缺少 sessionId 或 question' },
            { status: 400 }
          );
        }
        
        // 更新配置
        if (llmModel || embeddingModel || windowStrategy) {
          const newConfig: Partial<ContextManagerConfig> = {};
          if (llmModel) newConfig.llmModel = llmModel;
          if (embeddingModel) newConfig.embeddingModel = embeddingModel;
          if (windowStrategy || maxRounds || maxTokens) {
            newConfig.windowConfig = {
              ...manager.getConfig().windowConfig,
              ...(windowStrategy && { strategy: windowStrategy }),
              ...(maxRounds && { maxRounds }),
              ...(maxTokens && { maxTokens }),
            };
          }
          newConfig.enableQueryRewriting = enableQueryRewriting;
          manager.updateConfig(newConfig);
        }
        
        console.log(`[ContextManagement] Query: sessionId=${sessionId}, question="${question}"`);
        
        const result = await manager.processQuery(sessionId, question, {
          userId,
          topK,
          similarityThreshold,
        });
        
        return NextResponse.json({
          success: true,
          response: result.response,
          rewrittenQuery: result.rewrittenQuery,
          retrievedDocs: result.retrievedDocs,
          workflow: {
            steps: result.workflowSteps,
            totalDuration: result.workflowSteps.reduce((sum, s) => sum + (s.duration || 0), 0),
          },
          sessionInfo: {
            sessionId: result.state.metadata.sessionId,
            messageCount: result.state.metadata.messageCount,
            totalTokens: result.state.metadata.totalTokens,
            truncatedCount: result.state.metadata.truncatedCount,
          },
        });
      }
      
      case 'create-session': {
        const { userId } = params;
        const state = await manager.createSession(userId);
        
        return NextResponse.json({
          success: true,
          session: state,
          message: '会话创建成功',
        });
      }
      
      case 'delete-session': {
        const { sessionId } = params;
        if (!sessionId) {
          return NextResponse.json({ success: false, error: '缺少 sessionId' }, { status: 400 });
        }
        
        const deleted = await manager.deleteSession(sessionId);
        return NextResponse.json({
          success: deleted,
          message: deleted ? '删除成功' : '会话不存在',
        });
      }
      
      case 'compress': {
        const { sessionId } = params;
        if (!sessionId) {
          return NextResponse.json({ success: false, error: '缺少 sessionId' }, { status: 400 });
        }
        
        const result = await manager.compressBySummary(sessionId);
        return NextResponse.json({
          success: result.success,
          summary: result.summary,
          compressedCount: result.compressedCount,
          message: result.success ? '压缩成功' : '无需压缩或压缩失败',
        });
      }
      
      case 'update-config': {
        const { config } = params;
        if (!config) {
          return NextResponse.json({ success: false, error: '缺少 config' }, { status: 400 });
        }
        
        manager.updateConfig(config);
        return NextResponse.json({
          success: true,
          config: manager.getConfig(),
          message: '配置更新成功',
        });
      }
      
      default:
        return NextResponse.json({ success: false, error: '未知操作' }, { status: 400 });
    }
  } catch (error) {
    console.error('[ContextManagement API] POST error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : '未知错误' },
      { status: 500 }
    );
  }
}

/**
 * DELETE - 删除会话
 */
export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get('sessionId');
    
    if (!sessionId) {
      return NextResponse.json({ success: false, error: '缺少 sessionId' }, { status: 400 });
    }
    
    const manager = getContextManager();
    const deleted = await manager.deleteSession(sessionId);
    
    return NextResponse.json({
      success: deleted,
      message: deleted ? '删除成功' : '会话不存在',
    });
  } catch (error) {
    console.error('[ContextManagement API] DELETE error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : '未知错误' },
      { status: 500 }
    );
  }
}
