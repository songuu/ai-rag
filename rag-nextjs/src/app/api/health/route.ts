import { NextResponse } from 'next/server';
import { getRagSystem } from '@/lib/rag-instance';

// GET /api/health - 系统健康检查
export async function GET() {
  try {
    const ragSystem = await getRagSystem();
    const status = ragSystem.getStatus();

    return NextResponse.json({
      success: true,
      ragSystem: {
        initialized: status.initialized,
        documentCount: status.documentCount,
        embeddingDimension: status.embeddingDimension
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('健康检查错误:', error);
    return NextResponse.json(
      { 
        success: false,
        error: '健康检查失败',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}