import { NextRequest, NextResponse } from 'next/server';
import { getRagSystem } from '@/lib/rag-instance';

// GET /api/traces - 获取所有 Traces
export async function GET() {
  try {
    const ragSystem = await getRagSystem();
    const observabilityData = ragSystem.getObservabilityData();
    
    return NextResponse.json({
      success: true,
      traces: observabilityData.traces,
      stats: observabilityData.stats
    });
  } catch (error) {
    console.error("获取 Traces 错误:", error);
    return NextResponse.json(
      { 
        error: "获取 Traces 失败",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

// DELETE /api/traces - 清除所有 Traces
export async function DELETE() {
  try {
    const ragSystem = await getRagSystem();
    ragSystem.clearObservabilityData();
    
    return NextResponse.json({
      success: true,
      message: "可观测性数据已清除"
    });
  } catch (error) {
    console.error("清除数据错误:", error);
    return NextResponse.json(
      { 
        error: "清除数据失败",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}