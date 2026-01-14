import { NextResponse } from 'next/server';
import { getRagSystem } from '@/lib/rag-instance';
import { readdir, readFile } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');

// POST /api/reinitialize - 重新初始化 RAG 系统
export async function POST() {
  try {
    const ragSystem = await getRagSystem();
    
    // 重新加载所有上传的文件
    if (existsSync(UPLOAD_DIR)) {
      const files = await readdir(UPLOAD_DIR);
      const txtFiles = files.filter(f => f.endsWith('.txt'));
      
      const documents: Array<{ content: string; filename: string }> = [];
      for (const filename of txtFiles) {
        const filePath = path.join(UPLOAD_DIR, filename);
        const content = await readFile(filePath, 'utf-8');
        if (content.trim()) {
          documents.push({ content, filename });
        }
      }
      
      console.log(`[Reinitialize] 找到 ${documents.length} 个有效文档`);
      
      // 重新初始化系统
      await ragSystem.reinitialize(documents);
    } else {
      // 如果没有文件，清空系统
      await ragSystem.reinitialize([]);
    }

    return NextResponse.json({
      success: true,
      message: '系统重新初始化成功'
    });
  } catch (error) {
    console.error('重新初始化错误:', error);
    return NextResponse.json(
      { 
        error: '重新初始化失败',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}