import { NextRequest, NextResponse } from 'next/server';
import { readdir, stat } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');

// GET /api/files - 获取文件列表
export async function GET() {
  try {
    if (!existsSync(UPLOAD_DIR)) {
      return NextResponse.json({
        success: true,
        files: []
      });
    }

    const files = await readdir(UPLOAD_DIR);
    const fileList = await Promise.all(
      files
        .filter(file => file.endsWith('.txt'))
        .map(async (filename) => {
          const filePath = path.join(UPLOAD_DIR, filename);
          const stats = await stat(filePath);
          return {
            name: filename,
            size: stats.size,
            modified: stats.mtime.toISOString()
          };
        })
    );

    return NextResponse.json({
      success: true,
      files: fileList
    });
  } catch (error) {
    console.error('获取文件列表错误:', error);
    return NextResponse.json(
      { 
        error: '获取文件列表失败',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}