import { NextRequest, NextResponse } from 'next/server';
import { unlink } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');

// DELETE /api/files/[filename] - 删除文件
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  try {
    const { filename } = await params;
    const decodedFilename = decodeURIComponent(filename);
    
    // 安全验证：确保文件名不包含路径遍历字符
    if (decodedFilename.includes('..') || decodedFilename.includes('/') || decodedFilename.includes('\\')) {
      return NextResponse.json(
        { error: '无效的文件名' },
        { status: 400 }
      );
    }

    const filePath = path.join(UPLOAD_DIR, decodedFilename);

    if (!existsSync(filePath)) {
      return NextResponse.json(
        { error: '文件不存在' },
        { status: 404 }
      );
    }

    await unlink(filePath);

    return NextResponse.json({
      success: true,
      message: `文件 "${decodedFilename}" 删除成功`
    });
  } catch (error) {
    console.error('删除文件错误:', error);
    return NextResponse.json(
      { 
        error: '删除文件失败',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}