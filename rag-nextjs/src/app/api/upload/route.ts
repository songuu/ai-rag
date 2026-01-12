import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files = formData.getAll('files') as File[];

    if (files.length === 0) {
      return NextResponse.json(
        { error: '请选择要上传的文件' },
        { status: 400 }
      );
    }

    // 确保上传目录存在
    if (!existsSync(UPLOAD_DIR)) {
      await mkdir(UPLOAD_DIR, { recursive: true });
    }

    const results = [];
    const errors = [];

    for (const file of files) {
      try {
        // 验证文件类型
        if (!file.name.endsWith('.txt')) {
          errors.push({
            filename: file.name,
            error: '只支持 .txt 文本文件'
          });
          continue;
        }

        // 验证文件大小 (5MB)
        if (file.size > 5 * 1024 * 1024) {
          errors.push({
            filename: file.name,
            error: '文件太大，最大支持 5MB'
          });
          continue;
        }

        // 保存文件
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);
        const filePath = path.join(UPLOAD_DIR, file.name);
        
        await writeFile(filePath, buffer);

        results.push({
          filename: file.name,
          size: file.size,
          path: filePath
        });
      } catch (error) {
        errors.push({
          filename: file.name,
          error: error instanceof Error ? error.message : '上传失败'
        });
      }
    }

    return NextResponse.json({
      success: true,
      message: `成功上传 ${results.length} 个文件`,
      results,
      errors: errors.length > 0 ? errors : undefined
    });

  } catch (error) {
    console.error('文件上传错误:', error);
    return NextResponse.json(
      { 
        error: '处理文件上传时发生错误',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}