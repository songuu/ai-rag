import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { 
  parseDocument, 
  isSupportedFile, 
  getSupportedTypesDescription,
  SUPPORTED_EXTENSIONS 
} from '@/lib/document-parser';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

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
        if (!isSupportedFile(file.name)) {
          errors.push({
            filename: file.name,
            error: `不支持的文件类型。${getSupportedTypesDescription()}`
          });
          continue;
        }

        // 验证文件大小
        if (file.size > MAX_FILE_SIZE) {
          errors.push({
            filename: file.name,
            error: `文件太大，最大支持 ${MAX_FILE_SIZE / 1024 / 1024}MB`
          });
          continue;
        }

        // 读取文件内容
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);
        
        // 解析文档内容
        const parseResult = await parseDocument(buffer, file.name);
        
        if (!parseResult.success || !parseResult.document) {
          errors.push({
            filename: file.name,
            error: parseResult.error || '文件解析失败'
          });
          continue;
        }

        // 生成安全的文件名（处理原始文件）
        const timestamp = Date.now();
        const safeFilename = `${timestamp}_${file.name.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
        const originalFilePath = path.join(UPLOAD_DIR, safeFilename);
        
        // 保存原始文件
        await writeFile(originalFilePath, buffer);
        
        // 同时保存解析后的文本内容（.txt 格式）
        const textFilename = `${timestamp}_${path.basename(file.name, path.extname(file.name))}_parsed.txt`;
        const textFilePath = path.join(UPLOAD_DIR, textFilename);
        await writeFile(textFilePath, parseResult.document.content, 'utf-8');

        results.push({
          filename: file.name,
          savedAs: safeFilename,
          textFile: textFilename,
          size: file.size,
          contentLength: parseResult.document.content.length,
          metadata: parseResult.document.metadata,
          path: originalFilePath
        });

      } catch (error) {
        console.error(`处理文件 ${file.name} 时出错:`, error);
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
      errors: errors.length > 0 ? errors : undefined,
      supportedTypes: SUPPORTED_EXTENSIONS
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

// GET 端点返回支持的文件类型
export async function GET() {
  return NextResponse.json({
    supportedExtensions: SUPPORTED_EXTENSIONS,
    description: getSupportedTypesDescription(),
    maxSize: MAX_FILE_SIZE,
    maxSizeFormatted: `${MAX_FILE_SIZE / 1024 / 1024}MB`
  });
}
