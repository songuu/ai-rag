'use client';

import React, { useRef } from 'react';

interface FileUploadProps {
  selectedFiles: File[];
  isUploading: boolean;
  onFileSelect: (files: File[]) => void;
  onUpload: () => void;
}

export default function FileUpload({ selectedFiles, isUploading, onFileSelect, onUpload }: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files) {
      onFileSelect(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      onFileSelect(Array.from(e.target.files));
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="border-b px-6 py-4">
        <h3 className="text-lg font-medium text-gray-900">文档管理</h3>
        <p className="text-sm text-gray-500 mt-1">上传文本文件到知识库</p>
      </div>
      
      <div className="p-6">
        <div 
          className="file-upload-area rounded-lg p-6 text-center mb-4 cursor-pointer border-2 border-dashed border-gray-300 hover:border-blue-500 hover:bg-blue-50 transition-colors"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <i className="fas fa-cloud-upload-alt text-3xl text-gray-400 mb-2"></i>
          <p className="text-sm text-gray-600 mb-2">拖拽文件到此处或点击选择</p>
          <p className="text-xs text-gray-500">支持 .txt 文件，最大 5MB</p>
          <input 
            ref={fileInputRef}
            type="file" 
            accept=".txt" 
            multiple 
            className="hidden"
            onChange={handleFileSelect}
          />
        </div>
        
        {selectedFiles.length > 0 && (
          <div className="mb-4 text-xs text-gray-600">
            已选择 {selectedFiles.length} 个文件
          </div>
        )}
        
        <button 
          onClick={onUpload}
          disabled={selectedFiles.length === 0 || isUploading}
          className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <i className="fas fa-upload mr-2"></i>
          {isUploading ? '上传中...' : '上传文件'}
        </button>
      </div>
    </div>
  );
}