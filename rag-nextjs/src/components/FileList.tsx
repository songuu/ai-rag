'use client';

import React from 'react';

interface FileInfo {
  name: string;
  size: number;
  modified: string;
}

interface FileListProps {
  files: FileInfo[];
  onRefresh: () => void;
  onDelete: (filename: string) => void;
  formatFileSize: (bytes: number) => string;
}

export default function FileList({ files, onRefresh, onDelete, formatFileSize }: FileListProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="border-b px-6 py-4 flex justify-between items-center">
        <div>
          <h3 className="text-lg font-medium text-gray-900">已上传文件</h3>
          <p className="text-sm text-gray-500 mt-1">管理知识库中的文档</p>
        </div>
        <button 
          onClick={onRefresh}
          className="p-2 text-gray-400 hover:text-gray-600"
        >
          <i className="fas fa-sync-alt"></i>
        </button>
      </div>
      
      <div className="p-6">
        {files.length === 0 ? (
          <div className="text-center text-gray-500 text-sm">
            <i className="fas fa-folder-open text-2xl mb-2"></i>
            <p>暂无文件</p>
          </div>
        ) : (
          <div className="space-y-2">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                  <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                </div>
                <button
                  onClick={() => onDelete(file.name)}
                  className="ml-2 p-1 text-red-600 hover:text-red-800"
                >
                  <i className="fas fa-trash"></i>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}