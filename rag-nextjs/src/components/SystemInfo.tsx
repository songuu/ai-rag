'use client';

import React from 'react';

interface SystemInfoProps {
  docCount: number;
  embeddingDim: number;
  systemStatus: string;
  onReinitialize: () => void;
}

export default function SystemInfo({ docCount, embeddingDim, systemStatus, onReinitialize }: SystemInfoProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="border-b px-6 py-4">
        <h3 className="text-lg font-medium text-gray-900">系统信息</h3>
      </div>
      
      <div className="p-6 space-y-3">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">文档数量:</span>
          <span className="font-medium">{docCount || '-'}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">向量维度:</span>
          <span className="font-medium">{embeddingDim || '-'}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">系统状态:</span>
          <span className={`font-medium ${systemStatus === '运行中' ? 'text-green-600' : ''}`}>{systemStatus}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">模型:</span>
          <span className="font-medium text-xs">llama3.1</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">嵌入模型:</span>
          <span className="font-medium text-xs">nomic-embed-text</span>
        </div>
        
        <button 
          onClick={onReinitialize}
          className="w-full mt-4 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 text-sm"
        >
          <i className="fas fa-redo mr-2"></i>
          重新初始化
        </button>
      </div>
    </div>
  );
}