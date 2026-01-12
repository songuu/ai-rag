'use client';

import React from 'react';

interface ParameterControlsProps {
  topK: number;
  threshold: number;
  onTopKChange: (value: number) => void;
  onThresholdChange: (value: number) => void;
  showParams: boolean;
  onToggle: () => void;
}

export default function ParameterControls({
  topK,
  threshold,
  onTopKChange,
  onThresholdChange,
  showParams,
  onToggle
}: ParameterControlsProps) {
  if (!showParams) {
    return (
      <div className="mb-4">
        <button 
          onClick={onToggle}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          <i className="fas fa-chevron-down mr-1"></i>展开参数设置
        </button>
      </div>
    );
  }

  return (
    <div className="mb-4 p-4 bg-gray-50 rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-700">检索参数</h4>
        <button 
          onClick={onToggle}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          <i className="fas fa-chevron-up"></i> 收起
        </button>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-600 mb-1">Top-K 文档数</label>
          <input 
            type="range" 
            min="1" 
            max="100" 
            value={topK} 
            onChange={(e) => onTopKChange(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>1</span>
            <span className="font-medium">{topK}</span>
            <span>100</span>
          </div>
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">相似度阈值</label>
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.01" 
            value={threshold} 
            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>0.0</span>
            <span className="font-medium">{threshold.toFixed(2)}</span>
            <span>1.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}