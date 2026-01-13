'use client';

import React from 'react';

interface ParameterControlsProps {
  topK: number;
  threshold: number;
  tokenizerModel: string;
  onTopKChange: (value: number) => void;
  onThresholdChange: (value: number) => void;
  onTokenizerModelChange: (value: string) => void;
  showParams: boolean;
  onToggle: () => void;
}

const SUPPORTED_MODELS = [
  { value: 'Xenova/bert-base-multilingual-cased', label: 'BERT Multilingual (多语言)' },
  { value: 'Xenova/bge-small-zh-v1.5', label: 'BGE Small ZH (中文优化)' },
  { value: 'Xenova/all-MiniLM-L6-v2', label: 'MiniLM L6 v2 (轻量级)' }
];

export default function ParameterControls({
  topK,
  threshold,
  tokenizerModel,
  onTopKChange,
  onThresholdChange,
  onTokenizerModelChange,
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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
        <div>
          <label className="block text-xs text-gray-600 mb-1">Tokenizer 模型</label>
          <select
            value={tokenizerModel}
            onChange={(e) => onTokenizerModelChange(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {SUPPORTED_MODELS.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          <div className="text-xs text-gray-500 mt-1">
            当前: {SUPPORTED_MODELS.find(m => m.value === tokenizerModel)?.label || tokenizerModel}
          </div>
        </div>
      </div>
    </div>
  );
}