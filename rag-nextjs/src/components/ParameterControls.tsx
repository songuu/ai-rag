'use client';

import React, { useState } from 'react';

// 可用的 LLM 模型列表
export const LLM_MODELS = [
  { id: 'llama3.1', name: 'Llama 3.1', description: '高性能通用模型', provider: 'Meta' },
  { id: 'llama3.1:70b', name: 'Llama 3.1 70B', description: '大参数量版本', provider: 'Meta' },
  { id: 'llama3.2', name: 'Llama 3.2', description: '最新版本', provider: 'Meta' },
  { id: 'qwen2.5', name: 'Qwen 2.5', description: '阿里云通义千问', provider: 'Alibaba' },
  { id: 'qwen2.5:14b', name: 'Qwen 2.5 14B', description: '中等规模版本', provider: 'Alibaba' },
  { id: 'deepseek-r1:14b', name: 'DeepSeek R1 14B', description: '推理增强模型', provider: 'DeepSeek' },
  { id: 'deepseek-r1:32b', name: 'DeepSeek R1 32B', description: '大规模推理模型', provider: 'DeepSeek' },
  { id: 'mistral', name: 'Mistral', description: '高效欧洲模型', provider: 'Mistral AI' },
  { id: 'mixtral', name: 'Mixtral MoE', description: '混合专家模型', provider: 'Mistral AI' },
  { id: 'gemma2', name: 'Gemma 2', description: 'Google 开源模型', provider: 'Google' },
  { id: 'phi3', name: 'Phi-3', description: '小型高效模型', provider: 'Microsoft' },
];

// 可用的 Embedding 模型列表
export const EMBEDDING_MODELS = [
  { id: 'nomic-embed-text', name: 'Nomic Embed', description: '通用文本嵌入', dim: 768 },
  { id: 'mxbai-embed-large', name: 'MxBAI Large', description: '大规模嵌入模型', dim: 1024 },
  { id: 'bge-m3', name: 'BGE-M3', description: '多语言嵌入', dim: 1024 },
  { id: 'bge-large', name: 'BGE Large', description: '智源大规模嵌入', dim: 1024 },
  { id: 'snowflake-arctic-embed', name: 'Snowflake Arctic', description: '企业级嵌入', dim: 1024 },
  { id: 'all-minilm', name: 'All-MiniLM', description: '轻量级嵌入', dim: 384 },
];

interface ParameterControlsProps {
  topK: number;
  threshold: number;
  llmModel: string;
  embeddingModel: string;
  onTopKChange: (value: number) => void;
  onThresholdChange: (value: number) => void;
  onLLMModelChange: (value: string) => void;
  onEmbeddingModelChange: (value: string) => void;
  showParams: boolean;
  onToggle: () => void;
}

export default function ParameterControls({
  topK,
  threshold,
  llmModel,
  embeddingModel,
  onTopKChange,
  onThresholdChange,
  onLLMModelChange,
  onEmbeddingModelChange,
  showParams,
  onToggle
}: ParameterControlsProps) {
  const [activeTab, setActiveTab] = useState<'retrieval' | 'model'>('retrieval');

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

  const selectedLLM = LLM_MODELS.find(m => m.id === llmModel);
  const selectedEmbed = EMBEDDING_MODELS.find(m => m.id === embeddingModel);

  return (
    <div className="mb-4 p-4 bg-gray-50 rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('retrieval')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              activeTab === 'retrieval' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-600 hover:bg-gray-100 border border-gray-200'
            }`}
          >
            检索参数
          </button>
          <button
            onClick={() => setActiveTab('model')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              activeTab === 'model' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-600 hover:bg-gray-100 border border-gray-200'
            }`}
          >
            模型选择
          </button>
        </div>
        <button 
          onClick={onToggle}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          <i className="fas fa-chevron-up"></i> 收起
        </button>
      </div>

      {activeTab === 'retrieval' && (
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
      )}

      {activeTab === 'model' && (
        <div className="space-y-4">
          {/* LLM 模型选择 */}
          <div>
            <label className="block text-xs text-gray-600 mb-2">
              <span className="font-medium">LLM 模型</span>
              {selectedLLM && (
                <span className="ml-2 text-gray-400">
                  ({selectedLLM.provider})
                </span>
              )}
            </label>
            <div className="grid grid-cols-3 gap-2 max-h-32 overflow-y-auto">
              {LLM_MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => onLLMModelChange(model.id)}
                  className={`p-2 text-left rounded-md border transition-all ${
                    llmModel === model.id
                      ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500'
                      : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="text-xs font-medium text-gray-800 truncate">
                    {model.name}
                  </div>
                  <div className="text-[10px] text-gray-500 truncate">
                    {model.description}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Embedding 模型选择 */}
          <div>
            <label className="block text-xs text-gray-600 mb-2">
              <span className="font-medium">Embedding 模型</span>
              {selectedEmbed && (
                <span className="ml-2 text-gray-400">
                  ({selectedEmbed.dim} 维)
                </span>
              )}
            </label>
            <div className="grid grid-cols-3 gap-2">
              {EMBEDDING_MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => onEmbeddingModelChange(model.id)}
                  className={`p-2 text-left rounded-md border transition-all ${
                    embeddingModel === model.id
                      ? 'border-green-500 bg-green-50 ring-1 ring-green-500'
                      : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="text-xs font-medium text-gray-800 truncate">
                    {model.name}
                  </div>
                  <div className="text-[10px] text-gray-500 truncate">
                    {model.description}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* 当前选择摘要 */}
          <div className="mt-3 p-2 bg-white rounded border border-gray-200">
            <div className="text-[10px] text-gray-500 mb-1">当前配置</div>
            <div className="flex gap-4 text-xs">
              <div>
                <span className="text-gray-500">LLM:</span>{' '}
                <span className="font-medium text-blue-600">{selectedLLM?.name || llmModel}</span>
              </div>
              <div>
                <span className="text-gray-500">Embedding:</span>{' '}
                <span className="font-medium text-green-600">{selectedEmbed?.name || embeddingModel}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}