/*
 * @Author: songyu
 * @Date: 2026-01-12 20:38:58
 * @LastEditTime: 2026-01-14 21:05:30
 * @LastEditor: songyu
 */
'use client';

import React, { useState, useEffect } from 'react';

interface SystemInfoProps {
  docCount: number;
  embeddingDim: number;
  systemStatus: string;
  llmModel: string;
  embeddingModel: string;
  onReinitialize: () => void;
  onModelChange: (llmModel: string, embeddingModel: string) => void;
}

interface ModelInfo {
  name: string;
  displayName: string;
  category: string;
  sizeFormatted?: string;
  tag?: string;
}

export default function SystemInfo({ 
  docCount, 
  embeddingDim, 
  systemStatus, 
  llmModel,
  embeddingModel,
  onReinitialize,
  onModelChange
}: SystemInfoProps) {
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [availableModels, setAvailableModels] = useState<any>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [selectedLLM, setSelectedLLM] = useState(llmModel);
  const [selectedEmbedding, setSelectedEmbedding] = useState(embeddingModel);

  // 加载可用模型
  const loadModels = async () => {
    setLoadingModels(true);
    try {
      const response = await fetch('/api/ollama/models');
      const data = await response.json();
      setAvailableModels(data);
    } catch (error) {
      console.error('Failed to load models:', error);
    } finally {
      setLoadingModels(false);
    }
  };

  // 打开模型选择器时加载模型
  useEffect(() => {
    if (showModelSelector) {
      loadModels();
      setSelectedLLM(llmModel);
      setSelectedEmbedding(embeddingModel);
    }
  }, [showModelSelector, llmModel, embeddingModel]);

  // 应用模型变更
  const handleApplyModelChange = () => {
    if (selectedLLM !== llmModel || selectedEmbedding !== embeddingModel) {
      onModelChange(selectedLLM, selectedEmbedding);
      setShowModelSelector(false);
    } else {
      setShowModelSelector(false);
    }
  };

  // 格式化模型名称
  const formatModelName = (name: string) => {
    return name.split(':')[0];
  };

  return (
    <>
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
            <span className={`font-medium ${
              systemStatus === '运行中' ? 'text-green-600' : 
              systemStatus === '重新初始化中...' ? 'text-yellow-600' : 
              'text-gray-600'
            }`}>{systemStatus}</span>
          </div>
          
          {/* 模型信息 - 可点击切换 */}
          <div className="pt-3 border-t">
            <div className="flex justify-between items-center text-sm mb-2">
              <span className="text-gray-600">LLM 模型:</span>
              <span className="font-medium text-xs text-purple-700" title={llmModel}>
                {formatModelName(llmModel)}
              </span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-600">嵌入模型:</span>
              <span className="font-medium text-xs text-blue-700" title={embeddingModel}>
                {formatModelName(embeddingModel)}
              </span>
            </div>
            
            <button
              onClick={() => setShowModelSelector(true)}
              className="w-full mt-3 px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-sm transition-colors"
            >
              <i className="fas fa-exchange-alt mr-2"></i>
              切换模型
            </button>
          </div>
          
          <button 
            onClick={onReinitialize}
            disabled={systemStatus === '重新初始化中...'}
            className="w-full mt-4 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <i className="fas fa-redo mr-2"></i>
            重新初始化
          </button>
        </div>
      </div>

      {/* 模型选择模态框 */}
      {showModelSelector && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            {/* 标题 */}
            <div className="px-6 py-4 border-b flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">选择模型</h3>
              <button
                onClick={() => setShowModelSelector(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* 内容 */}
            <div className="flex-1 overflow-y-auto p-6">
              {loadingModels ? (
                <div className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="animate-spin h-8 w-8 border-4 border-purple-600 border-t-transparent rounded-full mx-auto mb-3"></div>
                    <p className="text-sm text-gray-500">正在加载模型列表...</p>
                  </div>
                </div>
              ) : availableModels && availableModels.success ? (
                <div className="space-y-6">
                  {/* LLM 模型选择 */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      🤖 LLM 模型 ({availableModels.llmModels?.length || 0})
                    </label>
                    {availableModels.llmModels && availableModels.llmModels.length > 0 ? (
                      <div className="grid grid-cols-2 gap-3">
                        {availableModels.llmModels.map((model: ModelInfo) => (
                          <button
                            key={model.name}
                            onClick={() => setSelectedLLM(model.name)}
                            className={`p-3 rounded-lg border-2 text-left transition-all ${
                              selectedLLM === model.name
                                ? 'border-purple-500 bg-purple-50'
                                : 'border-gray-200 hover:border-purple-300 bg-white'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-sm text-gray-900 truncate">
                                  {model.displayName}
                                </div>
                                <div className="text-xs text-gray-500 mt-0.5">
                                  {model.sizeFormatted || model.tag}
                                </div>
                              </div>
                              {selectedLLM === model.name && (
                                <svg className="w-5 h-5 text-purple-600 ml-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="text-sm text-gray-500 bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-center">
                        未检测到 LLM 模型，请先安装模型
                      </div>
                    )}
                  </div>

                  {/* Embedding 模型选择 */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      🧬 Embedding 模型 ({availableModels.embeddingModels?.length || 0})
                    </label>
                    {availableModels.embeddingModels && availableModels.embeddingModels.length > 0 ? (
                      <div className="grid grid-cols-2 gap-3">
                        {availableModels.embeddingModels.map((model: ModelInfo) => (
                          <button
                            key={model.name}
                            onClick={() => setSelectedEmbedding(model.name)}
                            className={`p-3 rounded-lg border-2 text-left transition-all ${
                              selectedEmbedding === model.name
                                ? 'border-blue-500 bg-blue-50'
                                : 'border-gray-200 hover:border-blue-300 bg-white'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-sm text-gray-900 truncate">
                                  {model.displayName}
                                </div>
                                <div className="text-xs text-gray-500 mt-0.5">
                                  {model.sizeFormatted || model.tag}
                                </div>
                              </div>
                              {selectedEmbedding === model.name && (
                                <svg className="w-5 h-5 text-blue-600 ml-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="text-sm text-gray-500 bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-center">
                        未检测到 Embedding 模型，请先安装模型
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-red-600 mb-4">
                    <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">{availableModels?.error || '无法加载模型列表'}</p>
                  <button
                    onClick={loadModels}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-sm transition-colors"
                  >
                    重试
                  </button>
                </div>
              )}
            </div>

            {/* 底部按钮 */}
            <div className="px-6 py-4 border-t bg-gray-50 flex items-center justify-between">
              <div className="text-xs text-gray-500">
                {selectedLLM !== llmModel || selectedEmbedding !== embeddingModel ? (
                  <span className="text-yellow-600 font-medium">⚠️ 应用后将重新初始化系统</span>
                ) : (
                  <span>未做任何更改</span>
                )}
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowModelSelector(false)}
                  className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-100 text-sm transition-colors"
                >
                  取消
                </button>
                <button
                  onClick={handleApplyModelChange}
                  disabled={!availableModels?.success || (selectedLLM === llmModel && selectedEmbedding === embeddingModel)}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  应用更改
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}