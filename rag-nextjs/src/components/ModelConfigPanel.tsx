'use client';

import React, { useState, useEffect } from 'react';

interface ModelConfigSummary {
  provider: string;
  llmModel: string;
  embeddingModel: string;
  reasoningModel: string;
  baseUrl: string;
  hasApiKey: boolean;
  registeredModels: Array<{
    id: string;
    type: string;
    provider: string;
    modelName: string;
    description?: string;
    createdAt: number;
  }>;
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
}

interface ModelConfigResponse {
  success: boolean;
  config: ModelConfigSummary;
  validation: ValidationResult;
  timestamp: string;
  error?: string;
}

export function ModelConfigPanel() {
  const [config, setConfig] = useState<ModelConfigSummary | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const fetchConfig = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/model-config');
      const data: ModelConfigResponse = await response.json();
      
      if (data.success) {
        setConfig(data.config);
        setValidation(data.validation);
        setError(null);
      } else {
        setError(data.error || '获取配置失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '网络错误');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  const handleReloadConfig = async () => {
    try {
      const response = await fetch('/api/model-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'reload' }),
      });
      const data = await response.json();
      
      if (data.success) {
        await fetchConfig();
      } else {
        setError(data.error || '重新加载配置失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '网络错误');
    }
  };

  const getProviderBadgeColor = (provider: string) => {
    switch (provider) {
      case 'ollama':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'openai':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'azure':
        return 'bg-purple-100 text-purple-800 border-purple-300';
      case 'custom':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getProviderIcon = (provider: string) => {
    switch (provider) {
      case 'ollama':
        return '🦙';
      case 'openai':
        return '🤖';
      case 'azure':
        return '☁️';
      case 'custom':
        return '🔧';
      default:
        return '❓';
    }
  };

  if (loading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
        <div className="flex items-center gap-2 text-slate-400">
          <div className="animate-spin w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full" />
          <span>加载配置中...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 rounded-lg p-4 border border-red-800">
        <div className="flex items-center gap-2 text-red-400">
          <span>❌</span>
          <span>{error}</span>
          <button
            onClick={fetchConfig}
            className="ml-auto px-2 py-1 text-xs bg-red-800 hover:bg-red-700 rounded"
          >
            重试
          </button>
        </div>
      </div>
    );
  }

  if (!config) return null;

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-700/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-lg">{getProviderIcon(config.provider)}</span>
          <span className="font-medium text-slate-200">模型配置</span>
          <span className={`px-2 py-0.5 text-xs font-medium rounded border ${getProviderBadgeColor(config.provider)}`}>
            {config.provider.toUpperCase()}
          </span>
          {validation && !validation.valid && (
            <span className="px-2 py-0.5 text-xs font-medium rounded bg-red-900/50 text-red-400 border border-red-700">
              ⚠️ 配置错误
            </span>
          )}
        </div>
        <span className="text-slate-400 text-sm">
          {expanded ? '▲' : '▼'}
        </span>
      </button>

      {/* Content */}
      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* 验证错误 */}
          {validation && !validation.valid && (
            <div className="bg-red-900/20 rounded p-3 border border-red-800">
              <div className="text-red-400 text-sm font-medium mb-2">配置错误:</div>
              <ul className="list-disc list-inside text-red-300 text-sm space-y-1">
                {validation.errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}

          {/* 模型信息 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="bg-slate-900/50 rounded p-3 border border-slate-700">
              <div className="text-xs text-slate-500 mb-1">LLM 模型</div>
              <div className="text-slate-200 font-mono text-sm truncate" title={config.llmModel}>
                {config.llmModel}
              </div>
            </div>
            <div className="bg-slate-900/50 rounded p-3 border border-slate-700">
              <div className="text-xs text-slate-500 mb-1">Embedding 模型</div>
              <div className="text-slate-200 font-mono text-sm truncate" title={config.embeddingModel}>
                {config.embeddingModel}
              </div>
            </div>
            <div className="bg-slate-900/50 rounded p-3 border border-slate-700">
              <div className="text-xs text-slate-500 mb-1">推理模型</div>
              <div className="text-slate-200 font-mono text-sm truncate" title={config.reasoningModel}>
                {config.reasoningModel}
              </div>
            </div>
          </div>

          {/* 连接信息 */}
          <div className="bg-slate-900/50 rounded p-3 border border-slate-700">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-slate-500 mb-1">API 端点</div>
                <div className="text-slate-200 font-mono text-sm truncate" title={config.baseUrl}>
                  {config.baseUrl}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {config.hasApiKey ? (
                  <span className="px-2 py-1 text-xs bg-green-900/50 text-green-400 rounded border border-green-700">
                    ✓ API Key 已配置
                  </span>
                ) : (
                  <span className="px-2 py-1 text-xs bg-yellow-900/50 text-yellow-400 rounded border border-yellow-700">
                    {config.provider === 'ollama' ? '本地模式' : '⚠️ 未配置 API Key'}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* 动态注册的模型 */}
          {config.registeredModels.length > 0 && (
            <div className="bg-slate-900/50 rounded p-3 border border-slate-700">
              <div className="text-xs text-slate-500 mb-2">动态注册的模型 ({config.registeredModels.length})</div>
              <div className="space-y-2">
                {config.registeredModels.map((model) => (
                  <div key={model.id} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 text-xs rounded ${
                        model.type === 'llm' ? 'bg-blue-900/50 text-blue-400' :
                        model.type === 'embedding' ? 'bg-green-900/50 text-green-400' :
                        'bg-purple-900/50 text-purple-400'
                      }`}>
                        {model.type}
                      </span>
                      <span className="text-slate-200 font-mono">{model.id}</span>
                    </div>
                    <span className="text-slate-500 text-xs">
                      {new Date(model.createdAt).toLocaleString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex items-center gap-2 pt-2">
            <button
              onClick={handleReloadConfig}
              className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-200 rounded transition-colors"
            >
              🔄 重新加载配置
            </button>
            <button
              onClick={fetchConfig}
              className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-200 rounded transition-colors"
            >
              ↻ 刷新
            </button>
          </div>

          {/* 配置说明 */}
          <div className="text-xs text-slate-500 pt-2 border-t border-slate-700">
            <p>通过环境变量 <code className="bg-slate-900 px-1 rounded">MODEL_PROVIDER</code> 控制使用本地 Ollama 或生产 API。</p>
            <p className="mt-1">可选值: <code className="bg-slate-900 px-1 rounded">ollama</code> | <code className="bg-slate-900 px-1 rounded">openai</code> | <code className="bg-slate-900 px-1 rounded">azure</code> | <code className="bg-slate-900 px-1 rounded">custom</code></p>
          </div>
        </div>
      )}
    </div>
  );
}
