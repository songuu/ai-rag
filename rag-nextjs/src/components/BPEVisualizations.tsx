'use client';

import React from 'react';
import dynamic from 'next/dynamic';

const ReactECharts = dynamic(() => import('echarts-for-react'), { ssr: false });

interface ProcessingStep {
  step: number;
  stage: 'preprocessing' | 'trie_lookup' | 'bpe_merge' | 'subword_split' | 'finalization';
  action: string;
  input: string;
  output: string;
  decision: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

interface VectorWeight {
  token: string;
  tokenId: number;
  weight: number;
  position: number;
  contribution: number;
}

interface DensityPoint {
  position: number;
  token: string;
  density: number;
  tokenCount: number;
  contextWindow: number;
}

interface BPEVisualizationsProps {
  processingSteps?: ProcessingStep[];
  vectorWeights?: VectorWeight[];
  densityHeatmap?: DensityPoint[];
  statistics?: {
    totalTokens: number;
    uniqueTokens: number;
    subwordRatio: number;
    averageTokenLength: number;
    processingTime: number;
  };
  modelInfo?: {
    name: string;
    vocabSize: number;
    mergesCount: number;
  };
}

export default function BPEVisualizations({
  processingSteps = [],
  vectorWeights = [],
  densityHeatmap = [],
  statistics,
  modelInfo
}: BPEVisualizationsProps) {
  // 调试日志
  React.useEffect(() => {
    console.log('[BPEVisualizations] 接收到的数据:', {
      processingSteps: processingSteps?.length || 0,
      vectorWeights: vectorWeights?.length || 0,
      densityHeatmap: densityHeatmap?.length || 0,
      statistics,
      modelInfo
    });
  }, [processingSteps, vectorWeights, densityHeatmap, statistics, modelInfo]);
  // 逻辑瀑布流图表配置
  const getWaterfallChartOption = () => {
    if (!processingSteps || processingSteps.length === 0) return null;

    const stages = processingSteps.map((step, index) => ({
      name: step.stage,
      value: step.timestamp - (index > 0 ? processingSteps[index - 1].timestamp : processingSteps[0].timestamp),
      itemStyle: {
        color: {
          preprocessing: '#3b82f6',
          trie_lookup: '#10b981',
          bpe_merge: '#f59e0b',
          subword_split: '#8b5cf6',
          finalization: '#ef4444'
        }[step.stage] || '#6b7280'
      }
    }));

    return {
      title: {
        text: '逻辑瀑布流',
        left: 'center',
        textStyle: { fontSize: 14, fontWeight: 'bold' }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const param = params[0];
          const step = processingSteps[param.dataIndex];
          return `
            <div style="padding: 8px;">
              <div><strong>阶段:</strong> ${step.stage}</div>
              <div><strong>操作:</strong> ${step.action}</div>
              <div><strong>输入:</strong> ${step.input.substring(0, 50)}${step.input.length > 50 ? '...' : ''}</div>
              <div><strong>输出:</strong> ${step.output.substring(0, 50)}${step.output.length > 50 ? '...' : ''}</div>
              <div><strong>决策:</strong> ${step.decision}</div>
            </div>
          `;
        }
      },
      xAxis: {
        type: 'category',
        data: processingSteps.map((s, i) => `步骤 ${i + 1}`),
        axisLabel: { fontSize: 10 }
      },
      yAxis: {
        type: 'value',
        name: '时间 (ms)',
        axisLabel: { fontSize: 10 }
      },
      series: [{
        type: 'bar',
        data: stages.map(s => s.value),
        itemStyle: {
          color: (params: any) => stages[params.dataIndex].itemStyle.color
        },
        label: {
          show: true,
          position: 'top',
          formatter: (params: any) => {
            const step = processingSteps[params.dataIndex];
            return step.stage.replace('_', ' ');
          },
          fontSize: 9
        }
      }]
    };
  };

  // 向量加权可视化图表配置
  const getVectorWeightChartOption = () => {
    if (!vectorWeights || vectorWeights.length === 0) return null;

    const topWeights = vectorWeights
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 20);

    return {
      title: {
        text: '向量加权可视化 (Top 20)',
        left: 'center',
        textStyle: { fontSize: 14, fontWeight: 'bold' }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const param = params[0];
          const weight = topWeights[param.dataIndex];
          return `
            <div style="padding: 8px;">
              <div><strong>Token:</strong> ${weight.token}</div>
              <div><strong>权重:</strong> ${weight.weight.toFixed(4)}</div>
              <div><strong>贡献度:</strong> ${(weight.contribution * 100).toFixed(2)}%</div>
              <div><strong>位置:</strong> ${weight.position}</div>
            </div>
          `;
        }
      },
      xAxis: {
        type: 'category',
        data: topWeights.map(w => w.token.length > 10 ? w.token.substring(0, 10) + '...' : w.token),
        axisLabel: { 
          fontSize: 9,
          rotate: 45,
          interval: 0
        }
      },
      yAxis: {
        type: 'value',
        name: '权重',
        axisLabel: { fontSize: 10 }
      },
      series: [{
        type: 'bar',
        data: topWeights.map(w => w.weight),
        itemStyle: {
          color: (params: any) => {
            const weight = topWeights[params.dataIndex].weight;
            if (weight > 0.8) return '#ef4444';
            if (weight > 0.6) return '#f59e0b';
            if (weight > 0.4) return '#10b981';
            return '#3b82f6';
          }
        },
        label: {
          show: true,
          position: 'top',
          formatter: (params: any) => params.value.toFixed(3),
          fontSize: 8
        }
      }]
    };
  };

  // 词元密度热力图配置
  const getDensityHeatmapOption = () => {
    if (!densityHeatmap || densityHeatmap.length === 0) return null;

    const positions = densityHeatmap.map(d => d.position);
    const densities = densityHeatmap.map(d => d.density);
    const tokens = densityHeatmap.map(d => d.token);

    return {
      title: {
        text: '词元密度热力图',
        left: 'center',
        textStyle: { fontSize: 14, fontWeight: 'bold' }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const param = params[0];
          const point = densityHeatmap[param.dataIndex];
          return `
            <div style="padding: 8px;">
              <div><strong>位置:</strong> ${point.position}</div>
              <div><strong>Token:</strong> ${point.token}</div>
              <div><strong>密度:</strong> ${point.density.toFixed(4)}</div>
              <div><strong>Token 数量:</strong> ${point.tokenCount}</div>
              <div><strong>上下文窗口:</strong> ${point.contextWindow}</div>
            </div>
          `;
        }
      },
      xAxis: {
        type: 'category',
        data: positions.map((p, i) => tokens[i].length > 8 ? tokens[i].substring(0, 8) + '...' : tokens[i]),
        axisLabel: { 
          fontSize: 8,
          rotate: 45,
          interval: Math.max(1, Math.floor(positions.length / 20))
        }
      },
      yAxis: {
        type: 'value',
        name: '密度',
        axisLabel: { fontSize: 10 }
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        orient: 'vertical',
        left: 'right',
        top: 'center',
        inRange: {
          color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        },
        text: ['高', '低'],
        textStyle: { fontSize: 10 }
      },
      series: [{
        type: 'bar',
        data: densities.map((d, i) => ({
          value: d,
          itemStyle: {
            color: d > 0.7 ? '#ef4444' : d > 0.5 ? '#f59e0b' : d > 0.3 ? '#10b981' : '#3b82f6'
          }
        })),
        label: {
          show: true,
          position: 'top',
          formatter: (params: any) => params.value.toFixed(2),
          fontSize: 8
        }
      }]
    };
  };

  const waterfallOption = getWaterfallChartOption();
  const vectorWeightOption = getVectorWeightChartOption();
  const densityOption = getDensityHeatmapOption();

  return (
    <div className="space-y-4">
      {/* 模型信息 */}
      {modelInfo && (
        <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
          <h5 className="text-xs font-medium text-blue-800 mb-2">模型信息</h5>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-blue-600">模型:</span>
              <div className="font-mono text-blue-800">{modelInfo.name}</div>
            </div>
            <div>
              <span className="text-blue-600">词汇表大小:</span>
              <div className="font-mono text-blue-800">{modelInfo.vocabSize.toLocaleString()}</div>
            </div>
            <div>
              <span className="text-blue-600">BPE 合并数:</span>
              <div className="font-mono text-blue-800">{modelInfo.mergesCount.toLocaleString()}</div>
            </div>
          </div>
        </div>
      )}

      {/* 统计信息 */}
      {statistics && (
        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
          <h5 className="text-xs font-medium text-gray-800 mb-2">统计信息</h5>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
            <div>
              <span className="text-gray-600">总词元数:</span>
              <div className="font-medium text-gray-800">{statistics.totalTokens}</div>
            </div>
            <div>
              <span className="text-gray-600">唯一词元:</span>
              <div className="font-medium text-gray-800">{statistics.uniqueTokens}</div>
            </div>
            <div>
              <span className="text-gray-600">子词比例:</span>
              <div className="font-medium text-gray-800">{(statistics.subwordRatio * 100).toFixed(1)}%</div>
            </div>
            <div>
              <span className="text-gray-600">平均长度:</span>
              <div className="font-medium text-gray-800">{statistics.averageTokenLength.toFixed(2)}</div>
            </div>
            <div>
              <span className="text-gray-600">处理时间:</span>
              <div className="font-medium text-gray-800">{statistics.processingTime}ms</div>
            </div>
          </div>
        </div>
      )}

      {/* 逻辑瀑布流 */}
      {waterfallOption && (
        <div className="bg-white rounded-lg p-3 border border-blue-200">
          <h5 className="text-xs font-medium text-blue-800 mb-2">1. 逻辑瀑布流</h5>
          <div style={{ width: '100%', height: '250px' }}>
            <ReactECharts option={waterfallOption} style={{ height: '100%', width: '100%' }} />
          </div>
          {processingSteps.length > 0 && (
            <div className="mt-2 space-y-1">
              {processingSteps.slice(0, 5).map((step, i) => (
                <div key={i} className="text-xs bg-gray-50 rounded p-2">
                  <div className="font-medium text-gray-700">
                    {i + 1}. {step.stage.replace('_', ' ').toUpperCase()}
                  </div>
                  <div className="text-gray-600 mt-1">{step.action}</div>
                  <div className="text-gray-500 text-xs mt-1">{step.decision}</div>
                </div>
              ))}
              {processingSteps.length > 5 && (
                <div className="text-xs text-gray-500 text-center">
                  ... 还有 {processingSteps.length - 5} 个步骤
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 向量加权可视化 */}
      {vectorWeightOption && (
        <div className="bg-white rounded-lg p-3 border border-green-200">
          <h5 className="text-xs font-medium text-green-800 mb-2">2. 向量加权可视化</h5>
          <div style={{ width: '100%', height: '300px' }}>
            <ReactECharts option={vectorWeightOption} style={{ height: '100%', width: '100%' }} />
          </div>
          {vectorWeights.length > 0 && (
            <div className="mt-2">
              <div className="text-xs text-gray-600 mb-1">Top 5 权重词元:</div>
              <div className="space-y-1">
                {vectorWeights
                  .sort((a, b) => b.weight - a.weight)
                  .slice(0, 5)
                  .map((w, i) => (
                    <div key={i} className="flex justify-between items-center text-xs bg-gray-50 rounded px-2 py-1">
                      <span className="font-mono">{w.token}</span>
                      <span className="text-green-600 font-medium">{w.weight.toFixed(4)}</span>
                      <span className="text-gray-500">({(w.contribution * 100).toFixed(2)}%)</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 词元密度热力图 */}
      {densityOption && (
        <div className="bg-white rounded-lg p-3 border border-purple-200">
          <h5 className="text-xs font-medium text-purple-800 mb-2">3. 词元密度热力图</h5>
          <div style={{ width: '100%', height: '300px' }}>
            <ReactECharts option={densityOption} style={{ height: '100%', width: '100%' }} />
          </div>
          {densityHeatmap.length > 0 && (
            <div className="mt-2">
              <div className="text-xs text-gray-600 mb-1">密度统计:</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span className="text-gray-500">最高密度:</span>
                  <div className="font-medium text-purple-600">
                    {Math.max(...densityHeatmap.map(d => d.density)).toFixed(4)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">平均密度:</span>
                  <div className="font-medium text-purple-600">
                    {(densityHeatmap.reduce((sum, d) => sum + d.density, 0) / densityHeatmap.length).toFixed(4)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">最低密度:</span>
                  <div className="font-medium text-purple-600">
                    {Math.min(...densityHeatmap.map(d => d.density)).toFixed(4)}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 空状态提示 */}
      {!waterfallOption && !vectorWeightOption && !densityOption && (
        <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200 text-center">
          <div className="text-sm text-yellow-800 mb-2">
            暂无 BPE 可视化数据。请先进行查询以生成数据。
          </div>
          <div className="text-xs text-yellow-600">
            <div>Processing Steps: {processingSteps?.length || 0}</div>
            <div>Vector Weights: {vectorWeights?.length || 0}</div>
            <div>Density Heatmap: {densityHeatmap?.length || 0}</div>
            <div>Statistics: {statistics ? '存在' : '不存在'}</div>
            <div>Model Info: {modelInfo ? '存在' : '不存在'}</div>
          </div>
        </div>
      )}
    </div>
  );
}
