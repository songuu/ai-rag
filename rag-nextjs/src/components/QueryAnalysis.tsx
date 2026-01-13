'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import BPEVisualizations from './BPEVisualizations';

const ReactECharts = dynamic(() => import('echarts-for-react'), { ssr: false });

interface TokenInfo {
  token: string;
  tokenId: number;
  type: 'chinese' | 'english' | 'number' | 'punctuation' | 'special';
}

interface QueryAnalysisProps {
  analysis: any;
  radarChartData?: any;
  topK: number;
  threshold: number;
  getRadarChartOption: () => any;
}

export default function QueryAnalysis({ 
  analysis, 
  radarChartData, 
  topK, 
  threshold, 
  getRadarChartOption 
}: QueryAnalysisProps) {
  if (!analysis) return null;
  
  return (
    <>
      {/* 词元化结果 */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h5 className="text-xs font-medium text-blue-700">A. 词元化 (Tokenization)</h5>
          <span className="text-xs text-blue-600">{analysis.tokenization?.processingTime || 0}ms</span>
        </div>
        <div className="bg-white rounded p-3 border border-blue-100">
          <div className="text-xs text-gray-600 mb-2">原始文本 → Token 序列</div>
          <div className="mb-2">
            <span className="text-xs text-gray-500">原始文本:</span>
            <div className="bg-gray-50 rounded px-2 py-1 text-sm font-mono mt-1">
              "{analysis.tokenization?.originalText || ''}"
            </div>
          </div>
          <div className="mb-2">
            <span className="text-xs text-gray-500">Token 分解 ({analysis.tokenization?.tokenCount || 0} 个词元):</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {analysis.tokenization?.tokens?.slice(0, 20).map((token: TokenInfo, i: number) => {
                const colors: Record<string, string> = {
                  chinese: 'bg-red-100 text-red-700 border-red-200',
                  english: 'bg-blue-100 text-blue-700 border-blue-200',
                  number: 'bg-green-100 text-green-700 border-green-200',
                  punctuation: 'bg-yellow-100 text-yellow-700 border-yellow-200',
                  special: 'bg-gray-100 text-gray-700 border-gray-200',
                  subword: 'bg-purple-100 text-purple-700 border-purple-200'
                };
                return (
                  <span
                    key={i}
                    className={`inline-block px-2 py-1 rounded text-xs border ${colors[token.type] || colors.special}`}
                    title={`Token ID: ${token.tokenId}, Type: ${token.type}`}
                  >
                    {token.token}
                    <sub className="text-xs opacity-60 ml-1">{token.tokenId}</sub>
                  </span>
                );
              })}
              {analysis.tokenization?.tokens && analysis.tokenization.tokens.length > 20 && (
                <span className="text-xs text-gray-500">... 还有 {analysis.tokenization.tokens.length - 20} 个 tokens</span>
              )}
            </div>
          </div>
          
          {/* BPE 可视化 */}
          {/* 始终显示 BPE 可视化组件，让它自己判断是否有数据 */}
          <div className="mt-4 pt-4 border-t border-blue-200">
            <BPEVisualizations
              processingSteps={analysis.tokenization?.processingSteps}
              vectorWeights={analysis.tokenization?.vectorWeights}
              densityHeatmap={analysis.tokenization?.densityHeatmap}
              statistics={analysis.tokenization?.statistics}
              modelInfo={analysis.tokenization?.modelInfo}
            />
          </div>
        </div>
      </div>
      
      {/* 向量化结果 */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h5 className="text-xs font-medium text-blue-700">B. 向量化 (Embedding)</h5>
          <span className="text-xs text-blue-600">完成</span>
        </div>
        <div className="bg-white rounded p-3 border border-blue-100">
          <div className="space-y-2 text-xs">
            <div className="mb-2">
              <span className="text-xs text-gray-500">语义分析:</span>
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded p-2 mt-1">
                <div className="text-sm font-medium text-purple-800">
                  {analysis.embedding?.semanticAnalysis?.context || '通用语境'}
                </div>
                <div className="text-xs text-purple-600 mt-1">
                  分类: {analysis.embedding?.semanticAnalysis?.semanticCategory || '一般'} 
                  (置信度: {((analysis.embedding?.semanticAnalysis?.confidence || 0) * 100).toFixed(1)}%)
                </div>
                <div className="text-xs text-purple-600 mt-1">
                  相关概念: {(analysis.embedding?.semanticAnalysis?.nearestConcepts || []).join(', ')}
                </div>
              </div>
            </div>
            {analysis.embedding?.semanticAnalysis?.vectorFeatures && (
              <div className="mb-2">
                <span className="text-xs text-gray-500">向量特征分析:</span>
                <div className="bg-gray-50 rounded p-2 mt-1 space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>技术特征:</span>
                    <span className="font-mono">{(analysis.embedding.semanticAnalysis.vectorFeatures.techScore || 0).toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>商业特征:</span>
                    <span className="font-mono">{(analysis.embedding.semanticAnalysis.vectorFeatures.businessScore || 0).toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>日常特征:</span>
                    <span className="font-mono">{(analysis.embedding.semanticAnalysis.vectorFeatures.dailyScore || 0).toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>情感倾向:</span>
                    <span className="font-mono">{(analysis.embedding.semanticAnalysis.vectorFeatures.emotionScore || 0).toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>向量模长:</span>
                    <span className="font-mono">{(analysis.embedding.semanticAnalysis.vectorFeatures.vectorMagnitude || 0).toFixed(3)}</span>
                  </div>
                </div>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-600">向量维度:</span>
              <span className="font-medium">{analysis.embedding?.embeddingDimension || 768}</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* 检索链路分析 */}
      {radarChartData && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <h5 className="text-xs font-medium text-blue-700">C. 检索链路分析</h5>
          </div>
          <div className="bg-white rounded p-3 border border-blue-100">
            {getRadarChartOption() && (
              <div className="mb-3" style={{ width: '100%', height: '200px' }}>
                <ReactECharts option={getRadarChartOption()} style={{ height: '100%', width: '100%' }} />
              </div>
            )}
            <div className="text-xs space-y-2">
              <div className="bg-blue-50 rounded p-2">
                <div className="text-xs font-medium text-blue-800 mb-1">1. 查询理解</div>
                <div className="text-xs text-blue-600">
                  • 词元数量: {analysis.tokenization?.tokenCount || 0}<br/>
                  • 语义分类: {analysis.embedding?.semanticAnalysis?.semanticCategory || '一般'}<br/>
                  • 置信度: {((analysis.embedding?.semanticAnalysis?.confidence || 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-green-50 rounded p-2">
                <div className="text-xs font-medium text-green-800 mb-1">2. 向量编码</div>
                <div className="text-xs text-green-600">
                  • 向量维度: {analysis.embedding?.embeddingDimension || 768}<br/>
                  • 向量模长: {(radarChartData.vectorMagnitude || 0).toFixed(3)}<br/>
                  • 主要特征: {radarChartData.techScore > radarChartData.businessScore ? '技术导向' : '商业导向'}
                </div>
              </div>
              <div className="bg-purple-50 rounded p-2">
                <div className="text-xs font-medium text-purple-800 mb-1">3. 相似度计算</div>
                <div className="text-xs text-purple-600">
                  • 算法: 余弦相似度<br/>
                  • 搜索空间: {analysis.embedding?.embeddingDimension || 768} 维向量空间<br/>
                  • 匹配策略: Top-K + 阈值过滤
                </div>
              </div>
              <div className="bg-orange-50 rounded p-2">
                <div className="text-xs font-medium text-orange-800 mb-1">4. 结果排序</div>
                <div className="text-xs text-orange-600">
                  • 排序依据: 相似度分数<br/>
                  • 过滤条件: 阈值 ≥ {threshold.toFixed(2)}<br/>
                  • 返回数量: Top-{topK}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}