'use client';

import React from 'react';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  traceId?: string;
  retrievalDetails?: any;
  queryAnalysis?: any;
}

interface ChatMessageProps {
  message: Message;
  currentQuery: string;
  highlightMatchingText: (content: string, query: string) => string;
}

export default function ChatMessage({ message, currentQuery, highlightMatchingText }: ChatMessageProps) {
  return (
    <div
      className={`flex chat-message ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 ${
          message.type === 'user'
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 text-gray-900'
        }`}
      >
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        <div className="flex items-center justify-between mt-2 text-xs opacity-70">
          <span>{message.timestamp.toLocaleTimeString()}</span>
          {message.traceId && (
            <span className="ml-2 text-xs opacity-60">
              Trace: {message.traceId.slice(0, 8)}
            </span>
          )}
        </div>
        
        {/* 检索详情 */}
        {message.retrievalDetails && message.type === 'assistant' && (
          <details className="mt-2 text-xs">
            <summary className="cursor-pointer hover:opacity-80">
              查看检索详情 ({message.retrievalDetails.searchResults?.length || 0} 个匹配文档)
            </summary>
            <div className="mt-2 space-y-2 pt-2 border-t border-gray-700">
              <div className="text-xs opacity-80">
                <p>检索耗时: {message.retrievalDetails.searchTime}ms</p>
                <p>总文档数: {message.retrievalDetails.totalDocuments}</p>
                <p>相似度阈值: {message.retrievalDetails.threshold}</p>
              </div>
              {message.retrievalDetails.searchResults?.map((result: any, index: number) => (
                <div key={index} className="bg-gray-800 rounded p-2 mt-2">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-medium">文档 {index + 1}</span>
                    <span className="text-blue-400">相似度: {(result.similarity * 100).toFixed(2)}%</span>
                  </div>
                  <p 
                    className="text-xs opacity-90 line-clamp-2"
                    dangerouslySetInnerHTML={{ 
                      __html: highlightMatchingText(
                        result.document?.content || '', 
                        currentQuery
                      ) 
                    }}
                  ></p>
                  <p className="text-xs opacity-60 mt-1">
                    来源: {result.document?.metadata?.source || 'Unknown'}
                  </p>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}