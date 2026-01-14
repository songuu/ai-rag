'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { io, Socket } from 'socket.io-client';
import { dbManager, type ConversationMessage } from '@/lib/indexeddb';
import ChatMessage from '@/components/ChatMessage';
import QueryAnalysis from '@/components/QueryAnalysis';
import QuestionSelector from '@/components/QuestionSelector';
import ParameterControls from '@/components/ParameterControls';
import FileUpload from '@/components/FileUpload';
import FileList from '@/components/FileList';
import RealtimeMonitoring from '@/components/RealtimeMonitoring';
import RetrievalDetailsPanel from '@/components/RetrievalDetailsPanel';
import SystemInfo from '@/components/SystemInfo';
import Toast from '@/components/Toast';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  traceId?: string;
  retrievalDetails?: any;
  queryAnalysis?: any;
}

interface FileInfo {
  name: string;
  size: number;
  modified: string;
}

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [topK, setTopK] = useState(3);
  const [threshold, setThreshold] = useState(0.0);
  const [llmModel, setLlmModel] = useState('llama3.1');
  const [embeddingModel, setEmbeddingModel] = useState('nomic-embed-text');
  const [queryAnalysis, setQueryAnalysis] = useState<any>(null);
  const [showParams, setShowParams] = useState(true);
  const [showQueryAnalysis, setShowQueryAnalysis] = useState(false);
  const [docCount, setDocCount] = useState(0);
  const [embeddingDim, setEmbeddingDim] = useState(0);
  const [systemStatus, setSystemStatus] = useState('检查中...');
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [vectorizationProgress, setVectorizationProgress] = useState(0);
  const [vectorizationStatus, setVectorizationStatus] = useState('');
  const [showVectorization, setShowVectorization] = useState(false);
  const [queryProcessingStatus, setQueryProcessingStatus] = useState('');
  const [showQueryProcessing, setShowQueryProcessing] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [radarChartData, setRadarChartData] = useState<any>(null);
  const [currentQuery, setCurrentQuery] = useState('');
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [viewingAnalysisFor, setViewingAnalysisFor] = useState<string | null>(null);
  const [retrievalDetails, setRetrievalDetails] = useState<any>(null);
  const [vectorizationDetails, setVectorizationDetails] = useState<any>(null);
  
  const socketRef = useRef<Socket | null>(null);

  // 初始化 WebSocket
  useEffect(() => {
    if (typeof window !== 'undefined') {
      socketRef.current = io();
      
      socketRef.current.on('connect', () => {
        showToast('实时监控连接成功', 'success');
      });
      
      socketRef.current.on('disconnect', () => {
        showToast('实时监控连接断开', 'warning');
      });
      
      socketRef.current.on('vectorization-progress', (progress: any) => {
        setShowVectorization(true);
        setVectorizationDetails(progress);
        if (progress.current && progress.total) {
          setVectorizationProgress((progress.current / progress.total) * 100);
        } else if (progress.progress) {
          setVectorizationProgress(progress.progress);
        }
        setVectorizationStatus(progress.status || progress.message || '处理中...');
      });
      
      socketRef.current.on('query-vectorization-progress', (progress: any) => {
        setShowQueryProcessing(true);
        setQueryProcessingStatus(progress.status || progress.message || '处理中...');
        if (progress.tokenization) {
          setQueryAnalysis((prev: any) => ({
            ...prev,
            tokenization: progress.tokenization
          }));
        }
        if (progress.embedding) {
          setQueryAnalysis((prev: any) => ({
            ...prev,
            embedding: progress.embedding
          }));
        }
      });
      
      socketRef.current.on('retrieval-details', (details: any) => {
        setRetrievalDetails(details);
      });
      
      return () => {
        if (socketRef.current) {
          socketRef.current.disconnect();
        }
      };
    }
  }, []);

  // 检查系统健康状态
  const checkSystemHealth = async () => {
    try {
      const response = await fetch('/api/health');
      const data = await response.json();
      if (data.success) {
        setSystemStatus('运行中');
        setDocCount(data.ragSystem?.documentCount || 0);
        setEmbeddingDim(data.ragSystem?.embeddingDimension || 0);
      } else {
        setSystemStatus('错误');
      }
    } catch (error) {
      setSystemStatus('错误');
    }
  };

  // 加载文件列表
  const loadFilesList = async () => {
    try {
      const response = await fetch('/api/files');
      const data = await response.json();
      if (data.success) {
        setFiles(data.files || []);
      }
    } catch (error) {
      console.error('加载文件列表失败:', error);
    }
  };

  // 删除文件
  const handleDeleteFile = async (filename: string) => {
    if (!confirm(`确定要删除文件 "${filename}" 吗？`)) return;
    
    try {
      const response = await fetch(`/api/files/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      const data = await response.json();
      if (data.success) {
        showToast('文件删除成功', 'success');
        loadFilesList();
        checkSystemHealth();
      } else {
        showToast(data.error || '删除失败', 'error');
      }
    } catch (error) {
      showToast('删除文件时发生错误', 'error');
    }
  };

  // 文件上传
  const handleFileUpload = async () => {
    if (selectedFiles.length === 0) {
      showToast('请先选择文件', 'warning');
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (data.success) {
        showToast(`成功上传 ${data.results.length} 个文件`, 'success');
        setSelectedFiles([]);
        loadFilesList();
        checkSystemHealth();
      } else {
        showToast(data.error || '上传失败', 'error');
      }
    } catch (error) {
      showToast('上传文件时发生错误', 'error');
    } finally {
      setIsUploading(false);
    }
  };

  // 重新初始化
  const handleReinitialize = async () => {
    if (!confirm('确定要重新初始化系统吗？这将重新加载所有文档。')) return;
    
    try {
      const response = await fetch('/api/reinitialize', { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        showToast('系统重新初始化成功', 'success');
        checkSystemHealth();
        loadFilesList();
      } else {
        showToast(data.error || '重新初始化失败', 'error');
      }
    } catch (error) {
      showToast('重新初始化时发生错误', 'error');
    }
  };

  // Toast 通知
  const showToast = (message: string, type: Toast['type'] = 'info') => {
    const id = Date.now().toString();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 5000);
  };

  // 高亮匹配文本
  const highlightMatchingText = (content: string, query: string) => {
    if (!query || query.length < 2) return content.substring(0, 200) + '...';
    const keywords = query.toLowerCase().split(/\s+/).filter(word => word.length > 1);
    let highlighted = content;
    keywords.forEach(keyword => {
      const regex = new RegExp(`(${keyword})`, 'gi');
      highlighted = highlighted.replace(regex, '<mark class="bg-yellow-200">$1</mark>');
    });
    return highlighted.length > 200 ? highlighted.substring(0, 200) + '...' : highlighted;
  };

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 保存消息到 IndexedDB
  const saveMessageToDB = async (message: ConversationMessage) => {
    try {
      console.log('[IndexedDB] 保存消息:', message.id, message.type);
      await dbManager.init();
      
      if (!currentConversationId) {
        console.log('[IndexedDB] 创建新对话');
        const conversation = await dbManager.createNewConversation(
          message.content.substring(0, 50) + (message.content.length > 50 ? '...' : '')
        );
        setCurrentConversationId(conversation.id);
        console.log('[IndexedDB] 新对话 ID:', conversation.id);
      }
      
      if (currentConversationId) {
        // 确保时间戳是 Date 对象
        const messageToSave: ConversationMessage = {
          ...message,
          timestamp: message.timestamp instanceof Date ? message.timestamp : new Date(message.timestamp)
        };
        
        await dbManager.addMessageToConversation(currentConversationId, messageToSave);
        console.log('[IndexedDB] 消息已保存到对话:', currentConversationId);
      }
    } catch (error) {
      console.error('[IndexedDB] 保存消息到数据库失败:', error);
      showToast('保存消息失败: ' + (error instanceof Error ? error.message : String(error)), 'error');
    }
  };

  // 从 IndexedDB 加载最新对话
  const loadLatestConversation = async () => {
    try {
      console.log('[IndexedDB] 开始加载最新对话...');
      await dbManager.init();
      
      // 先尝试获取所有对话，看看数据库里有什么
      const allConversations = await dbManager.getAllConversations();
      console.log(`[IndexedDB] 数据库中共有 ${allConversations.length} 个对话`);
      
      if (allConversations.length > 0) {
        allConversations.forEach((conv, index) => {
          console.log(`[IndexedDB] 对话 ${index + 1}: ID=${conv.id}, 消息数=${conv.messages?.length || 0}, 更新时间=${conv.updatedAt}`);
        });
      }
      
      const latestConv = await dbManager.getLatestConversation();
      
      if (latestConv) {
        console.log(`[IndexedDB] 找到最新对话: ${latestConv.id}`);
        console.log(`[IndexedDB] 对话消息数: ${latestConv.messages?.length || 0}`);
        
        if (latestConv.messages && latestConv.messages.length > 0) {
          setCurrentConversationId(latestConv.id);
          
          // 确保时间戳正确转换
          const restoredMessages: Message[] = latestConv.messages.map((msg, index) => {
            const timestamp = msg.timestamp instanceof Date 
              ? msg.timestamp 
              : new Date(msg.timestamp);
            
            console.log(`[IndexedDB] 消息 ${index + 1}: ${msg.type}, ID=${msg.id}, 内容长度=${msg.content?.length || 0}`);
            
            return {
              id: msg.id,
              type: msg.type,
              content: msg.content,
              timestamp,
              traceId: msg.traceId,
              retrievalDetails: msg.retrievalDetails || null,
              queryAnalysis: msg.queryAnalysis || null
            };
          });
          
          setMessages(restoredMessages);
          console.log(`[IndexedDB] 已恢复 ${restoredMessages.length} 条消息到界面`);
          
          // 恢复最后一条助手消息的检索详情
          const lastAssistantMessage = restoredMessages
            .filter(m => m.type === 'assistant' && m.retrievalDetails)
            .pop();
          if (lastAssistantMessage?.retrievalDetails) {
            setRetrievalDetails(lastAssistantMessage.retrievalDetails);
            console.log('[IndexedDB] 已恢复检索详情');
          }
          
          // 恢复最后一条用户消息的查询分析
          const lastUserMessage = restoredMessages
            .filter(m => m.type === 'user' && m.queryAnalysis)
            .pop();
          if (lastUserMessage?.queryAnalysis) {
            setQueryAnalysis(lastUserMessage.queryAnalysis);
            setShowQueryAnalysis(true);
            if (lastUserMessage.queryAnalysis.embedding?.semanticAnalysis?.vectorFeatures) {
              setRadarChartData(lastUserMessage.queryAnalysis.embedding.semanticAnalysis.vectorFeatures);
            }
            console.log('[IndexedDB] 已恢复查询分析数据');
          }
          
          showToast(`已恢复 ${restoredMessages.length} 条历史消息`, 'success');
        } else {
          console.warn('[IndexedDB] 对话存在但没有消息');
          setCurrentConversationId(latestConv.id);
          setMessages([]);
        }
      } else {
        console.log('[IndexedDB] 没有找到历史对话');
        setMessages([]);
        setCurrentConversationId(null);
        setQueryAnalysis(null);
        setRadarChartData(null);
        setRetrievalDetails(null);
        setShowQueryAnalysis(false);
      }
    } catch (error) {
      console.error('[IndexedDB] 加载历史对话失败:', error);
      console.error('[IndexedDB] 错误详情:', error instanceof Error ? error.stack : String(error));
      showToast('加载历史对话失败: ' + (error instanceof Error ? error.message : String(error)), 'error');
      // 即使失败，也清空状态
      setMessages([]);
      setCurrentConversationId(null);
    }
  };

  // 一键删除所有对话
  const handleDeleteAllConversations = async () => {
    if (!confirm('确定要删除所有对话记录吗？此操作不可恢复！')) return;
    
    try {
      await dbManager.init();
      await dbManager.deleteAllConversations();
      setMessages([]);
      setCurrentConversationId(null);
      setViewingAnalysisFor(null);
      setQueryAnalysis(null);
      setRadarChartData(null);
      setRetrievalDetails(null);
      showToast('所有对话已删除', 'success');
    } catch (error) {
      console.error('删除所有对话失败:', error);
      showToast('删除失败', 'error');
    }
  };

  // 生成模拟 Token
  const generateMockTokens = (text: string) => {
    const tokens: any[] = [];
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      tokens.push({
        token: char,
        tokenId: Math.floor(Math.random() * 5000) + 1000,
        type: /[\u4e00-\u9fff]/.test(char) ? 'chinese' : 
              /[a-zA-Z]/.test(char) ? 'english' :
              /[0-9]/.test(char) ? 'number' : 
              /[.,!?:;()]/.test(char) ? 'punctuation' : 'special'
      });
    }
    return tokens;
  };

  // 提交问题
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessageId = Date.now().toString();
    const userMessage: Message = {
      id: userMessageId,
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentQuery(input.trim());
    setIsLoading(true);
    setShowQueryAnalysis(false);
    setShowQueryProcessing(true);
    
    await saveMessageToDB({
      id: userMessageId,
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    });

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input.trim(),
          topK,
          similarityThreshold: threshold,
          llmModel,
          embeddingModel,
          userId: 'demo-user',
          sessionId: 'demo-session'
        }),
      });

      const data = await response.json();

      console.log('data', data);

      if (data.success) {
        let queryAnalysisData: any;
        if (data.queryAnalysis) {
          queryAnalysisData = data.queryAnalysis;
          if (data.queryAnalysis.embedding?.semanticAnalysis?.vectorFeatures) {
            setRadarChartData(data.queryAnalysis.embedding.semanticAnalysis.vectorFeatures);
          }
        } else {
          queryAnalysisData = {
            tokenization: {
              tokenCount: Math.floor(input.trim().length / 2),
              tokens: generateMockTokens(input.trim()),
              processingTime: 15,
              originalText: input.trim()
            },
            embedding: {
              embeddingDimension: 768,
              semanticAnalysis: {
                context: input.includes('智能') ? '人工智能语境' : '通用语境',
                semanticCategory: input.includes('智能') ? 'AI技术' : '一般',
                confidence: 0.85,
                nearestConcepts: input.includes('智能') 
                  ? ['人工智能', '机器学习', '深度学习'] 
                  : ['文本', '信息', '内容'],
                vectorFeatures: {
                  techScore: 0.7,
                  businessScore: 0.3,
                  dailyScore: 0.2,
                  emotionScore: 0.1,
                  vectorMagnitude: 1.2
                }
              }
            }
          };
          setRadarChartData({
            techScore: 0.7,
            businessScore: 0.3,
            dailyScore: 0.2,
            emotionScore: 0.1,
            vectorMagnitude: 1.2
          });
        }
        
        setMessages(prev => prev.map(msg => 
          msg.id === userMessageId 
            ? { ...msg, queryAnalysis: queryAnalysisData }
            : msg
        ));
        
        if (currentConversationId) {
          try {
            await dbManager.init();
            const conversation = await dbManager.getConversation(currentConversationId);
            if (conversation) {
              const userMsgIndex = conversation.messages.findIndex(m => m.id === userMessageId);
              if (userMsgIndex !== -1) {
                conversation.messages[userMsgIndex].queryAnalysis = queryAnalysisData;
                conversation.updatedAt = new Date();
                await dbManager.saveConversation(conversation);
              }
            }
          } catch (error) {
            console.error('更新用户消息分析数据失败:', error);
          }
        }
        
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: data.answer,
          timestamp: new Date(),
          traceId: data.traceId,
          retrievalDetails: data.retrievalDetails
        };

        setMessages(prev => [...prev, assistantMessage]);
        setShowQueryAnalysis(true);
        setQueryAnalysis(queryAnalysisData);
        
        if (data.retrievalDetails) {
          setRetrievalDetails(data.retrievalDetails);
        }
        
        await saveMessageToDB({
          id: assistantMessage.id,
          type: 'assistant',
          content: data.answer,
          timestamp: new Date(),
          traceId: data.traceId,
          retrievalDetails: data.retrievalDetails
        });
      } else {
        throw new Error(data.error || '请求失败');
      }
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `抱歉，处理您的问题时出现了错误：${error instanceof Error ? error.message : '未知错误'}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setInput('');
      setShowQueryProcessing(false);
    }
  };

  // 初始化时加载数据
  useEffect(() => {
    checkSystemHealth();
    loadFilesList();
    loadLatestConversation();
  }, []);

  // 雷达图配置
  const getRadarChartOption = () => {
    if (!radarChartData) return null;
    
    return {
      title: {
        text: '向量特征分析',
        left: 'center',
        textStyle: { fontSize: 12, color: '#374151' }
      },
      tooltip: { trigger: 'item' },
      radar: {
        indicator: [
          { name: '技术特征', max: 1 },
          { name: '商业特征', max: 1 },
          { name: '日常特征', max: 1 },
          { name: '情感倾向', max: 1, min: -1 },
          { name: '向量强度', max: Math.max(1, radarChartData.vectorMagnitude || 1) }
        ],
        radius: '60%',
        axisName: { fontSize: 10, color: '#6B7280' },
        splitLine: { lineStyle: { color: '#E5E7EB' } },
        axisLine: { lineStyle: { color: '#D1D5DB' } }
      },
      series: [{
        name: '向量特征',
        type: 'radar',
        data: [{
          value: [
            radarChartData.techScore || 0,
            radarChartData.businessScore || 0,
            radarChartData.dailyScore || 0,
            radarChartData.emotionScore || 0,
            (radarChartData.vectorMagnitude || 0) / Math.max(1, radarChartData.vectorMagnitude || 1)
          ],
          name: '当前查询',
          itemStyle: { color: '#3B82F6' },
          areaStyle: { color: 'rgba(59, 130, 246, 0.2)' }
        }]
      }]
    };
  };

  // 获取当前查看的分析数据
  const getCurrentAnalysis = () => {
    if (viewingAnalysisFor) {
      const message = messages.find(m => m.id === viewingAnalysisFor);
      return message?.queryAnalysis;
    }
    return queryAnalysis;
  };

  return (
    <div className="bg-gray-50 min-h-screen">
      {/* 导航栏 */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <i className="fas fa-brain text-blue-600 text-2xl mr-3"></i>
              <h1 className="text-xl font-semibold text-gray-900">本地 RAG 知识库系统</h1>
              <div className="ml-6 flex space-x-2">
                <Link 
                  href="/observability" 
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-blue-600 bg-blue-100 hover:bg-blue-200 transition-colors"
                >
                  <i className="fas fa-chart-line mr-2"></i>
                  可观测性仪表盘
                </Link>
                <Link 
                  href="/history" 
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-gray-600 bg-gray-100 hover:bg-gray-200 transition-colors"
                >
                  <i className="fas fa-history mr-2"></i>
                  历史对话
                </Link>
                <Link 
                  href="/trace-trie" 
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-purple-600 bg-purple-100 hover:bg-purple-200 transition-colors"
                >
                  <i className="fas fa-sitemap mr-2"></i>
                  Trace-Trie 分析
                </Link>
                <button
                  onClick={handleDeleteAllConversations}
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-red-600 bg-red-100 hover:bg-red-200 transition-colors"
                  title="删除所有对话"
                >
                  <i className="fas fa-trash-alt mr-2"></i>
                  清空对话
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full mr-2 ${systemStatus === '运行中' ? 'bg-green-400' : 'bg-gray-400'}`}></div>
                <span className="text-sm text-gray-600">{systemStatus}</span>
              </div>
              <button 
                onClick={checkSystemHealth}
                className="p-2 text-gray-400 hover:text-gray-600"
                title="刷新系统状态"
              >
                <i className="fas fa-sync-alt"></i>
              </button>
              <button 
                onClick={loadLatestConversation}
                className="p-2 text-gray-400 hover:text-gray-600"
                title="重新加载对话"
              >
                <i className="fas fa-redo"></i>
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* 主聊天区域 */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm border">
              {/* 聊天头部 */}
              <div className="border-b px-6 py-4">
                <h2 className="text-lg font-medium text-gray-900">智能问答</h2>
                <p className="text-sm text-gray-500 mt-1">向知识库提问，获得基于文档的准确回答</p>
              </div>
              
              {/* 聊天消息区域 */}
              <div className="h-96 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center text-gray-500 text-sm">
                    <i className="fas fa-comments text-2xl mb-2"></i>
                    <p>开始提问吧！我会根据已上传的文档来回答您的问题。</p>
                  </div>
                ) : (
                  messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={message}
                      currentQuery={currentQuery}
                      highlightMatchingText={highlightMatchingText}
                    />
                  ))
                )}
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 rounded-lg px-4 py-2">
                      <div className="flex items-center space-x-2">
                        <div className="typing-indicator"></div>
                        <span className="text-sm text-gray-600">AI 正在思考...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* 输入区域 */}
              <div className="border-t p-6">
                <ParameterControls
                  topK={topK}
                  threshold={threshold}
                  llmModel={llmModel}
                  embeddingModel={embeddingModel}
                  onTopKChange={setTopK}
                  onThresholdChange={setThreshold}
                  onLLMModelChange={setLlmModel}
                  onEmbeddingModelChange={setEmbeddingModel}
                  showParams={showParams}
                  onToggle={() => setShowParams(!showParams)}
                />
                
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="flex space-x-4">
                    <div className="flex-1">
                      <input 
                        type="text" 
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="请输入您的问题..."
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        disabled={isLoading}
                        required
                      />
                    </div>
                    <button 
                      type="submit"
                      disabled={isLoading || !input.trim()}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <i className="fas fa-paper-plane mr-2"></i>
                      发送
                    </button>
                  </div>
                  
                  {/* 用户问题处理结果展示 */}
                  <div className="space-y-4">
                    <QuestionSelector
                      messages={messages}
                      viewingAnalysisFor={viewingAnalysisFor}
                      onSelect={setViewingAnalysisFor}
                    />
                    
                    {/* 显示选中的问题分析 */}
                    {viewingAnalysisFor && messages.find(m => m.id === viewingAnalysisFor)?.queryAnalysis && (
                      <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                        <h4 className="text-sm font-medium text-blue-800 mb-3">
                          <i className="fas fa-cogs mr-2"></i>
                          用户问题处理分析
                          <span className="text-xs font-normal text-blue-600 ml-2">
                            ({messages.find(m => m.id === viewingAnalysisFor)?.content.substring(0, 50)}...)
                          </span>
                        </h4>
                        <QueryAnalysis
                          analysis={messages.find(m => m.id === viewingAnalysisFor)!.queryAnalysis}
                          radarChartData={radarChartData}
                          topK={topK}
                          threshold={threshold}
                          getRadarChartOption={getRadarChartOption}
                        />
                      </div>
                    )}
                    
                    {/* 显示当前查询的分析（如果没有选中历史问题） */}
                    {!viewingAnalysisFor && showQueryAnalysis && queryAnalysis && (
                      <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                        <h4 className="text-sm font-medium text-blue-800 mb-3">
                          <i className="fas fa-cogs mr-2"></i>
                          用户问题处理分析
                        </h4>
                        <QueryAnalysis
                          analysis={queryAnalysis}
                          radarChartData={radarChartData}
                          topK={topK}
                          threshold={threshold}
                          getRadarChartOption={getRadarChartOption}
                        />
                      </div>
                    )}
                  </div>
                </form>
              </div>
            </div>
          </div>

          {/* 侧边栏 */}
          <div className="space-y-6">
            <FileUpload
              selectedFiles={selectedFiles}
              isUploading={isUploading}
              onFileSelect={setSelectedFiles}
              onUpload={handleFileUpload}
            />
            
            <FileList
              files={files}
              onRefresh={loadFilesList}
              onDelete={handleDeleteFile}
              formatFileSize={formatFileSize}
            />
            
            <RealtimeMonitoring
              showVectorization={showVectorization}
              vectorizationDetails={vectorizationDetails}
              vectorizationProgress={vectorizationProgress}
              vectorizationStatus={vectorizationStatus}
              showQueryProcessing={showQueryProcessing}
              queryProcessingStatus={queryProcessingStatus}
              isLoading={isLoading}
              queryAnalysis={queryAnalysis}
              retrievalDetails={retrievalDetails}
            />
            
            {/* 检索详情面板 */}
            <RetrievalDetailsPanel
              retrievalDetails={retrievalDetails}
              queryText={currentQuery}
            />
            
            <SystemInfo
              docCount={docCount}
              embeddingDim={embeddingDim}
              systemStatus={systemStatus}
              onReinitialize={handleReinitialize}
            />
          </div>
        </div>
      </div>

      <Toast toasts={toasts} />
    </div>
  );
}