'use strict';

/**
 * 上下文管理系统 (Context Management System)
 * 
 * 自定义实现，不依赖 LangGraph 复杂组件：
 * - 文件持久化存储会话
 * - 自定义窗口管理
 * - 自定义查询改写
 */

import { ChatOllama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { getMilvusInstance, getModelDimension, selectModelByDimension } from './milvus-client';
import * as fs from 'fs';
import * as path from 'path';

// ==================== 类型定义 ====================

export type MessageRole = 'user' | 'assistant' | 'system';

export interface ConversationMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  tokenCount?: number;
}

export interface RetrievedDocument {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, any>;
}

export interface WorkflowStep {
  step: string;
  status: 'pending' | 'running' | 'completed' | 'skipped' | 'error';
  startTime?: number;
  endTime?: number;
  duration?: number;
  details?: Record<string, any>;
}

export interface SessionMetadata {
  sessionId: string;
  userId?: string;
  createdAt: number;
  lastActiveAt: number;
  totalTokens: number;
  messageCount: number;
  truncatedCount: number;
  summarizedRounds: number;
}

export interface SessionData {
  metadata: SessionMetadata;
  messages: ConversationMessage[];
  summary?: string;
}

export interface ContextState {
  messages: ConversationMessage[];
  metadata: SessionMetadata;
  summary?: string;
  artifacts: {
    rewrittenQuery?: string;
    retrievedDocuments?: RetrievedDocument[];
  };
  workflowSteps: WorkflowStep[];
}

export type WindowStrategy = 'sliding_window' | 'token_limit' | 'hybrid';

export interface WindowConfig {
  strategy: WindowStrategy;
  maxRounds?: number;
  maxTokens?: number;
  preserveSystemPrompt?: boolean;
}

export interface ContextManagerConfig {
  llmModel: string;
  embeddingModel: string;
  milvusCollection: string;
  windowConfig: WindowConfig;
  enableQueryRewriting: boolean;
  maxRetries: number;
  similarityThreshold: number;
  topK: number;
}

// ==================== 默认配置 ====================

const DEFAULT_CONFIG: ContextManagerConfig = {
  llmModel: 'qwen2.5:0.5b',
  embeddingModel: 'bge-m3:latest',
  milvusCollection: 'rag_documents',
  windowConfig: {
    strategy: 'hybrid',
    maxRounds: 10,
    maxTokens: 4000,
    preserveSystemPrompt: true,
  },
  enableQueryRewriting: true,
  maxRetries: 3,
  similarityThreshold: 0.3,
  topK: 5,
};

// ==================== 工具函数 ====================

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function estimateTokens(text: string): number {
  const chineseChars = (text.match(/[\u4e00-\u9fff]/g) || []).length;
  const otherChars = text.length - chineseChars;
  return Math.ceil(chineseChars / 1.5 + otherChars / 4);
}

function createStep(name: string): WorkflowStep {
  return { step: name, status: 'running', startTime: Date.now() };
}

function completeStep(step: WorkflowStep, details?: Record<string, any>): WorkflowStep {
  return {
    ...step,
    status: 'completed',
    endTime: Date.now(),
    duration: Date.now() - (step.startTime || Date.now()),
    details,
  };
}

// ==================== 文件持久化 ====================

const DATA_DIR = 'data/context-sessions';

function ensureDataDir(): void {
  const fullPath = path.join(process.cwd(), DATA_DIR);
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
  }
}

function getFilePath(sessionId: string): string {
  return path.join(process.cwd(), DATA_DIR, `${sessionId}.json`);
}

function saveSession(data: SessionData): void {
  ensureDataDir();
  fs.writeFileSync(getFilePath(data.metadata.sessionId), JSON.stringify(data, null, 2), 'utf-8');
}

function loadSession(sessionId: string): SessionData | null {
  const filePath = getFilePath(sessionId);
  if (!fs.existsSync(filePath)) return null;
  
  try {
    const raw = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    
    if (raw.metadata?.sessionId) {
      return { metadata: raw.metadata, messages: raw.messages || [], summary: raw.summary };
    } else if (raw.sessionId) {
      return { metadata: raw as SessionMetadata, messages: [] };
    }
    return null;
  } catch {
    return null;
  }
}

function listSessions(): SessionMetadata[] {
  ensureDataDir();
  const fullPath = path.join(process.cwd(), DATA_DIR);
  const sessions: SessionMetadata[] = [];
  
  for (const file of fs.readdirSync(fullPath)) {
    if (!file.endsWith('.json')) continue;
    try {
      const data = loadSession(file.replace('.json', ''));
      if (data?.metadata) {
        sessions.push({ ...data.metadata, messageCount: data.messages.length });
      }
    } catch { /* skip */ }
  }
  
  return sessions.sort((a, b) => (b.lastActiveAt || 0) - (a.lastActiveAt || 0));
}

function deleteSession(sessionId: string): boolean {
  const filePath = getFilePath(sessionId);
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
    return true;
  }
  return false;
}

// ==================== 窗口管理器 ====================

class WindowManager {
  constructor(private config: WindowConfig) {}
  
  trim(messages: ConversationMessage[]): { messages: ConversationMessage[]; trimmedCount: number } {
    let result = [...messages];
    const originalCount = result.length;
    
    // 1. 滑动窗口
    const maxRounds = this.config.maxRounds || 10;
    const maxMessages = maxRounds * 2;
    if (result.length > maxMessages) {
      const systemMsgs = this.config.preserveSystemPrompt 
        ? result.filter(m => m.role === 'system') 
        : [];
      const nonSystemMsgs = result.filter(m => m.role !== 'system').slice(-maxMessages);
      result = [...systemMsgs, ...nonSystemMsgs];
    }
    
    // 2. Token 限制
    if (this.config.strategy === 'token_limit' || this.config.strategy === 'hybrid') {
      const maxTokens = this.config.maxTokens || 4000;
      let totalTokens = 0;
      const kept: ConversationMessage[] = [];
      
      // 从最新的开始保留
      for (let i = result.length - 1; i >= 0; i--) {
        const msg = result[i];
        const tokens = msg.tokenCount || estimateTokens(msg.content);
        if (totalTokens + tokens <= maxTokens || msg.role === 'system') {
          kept.unshift(msg);
          totalTokens += tokens;
        }
      }
      result = kept;
    }
    
    return { messages: result, trimmedCount: originalCount - result.length };
  }
}

// ==================== 查询改写器 ====================

class QueryRewriter {
  constructor(private llm: ChatOllama) {}
  
  async rewrite(query: string, history: ConversationMessage[]): Promise<{
    rewrittenQuery: string;
    needsRewrite: boolean;
    reason: string;
  }> {
    // 无历史，不改写
    if (history.length === 0) {
      return { rewrittenQuery: query, needsRewrite: false, reason: '首轮对话' };
    }
    
    // 检测话题切换
    if (this.isNewTopic(query, history)) {
      return { rewrittenQuery: query, needsRewrite: false, reason: '新话题' };
    }
    
    // 检查是否需要改写
    const hasPronouns = /^(它|这|那|他|她|前面|上面)/.test(query);
    const isShort = query.length < 6;
    
    if (!hasPronouns && !isShort && query.length > 10) {
      return { rewrittenQuery: query, needsRewrite: false, reason: '查询完整' };
    }
    
    // 构建历史
    const recentHistory = history.slice(-6).map(m => 
      `${m.role === 'user' ? '用户' : 'AI'}: ${m.content}`
    ).join('\n');
    
    const prompt = `将用户问题改写为独立完整的问题。只补全代词和省略，不要添加无关内容。
如果问题已完整，原样返回。

对话历史:
${recentHistory}

当前问题: ${query}

改写后的问题（只输出问题本身）:`;

    try {
      const response = await this.llm.invoke(prompt);
      const rewritten = response.content.toString().trim();
      
      // 验证改写结果
      if (rewritten.length > query.length * 3 || !rewritten.includes(query.slice(0, 2))) {
        return { rewrittenQuery: query, needsRewrite: false, reason: '改写无效' };
      }
      
      return {
        rewrittenQuery: rewritten,
        needsRewrite: rewritten !== query,
        reason: '已改写',
      };
    } catch {
      return { rewrittenQuery: query, needsRewrite: false, reason: '改写失败' };
    }
  }
  
  private isNewTopic(query: string, history: ConversationMessage[]): boolean {
    // 检查苹果双关
    if (query.includes('苹果')) {
      const isFruit = /好吃|味道|水果|哪里的|产地/.test(query);
      const historyHasApple = history.some(m => /iPhone|iPad|苹果手机/.test(m.content));
      if (isFruit && historyHasApple) return true;
    }
    return false;
  }
}

// ==================== 上下文管理器 ====================

export class ContextManager {
  private config: ContextManagerConfig;
  private llm: ChatOllama;
  private embeddings: OllamaEmbeddings;
  private windowManager: WindowManager;
  private queryRewriter: QueryRewriter;
  
  constructor(config: Partial<ContextManagerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.llm = new ChatOllama({ model: this.config.llmModel, temperature: 0.7 });
    this.embeddings = new OllamaEmbeddings({ model: this.config.embeddingModel });
    this.windowManager = new WindowManager(this.config.windowConfig);
    this.queryRewriter = new QueryRewriter(this.llm);
  }
  
  getConfig(): ContextManagerConfig {
    return this.config;
  }
  
  updateConfig(newConfig: Partial<ContextManagerConfig>): void {
    this.config = { ...this.config, ...newConfig };
    if (newConfig.llmModel) {
      this.llm = new ChatOllama({ model: this.config.llmModel, temperature: 0.7 });
      this.queryRewriter = new QueryRewriter(this.llm);
    }
    if (newConfig.embeddingModel) {
      this.embeddings = new OllamaEmbeddings({ model: this.config.embeddingModel });
    }
    if (newConfig.windowConfig) {
      this.windowManager = new WindowManager(this.config.windowConfig);
    }
  }
  
  // ==================== 会话管理 ====================
  
  async createSession(userId?: string): Promise<ContextState> {
    const sessionId = generateId();
    const now = Date.now();
    
    const data: SessionData = {
      metadata: {
        sessionId, userId, createdAt: now, lastActiveAt: now,
        totalTokens: 0, messageCount: 0, truncatedCount: 0, summarizedRounds: 0,
      },
      messages: [],
    };
    
    saveSession(data);
    return { messages: [], metadata: data.metadata, artifacts: {}, workflowSteps: [] };
  }
  
  async getSession(sessionId: string): Promise<ContextState | null> {
    const data = loadSession(sessionId);
    if (!data) return null;
    
    return {
      messages: data.messages,
      metadata: {
        ...data.metadata,
        messageCount: data.messages.length,
        totalTokens: data.messages.reduce((sum, m) => sum + (m.tokenCount || 0), 0),
      },
      summary: data.summary,
      artifacts: {},
      workflowSteps: [],
    };
  }
  
  async listSessions(): Promise<SessionMetadata[]> {
    return listSessions();
  }
  
  async deleteSession(sessionId: string): Promise<boolean> {
    return deleteSession(sessionId);
  }
  
  // ==================== 核心查询处理 ====================
  
  async processQuery(
    sessionId: string,
    userQuery: string,
    options: { userId?: string; topK?: number; similarityThreshold?: number } = {}
  ): Promise<{
    response: string;
    state: ContextState;
    rewrittenQuery?: string;
    retrievedDocs: RetrievedDocument[];
    workflowSteps: WorkflowStep[];
  }> {
    const workflowSteps: WorkflowStep[] = [];
    const topK = options.topK || this.config.topK;
    const threshold = options.similarityThreshold || this.config.similarityThreshold;
    
    // 1. 加载/创建会话
    let loadStep = createStep('状态加载');
    let sessionData = loadSession(sessionId);
    if (!sessionData) {
      sessionData = {
        metadata: {
          sessionId, userId: options.userId, createdAt: Date.now(), lastActiveAt: Date.now(),
          totalTokens: 0, messageCount: 0, truncatedCount: 0, summarizedRounds: 0,
        },
        messages: [],
      };
    }
    workflowSteps.push(completeStep(loadStep, { isNew: !sessionData.messages.length }));
    
    // 2. 窗口截断
    let trimStep = createStep('窗口截断');
    const { messages: trimmedMessages, trimmedCount } = this.windowManager.trim(sessionData.messages);
    sessionData.messages = trimmedMessages;
    workflowSteps.push(completeStep(trimStep, { trimmedCount }));
    
    // 3. 查询改写
    let rewriteStep = createStep('查询改写');
    let rewrittenQuery = userQuery;
    let needsRewrite = false;
    let rewriteReason = '未启用';
    
    if (this.config.enableQueryRewriting && sessionData.messages.length > 0) {
      const rewriteResult = await this.queryRewriter.rewrite(userQuery, sessionData.messages);
      rewrittenQuery = rewriteResult.rewrittenQuery;
      needsRewrite = rewriteResult.needsRewrite;
      rewriteReason = rewriteResult.reason;
    }
    workflowSteps.push(completeStep(rewriteStep, {
      original: userQuery,
      rewritten: rewrittenQuery,
      needsRewrite,
      reason: rewriteReason,
    }));
    
    // 4. 判断是否需要检索（简单问候不检索）
    const isGreeting = this.isGreeting(userQuery);
    let retrievedDocs: RetrievedDocument[] = [];
    
    if (!isGreeting) {
      // 5. 向量检索
      let retrieveStep = createStep('向量检索');
      retrievedDocs = await this.retrieve(rewrittenQuery, topK, threshold);
      workflowSteps.push(completeStep(retrieveStep, {
        query: rewrittenQuery,
        resultCount: retrievedDocs.length,
      }));
      
      // 6. 相关性过滤
      let filterStep = createStep('相关性验证');
      const originalCount = retrievedDocs.length;
      retrievedDocs = this.filterRelevant(retrievedDocs, rewrittenQuery);
      workflowSteps.push(completeStep(filterStep, {
        originalCount,
        filteredCount: retrievedDocs.length,
      }));
    } else {
      workflowSteps.push(completeStep(createStep('向量检索'), { skipped: true, reason: '问候语' }));
      workflowSteps.push(completeStep(createStep('相关性验证'), { skipped: true }));
    }
    
    // 7. 生成响应
    let generateStep = createStep('响应生成');
    const response = await this.generate(userQuery, rewrittenQuery, sessionData.messages, retrievedDocs, isGreeting);
    workflowSteps.push(completeStep(generateStep, { responseLength: response.length }));
    
    // 8. 保存消息
    let saveStep = createStep('状态保存');
    const now = Date.now();
    const userMsg: ConversationMessage = {
      id: `${now}-user`, role: 'user', content: userQuery,
      timestamp: now, tokenCount: estimateTokens(userQuery),
    };
    const aiMsg: ConversationMessage = {
      id: `${now}-ai`, role: 'assistant', content: response,
      timestamp: now + 1, tokenCount: estimateTokens(response),
    };
    
    sessionData.messages.push(userMsg, aiMsg);
    sessionData.metadata.lastActiveAt = now;
    sessionData.metadata.messageCount = sessionData.messages.length;
    sessionData.metadata.totalTokens = sessionData.messages.reduce((sum, m) => sum + (m.tokenCount || 0), 0);
    sessionData.metadata.truncatedCount += trimmedCount;
    
    saveSession(sessionData);
    workflowSteps.push(completeStep(saveStep, { messageCount: sessionData.messages.length }));
    
    return {
      response,
      state: {
        messages: sessionData.messages,
        metadata: sessionData.metadata,
        summary: sessionData.summary,
        artifacts: { rewrittenQuery, retrievedDocuments: retrievedDocs },
        workflowSteps,
      },
      rewrittenQuery: needsRewrite ? rewrittenQuery : undefined,
      retrievedDocs,
      workflowSteps,
    };
  }
  
  // ==================== 辅助方法 ====================
  
  private isGreeting(query: string): boolean {
    const greetings = [
      /^(你好|您好|hi|hello|hey|嗨|哈喽)/i,
      /^(早上好|下午好|晚上好|早安|晚安)/,
      /^(你是谁|你叫什么|介绍一下你自己)/,
      /^(谢谢|感谢|多谢|辛苦了)/,
      /^(再见|拜拜|bye)/i,
    ];
    return greetings.some(p => p.test(query.trim()));
  }
  
  private async retrieve(query: string, topK: number, threshold: number): Promise<RetrievedDocument[]> {
    try {
      let dimension = getModelDimension(this.config.embeddingModel);
      let embeddings = this.embeddings;
      
      const milvus = getMilvusInstance({
        collectionName: this.config.milvusCollection,
        embeddingDimension: dimension,
      });
      
      // 自动适配维度
      try {
        await milvus.connect();
        const stats = await milvus.getCollectionStats();
        if (stats?.embeddingDimension && stats.embeddingDimension !== dimension) {
          const model = selectModelByDimension(stats.embeddingDimension);
          embeddings = new OllamaEmbeddings({ model });
        }
      } catch { /* use default */ }
      
      const queryEmbedding = await embeddings.embedQuery(query);
      const results = await milvus.search(queryEmbedding, topK, threshold);
      
      return results.map(r => ({
        id: r.id || generateId(),
        content: r.content,
        score: r.score,
        metadata: r.metadata,
      }));
    } catch (error) {
      console.error('[ContextManager] 检索失败:', error);
      return [];
    }
  }
  
  private filterRelevant(docs: RetrievedDocument[], query: string): RetrievedDocument[] {
    if (docs.length === 0) return [];
    
    const keywords = this.extractKeywords(query);
    
    return docs.filter(doc => {
      if (doc.score < 0.2) return false;
      const content = doc.content.toLowerCase();
      const matches = keywords.filter(kw => content.includes(kw.toLowerCase()));
      return matches.length > 0 || doc.score > 0.5;
    });
  }
  
  private extractKeywords(text: string): string[] {
    const keywords: string[] = [];
    
    const patterns = [
      /(?:华为|苹果|小米|三星)[A-Za-z0-9\u4e00-\u9fff]+/g,
      /(?:iPhone|iPad|MacBook|Mate|Galaxy)[A-Za-z0-9\s]*/gi,
      /版本|价格|配置|参数|续航|屏幕/g,
    ];
    
    for (const p of patterns) {
      const m = text.match(p);
      if (m) keywords.push(...m);
    }
    
    const chinese = text.match(/[\u4e00-\u9fff]{2,6}/g);
    if (chinese) {
      const stops = ['什么', '怎么', '如何', '为什么', '哪个', '那个', '这个'];
      keywords.push(...chinese.filter(w => !stops.includes(w)));
    }
    
    return [...new Set(keywords)];
  }
  
  private async generate(
    originalQuery: string,
    rewrittenQuery: string,
    history: ConversationMessage[],
    docs: RetrievedDocument[],
    isGreeting: boolean
  ): Promise<string> {
    // 构建历史
    const historyText = history.slice(-6).map(m => 
      `${m.role === 'user' ? '用户' : 'AI'}: ${m.content}`
    ).join('\n');
    
    // 构建上下文
    const context = docs.length > 0
      ? docs.map((d, i) => `[${i + 1}] ${d.content}`).join('\n\n')
      : '';
    
    let prompt: string;
    
    if (isGreeting) {
      // 问候语直接回答
      prompt = `你是一个友好的智能助手。请自然地回应用户的问候或问题。

${historyText ? `对话历史:\n${historyText}\n` : ''}
用户: ${originalQuery}

请友好地回应:`;
    } else {
      // 知识问答
      prompt = `你是一个智能助手。请根据参考资料回答用户问题。

${historyText ? `对话历史:\n${historyText}\n` : ''}
用户问题: ${originalQuery}
${rewrittenQuery !== originalQuery ? `理解后的问题: ${rewrittenQuery}\n` : ''}
${context ? `参考资料:\n${context}\n` : '参考资料: 无\n'}
要求:
1. 如果参考资料中有相关信息，基于资料回答
2. 如果参考资料中没有相关信息，尝试用你的知识回答，但要说明这不是来自资料库
3. 保持回答简洁友好

回答:`;
    }
    
    try {
      const response = await this.llm.invoke(prompt);
      return response.content.toString().trim();
    } catch (error) {
      console.error('[ContextManager] 生成失败:', error);
      return '抱歉，生成回答时出错，请稍后重试。';
    }
  }
  
  // ==================== 压缩 ====================
  
  async compressBySummary(sessionId: string): Promise<{
    success: boolean;
    summary?: string;
    compressedCount?: number;
  }> {
    const data = loadSession(sessionId);
    if (!data || data.messages.length < 6) {
      return { success: false };
    }
    
    const oldMsgs = data.messages.slice(0, -4);
    const recentMsgs = data.messages.slice(-4);
    
    if (oldMsgs.length < 4) return { success: false };
    
    const conversation = oldMsgs.map(m => 
      `${m.role === 'user' ? '用户' : 'AI'}: ${m.content}`
    ).join('\n');
    
    try {
      const response = await this.llm.invoke(
        `将以下对话压缩为100-200字摘要:\n\n${conversation}\n\n摘要:`
      );
      const summary = response.content.toString().trim();
      
      data.messages = recentMsgs;
      data.summary = summary;
      data.metadata.summarizedRounds += Math.floor(oldMsgs.length / 2);
      data.metadata.messageCount = recentMsgs.length;
      
      saveSession(data);
      return { success: true, summary, compressedCount: oldMsgs.length };
    } catch {
      return { success: false };
    }
  }
  
  getTokenStats(state: ContextState) {
    const totalTokens = state.messages.reduce((sum, m) => sum + (m.tokenCount || 0), 0);
    return {
      totalTokens,
      messageCount: state.messages.length,
      averageTokensPerMessage: state.messages.length > 0 ? Math.round(totalTokens / state.messages.length) : 0,
      isOverLimit: totalTokens > (this.config.windowConfig.maxTokens || 4000),
    };
  }
}

// ==================== 导出 ====================

export function createContextManager(config: Partial<ContextManagerConfig> = {}): ContextManager {
  return new ContextManager(config);
}

export { DEFAULT_CONFIG as CONTEXT_MANAGER_DEFAULT_CONFIG, estimateTokens, generateId };
