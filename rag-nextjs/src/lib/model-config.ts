/**
 * 统一模型配置系统
 * 
 * 支持通过环境变量控制使用本地 Ollama 或生产 API (OpenAI/Azure/其他)
 * 
 * 架构设计：
 * 1. ModelProvider: 模型提供商枚举 (ollama, openai, azure, custom)
 * 2. ModelType: 模型类型枚举 (llm, embedding, reasoning)
 * 3. ModelConfig: 模型配置接口
 * 4. ModelFactory: 模型工厂类，统一创建模型实例
 * 5. ModelRegistry: 模型注册表，支持动态添加模型
 */

import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Embeddings } from '@langchain/core/embeddings';

// ==================== 类型定义 ====================

/** 模型提供商 */
export type ModelProvider = 'ollama' | 'openai' | 'azure' | 'custom';

/** 模型类型 */
export type ModelType = 'llm' | 'embedding' | 'reasoning';

/** 模型配置接口 */
export interface ModelConfig {
  provider: ModelProvider;
  modelName: string;
  apiKey?: string;
  baseUrl?: string;
  temperature?: number;
  maxTokens?: number;
  /** 模型维度 (仅 embedding 模型) */
  dimension?: number;
  /** 额外配置 */
  options?: Record<string, any>;
}

/** 环境变量配置 */
export interface EnvConfig {
  // 主开关：控制使用本地还是生产模型
  MODEL_PROVIDER: ModelProvider;
  
  // Ollama 配置
  OLLAMA_BASE_URL: string;
  OLLAMA_LLM_MODEL: string;
  OLLAMA_EMBEDDING_MODEL: string;
  OLLAMA_REASONING_MODEL: string;
  
  // OpenAI 配置
  OPENAI_API_KEY?: string;
  OPENAI_BASE_URL?: string;
  OPENAI_LLM_MODEL: string;
  OPENAI_EMBEDDING_MODEL: string;
  OPENAI_REASONING_MODEL: string;
  
  // Azure OpenAI 配置
  AZURE_OPENAI_API_KEY?: string;
  AZURE_OPENAI_ENDPOINT?: string;
  AZURE_OPENAI_LLM_DEPLOYMENT?: string;
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT?: string;
  
  // 自定义 API 配置
  CUSTOM_API_KEY?: string;
  CUSTOM_BASE_URL?: string;
  CUSTOM_LLM_MODEL?: string;
  CUSTOM_EMBEDDING_MODEL?: string;
}

/** 模型实例缓存 */
interface ModelCache {
  llm: Map<string, BaseChatModel>;
  embedding: Map<string, Embeddings>;
  reasoning: Map<string, BaseChatModel>;
}

/** 动态模型注册项 */
export interface DynamicModelEntry {
  id: string;
  type: ModelType;
  config: ModelConfig;
  description?: string;
  createdAt: number;
}

// ==================== 常量定义 ====================

/** 默认模型维度映射 */
export const MODEL_DIMENSIONS: Record<string, number> = {
  // Ollama Embedding 模型
  'nomic-embed-text': 768,
  'nomic-embed-text-v2-moe': 768,
  'bge-m3': 1024,
  'bge-large': 1024,
  'all-minilm': 384,
  'mxbai-embed-large': 1024,
  'snowflake-arctic-embed': 1024,
  'qwen3-embedding': 1024,
  
  // OpenAI Embedding 模型
  'text-embedding-3-small': 1536,
  'text-embedding-3-large': 3072,
  'text-embedding-ada-002': 1536,
};

/** 默认模型配置 */
const DEFAULT_OLLAMA_CONFIG = {
  llm: 'llama3.1',
  embedding: 'nomic-embed-text',
  reasoning: 'deepseek-r1',
};

const DEFAULT_OPENAI_CONFIG = {
  llm: 'gpt-4o-mini',
  embedding: 'text-embedding-3-small',
  reasoning: 'gpt-4o',
};

// ==================== 环境变量解析 ====================

/**
 * 从环境变量读取配置
 */
export function loadEnvConfig(): EnvConfig {
  return {
    // 主开关
    MODEL_PROVIDER: (process.env.MODEL_PROVIDER as ModelProvider) || 'ollama',
    
    // Ollama
    OLLAMA_BASE_URL: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
    OLLAMA_LLM_MODEL: process.env.OLLAMA_LLM_MODEL || DEFAULT_OLLAMA_CONFIG.llm,
    OLLAMA_EMBEDDING_MODEL: process.env.OLLAMA_EMBEDDING_MODEL || DEFAULT_OLLAMA_CONFIG.embedding,
    OLLAMA_REASONING_MODEL: process.env.OLLAMA_REASONING_MODEL || DEFAULT_OLLAMA_CONFIG.reasoning,
    
    // OpenAI
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
    OPENAI_LLM_MODEL: process.env.OPENAI_LLM_MODEL || DEFAULT_OPENAI_CONFIG.llm,
    OPENAI_EMBEDDING_MODEL: process.env.OPENAI_EMBEDDING_MODEL || DEFAULT_OPENAI_CONFIG.embedding,
    OPENAI_REASONING_MODEL: process.env.OPENAI_REASONING_MODEL || DEFAULT_OPENAI_CONFIG.reasoning,
    
    // Azure
    AZURE_OPENAI_API_KEY: process.env.AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT: process.env.AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_LLM_DEPLOYMENT: process.env.AZURE_OPENAI_LLM_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: process.env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    
    // Custom
    CUSTOM_API_KEY: process.env.CUSTOM_API_KEY,
    CUSTOM_BASE_URL: process.env.CUSTOM_BASE_URL,
    CUSTOM_LLM_MODEL: process.env.CUSTOM_LLM_MODEL,
    CUSTOM_EMBEDDING_MODEL: process.env.CUSTOM_EMBEDDING_MODEL,
  };
}

// ==================== 模型注册表 ====================

/**
 * 模型注册表 - 管理动态添加的模型
 */
class ModelRegistry {
  private static instance: ModelRegistry;
  private models: Map<string, DynamicModelEntry> = new Map();
  
  private constructor() {}
  
  static getInstance(): ModelRegistry {
    if (!ModelRegistry.instance) {
      ModelRegistry.instance = new ModelRegistry();
    }
    return ModelRegistry.instance;
  }
  
  /**
   * 注册新模型
   */
  register(entry: Omit<DynamicModelEntry, 'createdAt'>): void {
    const fullEntry: DynamicModelEntry = {
      ...entry,
      createdAt: Date.now(),
    };
    this.models.set(entry.id, fullEntry);
    console.log(`[ModelRegistry] 已注册模型: ${entry.id} (${entry.type})`);
  }
  
  /**
   * 注销模型
   */
  unregister(id: string): boolean {
    const deleted = this.models.delete(id);
    if (deleted) {
      console.log(`[ModelRegistry] 已注销模型: ${id}`);
    }
    return deleted;
  }
  
  /**
   * 获取模型配置
   */
  get(id: string): DynamicModelEntry | undefined {
    return this.models.get(id);
  }
  
  /**
   * 获取所有模型
   */
  getAll(): DynamicModelEntry[] {
    return Array.from(this.models.values());
  }
  
  /**
   * 按类型获取模型
   */
  getByType(type: ModelType): DynamicModelEntry[] {
    return this.getAll().filter(m => m.type === type);
  }
  
  /**
   * 检查模型是否存在
   */
  has(id: string): boolean {
    return this.models.has(id);
  }
  
  /**
   * 清空注册表
   */
  clear(): void {
    this.models.clear();
    console.log('[ModelRegistry] 已清空所有注册模型');
  }
}

// ==================== 模型工厂 ====================

/**
 * 模型工厂类 - 统一创建和管理模型实例
 */
export class ModelFactory {
  private static instance: ModelFactory;
  private envConfig: EnvConfig;
  private cache: ModelCache = {
    llm: new Map(),
    embedding: new Map(),
    reasoning: new Map(),
  };
  private registry: ModelRegistry;
  
  private constructor() {
    this.envConfig = loadEnvConfig();
    this.registry = ModelRegistry.getInstance();
    console.log(`[ModelFactory] 初始化完成, 当前提供商: ${this.envConfig.MODEL_PROVIDER}`);
  }
  
  static getInstance(): ModelFactory {
    if (!ModelFactory.instance) {
      ModelFactory.instance = new ModelFactory();
    }
    return ModelFactory.instance;
  }
  
  /**
   * 重新加载环境配置
   */
  reloadConfig(): void {
    this.envConfig = loadEnvConfig();
    this.clearCache();
    console.log(`[ModelFactory] 配置已重新加载, 当前提供商: ${this.envConfig.MODEL_PROVIDER}`);
  }
  
  /**
   * 获取当前提供商
   */
  getProvider(): ModelProvider {
    return this.envConfig.MODEL_PROVIDER;
  }
  
  /**
   * 获取当前环境配置
   */
  getEnvConfig(): EnvConfig {
    return { ...this.envConfig };
  }
  
  /**
   * 动态注册模型
   */
  registerModel(entry: Omit<DynamicModelEntry, 'createdAt'>): void {
    this.registry.register(entry);
  }
  
  /**
   * 获取已注册的模型列表
   */
  getRegisteredModels(): DynamicModelEntry[] {
    return this.registry.getAll();
  }
  
  // ==================== LLM 模型 ====================
  
  /**
   * 创建 LLM 模型实例
   * @param modelName 可选的模型名称，不提供则使用环境变量配置
   * @param options 额外配置选项
   */
  createLLM(modelName?: string, options: Partial<ModelConfig> = {}): BaseChatModel {
    const provider = this.envConfig.MODEL_PROVIDER;
    const cacheKey = `${provider}:${modelName || 'default'}:${JSON.stringify(options)}`;
    
    // 检查缓存
    if (this.cache.llm.has(cacheKey)) {
      return this.cache.llm.get(cacheKey)!;
    }
    
    let llm: BaseChatModel;
    
    switch (provider) {
      case 'ollama':
        llm = this.createOllamaLLM(modelName, options);
        break;
      case 'openai':
        llm = this.createOpenAILLM(modelName, options);
        break;
      case 'azure':
        llm = this.createAzureLLM(modelName, options);
        break;
      case 'custom':
        llm = this.createCustomLLM(modelName, options);
        break;
      default:
        throw new Error(`不支持的模型提供商: ${provider}`);
    }
    
    this.cache.llm.set(cacheKey, llm);
    return llm;
  }
  
  private createOllamaLLM(modelName?: string, options: Partial<ModelConfig> = {}): ChatOllama {
    const actualModel = modelName || this.envConfig.OLLAMA_LLM_MODEL;
    console.log(`[ModelFactory] 创建 Ollama LLM: ${actualModel}`);
    
    return new ChatOllama({
      baseUrl: options.baseUrl || this.envConfig.OLLAMA_BASE_URL,
      model: actualModel,
      temperature: options.temperature ?? 0.7,
      ...options.options,
    });
  }
  
  private createOpenAILLM(modelName?: string, options: Partial<ModelConfig> = {}): ChatOpenAI {
    const actualModel = modelName || this.envConfig.OPENAI_LLM_MODEL;
    const apiKey = options.apiKey || this.envConfig.OPENAI_API_KEY;
    
    if (!apiKey) {
      throw new Error('OpenAI API Key 未配置。请设置 OPENAI_API_KEY 环境变量。');
    }
    
    console.log(`[ModelFactory] 创建 OpenAI LLM: ${actualModel}`);
    
    return new ChatOpenAI({
      openAIApiKey: apiKey,
      modelName: actualModel,
      temperature: options.temperature ?? 0.7,
      maxTokens: options.maxTokens,
      configuration: this.envConfig.OPENAI_BASE_URL ? {
        baseURL: options.baseUrl || this.envConfig.OPENAI_BASE_URL,
      } : undefined,
      ...options.options,
    });
  }
  
  private createAzureLLM(modelName?: string, options: Partial<ModelConfig> = {}): ChatOpenAI {
    const deployment = modelName || this.envConfig.AZURE_OPENAI_LLM_DEPLOYMENT;
    const apiKey = options.apiKey || this.envConfig.AZURE_OPENAI_API_KEY;
    const endpoint = options.baseUrl || this.envConfig.AZURE_OPENAI_ENDPOINT;
    
    if (!apiKey || !endpoint) {
      throw new Error('Azure OpenAI 配置不完整。请设置 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT。');
    }
    
    console.log(`[ModelFactory] 创建 Azure OpenAI LLM: ${deployment}`);
    
    return new ChatOpenAI({
      azureOpenAIApiKey: apiKey,
      azureOpenAIApiDeploymentName: deployment,
      azureOpenAIApiInstanceName: endpoint.replace('https://', '').replace('.openai.azure.com', ''),
      azureOpenAIApiVersion: '2024-02-15-preview',
      temperature: options.temperature ?? 0.7,
      maxTokens: options.maxTokens,
      ...options.options,
    });
  }
  
  private createCustomLLM(modelName?: string, options: Partial<ModelConfig> = {}): ChatOpenAI {
    const actualModel = modelName || this.envConfig.CUSTOM_LLM_MODEL || 'default';
    const apiKey = options.apiKey || this.envConfig.CUSTOM_API_KEY;
    const baseUrl = options.baseUrl || this.envConfig.CUSTOM_BASE_URL;
    
    if (!apiKey || !baseUrl) {
      throw new Error('自定义 API 配置不完整。请设置 CUSTOM_API_KEY 和 CUSTOM_BASE_URL。');
    }
    
    console.log(`[ModelFactory] 创建自定义 LLM: ${actualModel} @ ${baseUrl}`);
    
    // 使用 OpenAI 兼容 API
    return new ChatOpenAI({
      openAIApiKey: apiKey,
      modelName: actualModel,
      temperature: options.temperature ?? 0.7,
      maxTokens: options.maxTokens,
      configuration: {
        baseURL: baseUrl,
      },
      ...options.options,
    });
  }
  
  // ==================== Embedding 模型 ====================
  
  /**
   * 创建 Embedding 模型实例
   * @param modelName 可选的模型名称
   * @param options 额外配置选项
   */
  createEmbedding(modelName?: string, options: Partial<ModelConfig> = {}): Embeddings {
    const provider = this.envConfig.MODEL_PROVIDER;
    const cacheKey = `${provider}:${modelName || 'default'}:${JSON.stringify(options)}`;
    
    if (this.cache.embedding.has(cacheKey)) {
      return this.cache.embedding.get(cacheKey)!;
    }
    
    let embedding: Embeddings;
    
    switch (provider) {
      case 'ollama':
        embedding = this.createOllamaEmbedding(modelName, options);
        break;
      case 'openai':
      case 'azure':
      case 'custom':
        embedding = this.createOpenAIEmbedding(modelName, options);
        break;
      default:
        throw new Error(`不支持的模型提供商: ${provider}`);
    }
    
    this.cache.embedding.set(cacheKey, embedding);
    return embedding;
  }
  
  private createOllamaEmbedding(modelName?: string, options: Partial<ModelConfig> = {}): OllamaEmbeddings {
    const actualModel = modelName || this.envConfig.OLLAMA_EMBEDDING_MODEL;
    console.log(`[ModelFactory] 创建 Ollama Embedding: ${actualModel}`);
    
    return new OllamaEmbeddings({
      baseUrl: options.baseUrl || this.envConfig.OLLAMA_BASE_URL,
      model: actualModel,
      ...options.options,
    });
  }
  
  private createOpenAIEmbedding(modelName?: string, options: Partial<ModelConfig> = {}): OpenAIEmbeddings {
    const provider = this.envConfig.MODEL_PROVIDER;
    let actualModel: string;
    let apiKey: string | undefined;
    let baseUrl: string | undefined;
    
    if (provider === 'azure') {
      actualModel = modelName || this.envConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT || 'text-embedding-ada-002';
      apiKey = options.apiKey || this.envConfig.AZURE_OPENAI_API_KEY;
      
      if (!apiKey) {
        throw new Error('Azure OpenAI API Key 未配置。');
      }
      
      console.log(`[ModelFactory] 创建 Azure Embedding: ${actualModel}`);
      
      return new OpenAIEmbeddings({
        azureOpenAIApiKey: apiKey,
        azureOpenAIApiDeploymentName: actualModel,
        azureOpenAIApiInstanceName: this.envConfig.AZURE_OPENAI_ENDPOINT?.replace('https://', '').replace('.openai.azure.com', ''),
        azureOpenAIApiVersion: '2024-02-15-preview',
        ...options.options,
      });
    } else if (provider === 'custom') {
      actualModel = modelName || this.envConfig.CUSTOM_EMBEDDING_MODEL || 'text-embedding-3-small';
      apiKey = options.apiKey || this.envConfig.CUSTOM_API_KEY;
      baseUrl = options.baseUrl || this.envConfig.CUSTOM_BASE_URL;
      
      if (!apiKey || !baseUrl) {
        throw new Error('自定义 API 配置不完整。');
      }
      
      console.log(`[ModelFactory] 创建自定义 Embedding: ${actualModel} @ ${baseUrl}`);
      
      return new OpenAIEmbeddings({
        openAIApiKey: apiKey,
        modelName: actualModel,
        configuration: {
          baseURL: baseUrl,
        },
        ...options.options,
      });
    } else {
      actualModel = modelName || this.envConfig.OPENAI_EMBEDDING_MODEL;
      apiKey = options.apiKey || this.envConfig.OPENAI_API_KEY;
      baseUrl = options.baseUrl || this.envConfig.OPENAI_BASE_URL;
      
      if (!apiKey) {
        throw new Error('OpenAI API Key 未配置。请设置 OPENAI_API_KEY 环境变量。');
      }
      
      console.log(`[ModelFactory] 创建 OpenAI Embedding: ${actualModel}`);
      
      return new OpenAIEmbeddings({
        openAIApiKey: apiKey,
        modelName: actualModel,
        configuration: baseUrl ? { baseURL: baseUrl } : undefined,
        ...options.options,
      });
    }
  }
  
  // ==================== Reasoning 模型 ====================
  
  /**
   * 创建推理模型实例 (用于复杂推理任务)
   * @param modelName 可选的模型名称
   * @param options 额外配置选项
   */
  createReasoningModel(modelName?: string, options: Partial<ModelConfig> = {}): BaseChatModel {
    const provider = this.envConfig.MODEL_PROVIDER;
    const cacheKey = `reasoning:${provider}:${modelName || 'default'}:${JSON.stringify(options)}`;
    
    if (this.cache.reasoning.has(cacheKey)) {
      return this.cache.reasoning.get(cacheKey)!;
    }
    
    // 推理模型通常需要更低的 temperature
    const reasoningOptions = {
      ...options,
      temperature: options.temperature ?? 0,
    };
    
    let model: BaseChatModel;
    
    switch (provider) {
      case 'ollama':
        const ollamaModel = modelName || this.envConfig.OLLAMA_REASONING_MODEL;
        console.log(`[ModelFactory] 创建 Ollama 推理模型: ${ollamaModel}`);
        model = new ChatOllama({
          baseUrl: reasoningOptions.baseUrl || this.envConfig.OLLAMA_BASE_URL,
          model: ollamaModel,
          temperature: reasoningOptions.temperature,
          ...reasoningOptions.options,
        });
        break;
        
      case 'openai':
        const openaiModel = modelName || this.envConfig.OPENAI_REASONING_MODEL;
        console.log(`[ModelFactory] 创建 OpenAI 推理模型: ${openaiModel}`);
        model = this.createOpenAILLM(openaiModel, reasoningOptions);
        break;
        
      case 'azure':
      case 'custom':
        model = this.createLLM(modelName, reasoningOptions);
        break;
        
      default:
        throw new Error(`不支持的模型提供商: ${provider}`);
    }
    
    this.cache.reasoning.set(cacheKey, model);
    return model;
  }
  
  // ==================== 辅助方法 ====================
  
  /**
   * 获取模型维度
   */
  getModelDimension(modelName?: string): number {
    const actualModel = modelName || 
      (this.envConfig.MODEL_PROVIDER === 'ollama' 
        ? this.envConfig.OLLAMA_EMBEDDING_MODEL 
        : this.envConfig.OPENAI_EMBEDDING_MODEL);
    
    return MODEL_DIMENSIONS[actualModel] || 768;
  }
  
  /**
   * 根据维度选择合适的模型
   */
  selectModelByDimension(dimension: number): string {
    const provider = this.envConfig.MODEL_PROVIDER;
    
    // 按提供商筛选模型
    const providerModels = provider === 'ollama' 
      ? ['nomic-embed-text', 'bge-m3', 'all-minilm', 'qwen3-embedding']
      : ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'];
    
    for (const model of providerModels) {
      if (MODEL_DIMENSIONS[model] === dimension) {
        return model;
      }
    }
    
    // 返回默认模型
    return provider === 'ollama' ? 'nomic-embed-text' : 'text-embedding-3-small';
  }
  
  /**
   * 清空模型缓存
   */
  clearCache(): void {
    this.cache.llm.clear();
    this.cache.embedding.clear();
    this.cache.reasoning.clear();
    console.log('[ModelFactory] 模型缓存已清空');
  }
  
  /**
   * 获取当前配置摘要
   */
  getConfigSummary(): {
    provider: ModelProvider;
    llmModel: string;
    embeddingModel: string;
    reasoningModel: string;
    baseUrl: string;
    hasApiKey: boolean;
  } {
    const provider = this.envConfig.MODEL_PROVIDER;
    
    return {
      provider,
      llmModel: provider === 'ollama' 
        ? this.envConfig.OLLAMA_LLM_MODEL 
        : this.envConfig.OPENAI_LLM_MODEL,
      embeddingModel: provider === 'ollama'
        ? this.envConfig.OLLAMA_EMBEDDING_MODEL
        : this.envConfig.OPENAI_EMBEDDING_MODEL,
      reasoningModel: provider === 'ollama'
        ? this.envConfig.OLLAMA_REASONING_MODEL
        : this.envConfig.OPENAI_REASONING_MODEL,
      baseUrl: provider === 'ollama'
        ? this.envConfig.OLLAMA_BASE_URL
        : (this.envConfig.OPENAI_BASE_URL || 'https://api.openai.com'),
      hasApiKey: provider === 'ollama' || !!this.envConfig.OPENAI_API_KEY,
    };
  }
  
  /**
   * 验证配置是否有效
   */
  validateConfig(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    const provider = this.envConfig.MODEL_PROVIDER;
    
    switch (provider) {
      case 'openai':
        if (!this.envConfig.OPENAI_API_KEY) {
          errors.push('OPENAI_API_KEY 环境变量未设置');
        }
        break;
        
      case 'azure':
        if (!this.envConfig.AZURE_OPENAI_API_KEY) {
          errors.push('AZURE_OPENAI_API_KEY 环境变量未设置');
        }
        if (!this.envConfig.AZURE_OPENAI_ENDPOINT) {
          errors.push('AZURE_OPENAI_ENDPOINT 环境变量未设置');
        }
        break;
        
      case 'custom':
        if (!this.envConfig.CUSTOM_API_KEY) {
          errors.push('CUSTOM_API_KEY 环境变量未设置');
        }
        if (!this.envConfig.CUSTOM_BASE_URL) {
          errors.push('CUSTOM_BASE_URL 环境变量未设置');
        }
        break;
        
      case 'ollama':
        // Ollama 不需要 API Key，但需要确保服务可用
        break;
    }
    
    return {
      valid: errors.length === 0,
      errors,
    };
  }
}

// ==================== 便捷导出函数 ====================

/**
 * 获取全局模型工厂实例
 */
export function getModelFactory(): ModelFactory {
  return ModelFactory.getInstance();
}

/**
 * 快捷创建 LLM
 */
export function createLLM(modelName?: string, options?: Partial<ModelConfig>): BaseChatModel {
  return getModelFactory().createLLM(modelName, options);
}

/**
 * 快捷创建 Embedding
 */
export function createEmbedding(modelName?: string, options?: Partial<ModelConfig>): Embeddings {
  return getModelFactory().createEmbedding(modelName, options);
}

/**
 * 快捷创建推理模型
 */
export function createReasoningModel(modelName?: string, options?: Partial<ModelConfig>): BaseChatModel {
  return getModelFactory().createReasoningModel(modelName, options);
}

/**
 * 获取模型维度
 */
export function getModelDimension(modelName?: string): number {
  return getModelFactory().getModelDimension(modelName);
}

/**
 * 根据维度选择模型
 */
export function selectModelByDimension(dimension: number): string {
  return getModelFactory().selectModelByDimension(dimension);
}

/**
 * 获取当前提供商
 */
export function getCurrentProvider(): ModelProvider {
  return getModelFactory().getProvider();
}

/**
 * 获取配置摘要
 */
export function getConfigSummary() {
  return getModelFactory().getConfigSummary();
}

// ==================== 类型守卫 ====================

/**
 * 检查是否为 Ollama 提供商
 */
export function isOllamaProvider(): boolean {
  return getModelFactory().getProvider() === 'ollama';
}

/**
 * 检查是否为 OpenAI 提供商
 */
export function isOpenAIProvider(): boolean {
  return getModelFactory().getProvider() === 'openai';
}

// ==================== 原有兼容层 ====================

/**
 * 兼容旧版 OllamaEmbeddings 导出
 * @deprecated 请使用 createEmbedding()
 */
export function getOllamaEmbeddings(modelName?: string): OllamaEmbeddings {
  const factory = getModelFactory();
  if (factory.getProvider() !== 'ollama') {
    console.warn('[ModelFactory] 当前提供商不是 Ollama，但请求了 OllamaEmbeddings');
  }
  return factory.createEmbedding(modelName) as OllamaEmbeddings;
}

/**
 * 兼容旧版 ChatOllama 导出
 * @deprecated 请使用 createLLM()
 */
export function getChatOllama(modelName?: string): ChatOllama {
  const factory = getModelFactory();
  if (factory.getProvider() !== 'ollama') {
    console.warn('[ModelFactory] 当前提供商不是 Ollama，但请求了 ChatOllama');
  }
  return factory.createLLM(modelName) as ChatOllama;
}
