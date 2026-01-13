// BPE (Byte Pair Encoding) Tokenizer with Trie Tree
// 完整的子词化系统，支持多模型切换，包含逻辑瀑布流、向量加权可视化和词元密度热力图

import { v4 as uuidv4 } from 'uuid';
import { AutoTokenizer } from '@xenova/transformers';
import { ObservabilityEngine } from './observability';

// ============= 数据结构定义 =============

export interface TokenInfo {
  token: string;
  tokenId: number;
  position: number;
  type: 'chinese' | 'english' | 'number' | 'punctuation' | 'special' | 'subword';
  // BPE 相关
  subwordParts?: string[];  // 子词组成部分
  mergeOperations?: MergeOperation[];  // 合并操作历史
  frequency?: number;  // 词频
  weight?: number;  // 向量加权值
  density?: number;  // 词元密度
  bpeRank?: number;  // BPE 合并顺序（rank）
}

export interface MergeOperation {
  pair: [string, string];  // 合并的字符对
  newToken: string;  // 合并后的新 token
  frequency: number;  // 合并频率
  step: number;  // 合并步骤
  rank: number;  // BPE rank
}

export interface BPETokenizationResult {
  tokens: TokenInfo[];
  originalText: string;
  processingSteps: ProcessingStep[];  // 逻辑瀑布流
  vectorWeights: VectorWeight[];  // 向量加权信息
  densityHeatmap: DensityPoint[];  // 词元密度热力图数据
  statistics: {
    totalTokens: number;
    uniqueTokens: number;
    subwordRatio: number;
    averageTokenLength: number;
    processingTime: number;
  };
  modelInfo: {
    name: string;
    vocabSize: number;
    mergesCount: number;
  };
}

export interface ProcessingStep {
  step: number;
  stage: 'preprocessing' | 'trie_lookup' | 'bpe_merge' | 'subword_split' | 'finalization';
  action: string;
  input: string;
  output: string;
  decision: string;
  traceId?: string;
  spanId?: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface VectorWeight {
  token: string;
  tokenId: number;
  weight: number;
  position: number;
  contribution: number;  // 对最终向量的贡献度
}

export interface DensityPoint {
  position: number;
  token: string;
  density: number;  // 0-1 之间的密度值
  tokenCount: number;
  contextWindow: number;
}

// ============= Trie 树节点 =============

class TrieNode {
  children: Map<string, TrieNode> = new Map();
  isEndOfWord: boolean = false;
  tokenId?: number;
  frequency: number = 0;
  metadata?: {
    token: string;
    type: TokenInfo['type'];
    weight?: number;
    bpeRank?: number;
  };

  constructor() {}
}

// ============= BPE Tokenizer 核心类 =============

export class BPETokenizer {
  private tokenizer: any = null;
  private loadingPromise: Map<string, Promise<void>> = new Map();
  private currentModel: string = '';
  private trie: TrieNode = new TrieNode();  // Trie 树根节点
  private observabilityEngine?: ObservabilityEngine;
  private traceId?: string;
  
  // 支持的模型列表
  private readonly supportedModels = [
    'Xenova/bert-base-multilingual-cased',
    'Xenova/bge-small-zh-v1.5',
    'Xenova/all-MiniLM-L6-v2'
  ];

  constructor(
    private defaultModel: string = 'Xenova/bert-base-multilingual-cased',
    observabilityEngine?: ObservabilityEngine
  ) {
    this.observabilityEngine = observabilityEngine;
    this.currentModel = defaultModel;
  }

  /**
   * 切换模型
   */
  async switchModel(modelName: string): Promise<void> {
    if (!this.supportedModels.includes(modelName)) {
      throw new Error(`不支持的模型: ${modelName}. 支持的模型: ${this.supportedModels.join(', ')}`);
    }

    if (this.currentModel === modelName && this.tokenizer) {
      console.log(`[BPE Tokenizer] 模型 ${modelName} 已加载，无需切换`);
      return;
    }

    console.log(`[BPE Tokenizer] 切换模型: ${this.currentModel} -> ${modelName}`);
    this.currentModel = modelName;
    this.tokenizer = null;  // 清除旧模型
    this.trie = new TrieNode();  // 重建 Trie 树
    
    // 加载新模型
    await this.initialize();
  }

  /**
   * 获取当前模型
   */
  getCurrentModel(): string {
    return this.currentModel;
  }

  /**
   * 获取支持的模型列表
   */
  getSupportedModels(): string[] {
    return [...this.supportedModels];
  }

  /**
   * 初始化 tokenizer（异步加载模型）
   */
  private async initialize(): Promise<void> {
    if (this.tokenizer && this.currentModel === this.defaultModel) {
      return; // 已经初始化
    }

    if (this.loadingPromise.has(this.currentModel)) {
      return this.loadingPromise.get(this.currentModel)!; // 正在加载，等待完成
    }

    const loadingPromise = (async () => {
      try {
        console.log(`[BPE Tokenizer] 开始加载模型: ${this.currentModel}`);
        
        // 使用 AutoTokenizer 加载指定模型
        this.tokenizer = await AutoTokenizer.from_pretrained(this.currentModel, {
          progress_callback: undefined,
        });
        
        console.log(`[BPE Tokenizer] 模型加载成功: ${this.currentModel}`);
        
        // 构建 Trie 树
        this.buildTrie();
        
        console.log(`[BPE Tokenizer] Trie 树构建完成，节点数: ${this.countTrieNodes(this.trie)}`);
      } catch (error) {
        console.error(`[BPE Tokenizer] 模型加载失败: ${this.currentModel}`, error);
        this.tokenizer = null;
        throw error;
      } finally {
        this.loadingPromise.delete(this.currentModel);
      }
    })();

    this.loadingPromise.set(this.currentModel, loadingPromise);
    return loadingPromise;
  }

  /**
   * 词元化文本（主要入口）
   */
  async tokenize(
    text: string,
    addSpecialTokens: boolean = true,
    parentTraceId?: string
  ): Promise<BPETokenizationResult> {
    const startTime = Date.now();
    
    // 如果有父 Trace ID，创建 Span；否则创建独立的 Trace
    let traceId: string;
    let bpeSpanId: string | undefined;
    
    if (parentTraceId && this.observabilityEngine) {
      traceId = parentTraceId;
      bpeSpanId = this.createSpan(parentTraceId, 'BPE Tokenization', { 
        text, 
        model: this.currentModel 
      });
    } else if (this.observabilityEngine) {
      traceId = this.createTrace('BPE Tokenization');
    } else {
      traceId = uuidv4();
    }
    
    this.traceId = traceId;
    const processingSteps: ProcessingStep[] = [];
    
    try {
      // 确保 tokenizer 已初始化
      await this.initialize();

      if (!this.tokenizer) {
        throw new Error('Tokenizer not initialized');
      }

      // 确保 text 是字符串类型
      if (typeof text !== 'string') {
        text = String(text || '');
      }

      // 如果 text 为空，返回空结果
      if (!text.trim()) {
        return this.createEmptyResult(text, startTime);
      }

      // 步骤 1: 预处理
      const preprocessed = this.preprocess(text, traceId, processingSteps);
      
      // 步骤 2: Trie 树查找（快速匹配已知词）
      const trieMatches = this.trieLookup(preprocessed, traceId, processingSteps);
      
      // 步骤 3: 使用 @xenova/transformers 进行 BPE 编码
      console.log(`[BPE Tokenizer] 开始 BPE 编码，预处理文本: "${preprocessed}"`);
      const bpeResult = await this.encodeWithBPE(preprocessed, traceId, processingSteps);
      console.log(`[BPE Tokenizer] BPE 编码完成:`, {
        inputIdsCount: bpeResult.inputIds?.length || 0,
        tokenTextsCount: bpeResult.tokenTexts?.length || 0,
        mergesCount: bpeResult.merges?.length || 0
      });
      
      // 步骤 4: 构建 TokenInfo 数组
      const tokens = this.buildTokenInfos(bpeResult, preprocessed, traceId, processingSteps);
      console.log(`[BPE Tokenizer] TokenInfo 构建完成: ${tokens.length} 个 tokens`);
      
      // 步骤 5: 添加特殊 tokens
      if (addSpecialTokens) {
        const specialTokens = this.getSpecialTokens();
        if (specialTokens.bos) {
          tokens.unshift({
            token: specialTokens.bos,
            tokenId: bpeResult.bosId || 101,
            position: -1,
            type: 'special'
          });
        }
        if (specialTokens.eos) {
          tokens.push({
            token: specialTokens.eos,
            tokenId: bpeResult.eosId || 102,
            position: tokens.length,
            type: 'special'
          });
        }
      }

      // 步骤 6: 计算向量加权
      const vectorWeights = this.calculateVectorWeights(tokens);
      
      // 步骤 7: 生成词元密度热力图数据
      const densityHeatmap = this.generateDensityHeatmap(tokens, preprocessed);
      
      // 步骤 8: 计算统计信息
      const statistics = this.calculateStatistics(tokens, Date.now() - startTime);
      
      // 获取模型信息
      const modelInfo = this.getModelInfo();

      console.log(`[BPE Tokenizer] 词元化完成: ${tokens.length} tokens, ${processingSteps.length} 步骤, ${vectorWeights.length} 权重, ${densityHeatmap.length} 密度点`);

      const result: BPETokenizationResult = {
        tokens,
        originalText: text,
        processingSteps,
        vectorWeights,
        densityHeatmap,
        statistics,
        modelInfo
      };

      // 完成 Span 或 Trace
      if (bpeSpanId) {
        this.finishSpan(bpeSpanId, {
          tokenCount: tokens.length,
          processingTime: Date.now() - startTime,
          model: this.currentModel
        });
      } else if (this.observabilityEngine) {
        this.finishTrace(traceId, 'SUCCESS', {
          tokenCount: tokens.length,
          processingTime: Date.now() - startTime,
          model: this.currentModel
        });
      }

      return result;

    } catch (error) {
      console.error('[BPE Tokenizer] 词元化错误:', error);
      
      if (bpeSpanId) {
        this.finishSpan(bpeSpanId, { error: String(error) });
      } else if (this.observabilityEngine) {
        this.finishTrace(traceId, 'ERROR', { error: String(error) });
      }
      
      // 即使出错，也返回基本结果
      return this.createEmptyResult(text, startTime);
    }
  }

  /**
   * 预处理文本
   */
  private preprocess(
    text: string,
    traceId: string,
    processingSteps: ProcessingStep[]
  ): string {
    const spanId = this.createSpan(traceId, 'Text Preprocessing', { text });
    
    // Unicode 标准化
    let normalized = text.normalize('NFKC');
    
    // 清理空白字符
    normalized = normalized.replace(/\s+/g, ' ').trim();
    
    processingSteps.push({
      step: processingSteps.length + 1,
      stage: 'preprocessing',
      action: 'Text normalization and cleaning',
      input: text,
      output: normalized,
      decision: 'Applied Unicode normalization (NFKC) and whitespace handling',
      traceId,
      spanId,
      timestamp: Date.now()
    });
    
    this.finishSpan(spanId, { 
      originalLength: text.length,
      normalizedLength: normalized.length 
    });
    
    return normalized;
  }

  /**
   * Trie 树查找
   */
  private trieLookup(
    text: string,
    traceId: string,
    processingSteps: ProcessingStep[]
  ): Map<number, { token: string; start: number; end: number }> {
    const spanId = this.createSpan(traceId, 'Trie Lookup', { text });
    const matches = new Map<number, { token: string; start: number; end: number }>();
    
    let i = 0;
    let matchCount = 0;
    
    while (i < text.length) {
      let node = this.trie;
      let longestMatch = '';
      let matchEnd = i;
      let matchTokenId: number | undefined;

      // 在 Trie 中查找最长匹配
      for (let j = i; j < text.length; j++) {
        const char = text[j];
        const child = node.children.get(char);
        
        if (!child) break;
        
        node = child;
        if (node.isEndOfWord && node.tokenId !== undefined) {
          longestMatch = text.substring(i, j + 1);
          matchEnd = j + 1;
          matchTokenId = node.tokenId;
        }
      }

      if (longestMatch && matchTokenId !== undefined) {
        matches.set(i, {
          token: longestMatch,
          start: i,
          end: matchEnd
        });
        matchCount++;
        
        this.createEvent(traceId, 'Trie Match Found', {
          position: i,
          token: longestMatch,
          tokenId: matchTokenId
        });
        
        processingSteps.push({
          step: processingSteps.length + 1,
          stage: 'trie_lookup',
          action: 'Longest prefix matching',
          input: text.substring(i, Math.min(i + 20, text.length)),
          output: longestMatch,
          decision: `Matched token "${longestMatch}" at position ${i} (Token ID: ${matchTokenId})`,
          traceId,
          spanId,
          timestamp: Date.now(),
          metadata: { tokenId: matchTokenId, start: i, end: matchEnd }
        });

        i = matchEnd;
      } else {
        i++;
      }
    }
    
    this.finishSpan(spanId, { 
      matchesFound: matchCount,
      textLength: text.length 
    });
    
    return matches;
  }

  /**
   * 使用 @xenova/transformers 进行 BPE 编码
   */
  private async encodeWithBPE(
    text: string,
    traceId: string,
    processingSteps: ProcessingStep[]
  ): Promise<{
    inputIds: number[];
    tokenTexts: string[];
    attentionMask?: number[];
    merges?: any[];
    specialTokens?: { bos?: string; eos?: string; bosId?: number; eosId?: number };
  }> {
    const spanId = this.createSpan(traceId, 'BPE Encoding', { text, model: this.currentModel });
    
    try {
      // 使用 tokenizer 进行编码
      let encoded: number[] = [];
      try {
        const encodeResult = this.tokenizer.encode(text, {
          add_special_tokens: false,  // 我们稍后手动添加
          return_tensors: false,
        });
        
        // 确保 encoded 是数组
        if (Array.isArray(encodeResult)) {
          encoded = encodeResult;
        } else if (encodeResult && Array.isArray(encodeResult.input_ids)) {
          encoded = encodeResult.input_ids;
        } else if (encodeResult && typeof encodeResult === 'object' && 'data' in encodeResult) {
          encoded = Array.from(encodeResult.data || []);
        } else {
          console.warn('[BPE Tokenizer] 编码结果格式异常:', encodeResult);
          encoded = [];
        }
      } catch (error) {
        console.error('[BPE Tokenizer] 编码失败:', error);
        throw error;
      }

      console.log(`[BPE Tokenizer] 编码结果: ${encoded.length} 个 token IDs`);

      // 获取 token 文本
      let tokenTexts: string[] = [];
      
      // 尝试多种方法获取 token 文本
      try {
        // 方法1: 使用 tokenizer 的 convert_ids_to_tokens 方法
        if (typeof this.tokenizer.convert_ids_to_tokens === 'function') {
          const converted = this.tokenizer.convert_ids_to_tokens(encoded);
          if (Array.isArray(converted)) {
            tokenTexts = converted;
            console.log(`[BPE Tokenizer] 使用方法1 (convert_ids_to_tokens) 获取到 ${tokenTexts.length} 个 tokens`);
          }
        }
        
        // 如果方法1失败或结果为空，尝试方法2
        if (tokenTexts.length === 0 && typeof this.tokenizer.get_vocab === 'function') {
          const vocab = this.tokenizer.get_vocab();
          if (vocab && typeof vocab === 'object') {
            const idToToken = new Map<number, string>();
            Object.entries(vocab).forEach(([token, id]) => {
              if (token && typeof id === 'number') {
                idToToken.set(id, token);
              }
            });
            tokenTexts = encoded.map((id: number) => idToToken.get(id) || `[UNK:${id}]`);
            console.log(`[BPE Tokenizer] 使用方法2 (get_vocab) 获取到 ${tokenTexts.length} 个 tokens, 词汇表大小: ${idToToken.size}`);
          }
        }
        
        // 如果方法2也失败，尝试方法3: 使用 tokenize 方法
        if (tokenTexts.length === 0 && typeof this.tokenizer.tokenize === 'function') {
          try {
            const tokenized = this.tokenizer.tokenize(text);
            if (Array.isArray(tokenized)) {
              tokenTexts = tokenized;
              console.log(`[BPE Tokenizer] 使用方法3 (tokenize) 获取到 ${tokenTexts.length} 个 tokens`);
              // 如果 tokenize 返回的 tokens 数量与 encoded 不一致，需要调整
              if (tokenTexts.length !== encoded.length) {
                console.warn(`[BPE Tokenizer] tokenize 结果数量 (${tokenTexts.length}) 与 encode 结果数量 (${encoded.length}) 不一致`);
                // 尝试从词汇表补充
                if (typeof this.tokenizer.get_vocab === 'function') {
                  const vocab = this.tokenizer.get_vocab();
                  const idToToken = new Map<number, string>();
                  Object.entries(vocab).forEach(([token, id]) => {
                    if (token && typeof id === 'number') {
                      idToToken.set(id, token);
                    }
                  });
                  tokenTexts = encoded.map((id: number) => idToToken.get(id) || `[TOKEN:${id}]`);
                }
              }
            }
          } catch (error) {
            console.warn('[BPE Tokenizer] tokenize 方法失败:', error);
          }
        }
        
        // 如果所有方法都失败，使用 decode（逐个解码）
        if (tokenTexts.length === 0) {
          tokenTexts = encoded.map((id: number) => {
            try {
              const decoded = this.tokenizer.decode([id], { skip_special_tokens: false });
              return decoded || `[TOKEN:${id}]`;
            } catch {
              return `[TOKEN:${id}]`;
            }
          });
          console.log(`[BPE Tokenizer] 使用方法4 (decode) 获取到 ${tokenTexts.length} 个 tokens`);
        }
      } catch (error) {
        console.error('[BPE Tokenizer] 获取 token 文本失败:', error);
        // 后备：使用 ID 作为文本
        tokenTexts = encoded.map((id: number) => `[TOKEN:${id}]`);
      }
      
      // 确保 tokenTexts 长度与 encoded 一致
      if (tokenTexts.length !== encoded.length) {
        console.warn(`[BPE Tokenizer] tokenTexts 长度 (${tokenTexts.length}) 与 encoded 长度 (${encoded.length}) 不一致，进行调整`);
        while (tokenTexts.length < encoded.length) {
          tokenTexts.push(`[TOKEN:${encoded[tokenTexts.length]}]`);
        }
        tokenTexts = tokenTexts.slice(0, encoded.length);
      }
      
      console.log(`[BPE Tokenizer] 最终获取到 ${tokenTexts.length} 个 token 文本`);

      // 获取 BPE merges（如果可用）
      let merges: any[] = [];
      try {
        if (this.tokenizer.model?.merges && Array.isArray(this.tokenizer.model.merges)) {
          merges = this.tokenizer.model.merges;
        } else if (this.tokenizer.merges && Array.isArray(this.tokenizer.merges)) {
          merges = this.tokenizer.merges;
        } else if (this.tokenizer.model?.bpe_merges && Array.isArray(this.tokenizer.model.bpe_merges)) {
          merges = this.tokenizer.model.bpe_merges;
        } else if (this.tokenizer.bpe_merges && Array.isArray(this.tokenizer.bpe_merges)) {
          merges = this.tokenizer.bpe_merges;
        }
        console.log(`[BPE Tokenizer] 获取到 ${merges.length} 个 BPE merges`);
      } catch (error) {
        console.warn('[BPE Tokenizer] 无法获取 BPE merges:', error);
      }

      // 获取特殊 tokens
      const specialTokens = this.extractSpecialTokens();

      processingSteps.push({
        step: processingSteps.length + 1,
        stage: 'bpe_merge',
        action: 'BPE encoding with model',
        input: text,
        output: `${tokenTexts.length} tokens`,
        decision: `Encoded using ${this.currentModel}, found ${merges.length} BPE merges`,
        traceId,
        spanId,
        timestamp: Date.now(),
        metadata: { 
          model: this.currentModel,
          tokenCount: tokenTexts.length,
          mergesCount: merges.length
        }
      });

      this.finishSpan(spanId, {
        tokenCount: tokenTexts.length,
        mergesCount: merges.length,
        model: this.currentModel
      });

      return {
        inputIds: encoded,
        tokenTexts,
        merges,
        specialTokens
      };
    } catch (error) {
      this.finishSpan(spanId, { error: String(error) });
      throw error;
    }
  }

  /**
   * 提取特殊 tokens
   */
  private extractSpecialTokens(): { bos?: string; eos?: string; bosId?: number; eosId?: number } {
    const specialTokens: any = {};
    
    try {
      // 尝试获取特殊 tokens
      if (this.tokenizer.bos_token) {
        specialTokens.bos = this.tokenizer.bos_token;
        specialTokens.bosId = this.tokenizer.bos_token_id;
      }
      if (this.tokenizer.eos_token) {
        specialTokens.eos = this.tokenizer.eos_token;
        specialTokens.eosId = this.tokenizer.eos_token_id;
      }
      if (this.tokenizer.cls_token) {
        specialTokens.bos = this.tokenizer.cls_token;
        specialTokens.bosId = this.tokenizer.cls_token_id;
      }
      if (this.tokenizer.sep_token) {
        specialTokens.eos = this.tokenizer.sep_token;
        specialTokens.eosId = this.tokenizer.sep_token_id;
      }
    } catch (error) {
      console.warn('[BPE Tokenizer] 无法提取特殊 tokens:', error);
    }
    
    return specialTokens;
  }

  /**
   * 获取特殊 tokens（用于添加）
   */
  private getSpecialTokens(): { bos?: string; eos?: string } {
    return this.extractSpecialTokens();
  }

  /**
   * 构建 TokenInfo 数组
   */
  private buildTokenInfos(
    bpeResult: {
      inputIds: number[];
      tokenTexts: string[];
      merges?: any[];
    },
    originalText: string,
    traceId: string,
    processingSteps: ProcessingStep[]
  ): TokenInfo[] {
    const spanId = this.createSpan(traceId, 'Token Info Construction', {});
    const tokenInfos: TokenInfo[] = [];

    // 构建 BPE rank 映射（如果 merges 可用）
    const bpeRankMap = new Map<string, number>();
    if (bpeResult.merges && Array.isArray(bpeResult.merges)) {
      bpeResult.merges.forEach((merge: any, index: number) => {
        if (typeof merge === 'string') {
          bpeRankMap.set(merge, index + 1);
        } else if (Array.isArray(merge) && merge.length === 2) {
          const merged = merge[0] + merge[1];
          bpeRankMap.set(merged, index + 1);
        }
      });
    }

    bpeResult.inputIds.forEach((tokenId: number, index: number) => {
      const tokenText = bpeResult.tokenTexts[index] || `[TOKEN:${tokenId}]`;
      const tokenType = this.getTokenType(tokenText);
      const subwordParts = this.extractSubwordParts(tokenText);
      const bpeRank = bpeRankMap.get(tokenText);

      tokenInfos.push({
        token: tokenText,
        tokenId: tokenId,
        position: index,
        type: tokenType,
        subwordParts,
        mergeOperations: bpeRank ? [{
          pair: [tokenText[0] || '', tokenText.slice(1) || ''],
          newToken: tokenText,
          frequency: 1,
          step: bpeRank,
          rank: bpeRank
        }] : undefined,
        frequency: 1,
        weight: 1.0,  // 初始权重，后续会计算
        density: 0.0,  // 初始密度，后续会计算
        bpeRank
      });
    });

    processingSteps.push({
      step: processingSteps.length + 1,
      stage: 'finalization',
      action: 'Token info construction',
      input: `${bpeResult.inputIds.length} BPE tokens`,
      output: `${tokenInfos.length} TokenInfo objects`,
      decision: 'Created TokenInfo with metadata and BPE information',
      traceId,
      spanId,
      timestamp: Date.now()
    });

    this.finishSpan(spanId, { tokensCreated: tokenInfos.length });
    return tokenInfos;
  }

  /**
   * 计算向量加权
   */
  private calculateVectorWeights(tokens: TokenInfo[]): VectorWeight[] {
    const weights: VectorWeight[] = [];
    const totalTokens = tokens.length;

    tokens.forEach((token, index) => {
      // 基于位置的权重（位置越靠前，权重越高）
      const positionWeight = 1.0 - (index / totalTokens) * 0.3;
      
      // 基于频率的权重
      const frequencyWeight = Math.log((token.frequency || 1) + 1) / Math.log(100);
      
      // 基于类型的权重
      const typeWeight = this.getTypeWeight(token.type);
      
      // 基于 BPE rank 的权重（rank 越小，权重越高）
      const bpeWeight = token.bpeRank ? 1.0 / (1.0 + token.bpeRank / 1000) : 1.0;
      
      // 综合权重
      const weight = positionWeight * 0.3 + frequencyWeight * 0.2 + typeWeight * 0.3 + bpeWeight * 0.2;
      
      // 贡献度（归一化）
      const contribution = weight / totalTokens;

      weights.push({
        token: token.token,
        tokenId: token.tokenId,
        weight,
        position: index,
        contribution
      });

      // 更新 token 的权重
      token.weight = weight;
    });

    return weights;
  }

  /**
   * 生成词元密度热力图数据
   */
  private generateDensityHeatmap(
    tokens: TokenInfo[],
    text: string
  ): DensityPoint[] {
    const heatmap: DensityPoint[] = [];
    const contextWindow = 5;  // 上下文窗口大小

    tokens.forEach((token, index) => {
      // 计算当前位置的密度
      const start = Math.max(0, index - contextWindow);
      const end = Math.min(tokens.length, index + contextWindow + 1);
      const windowTokens = tokens.slice(start, end);
      
      // 密度 = 窗口内 token 数量 / 窗口大小
      const density = windowTokens.length / (2 * contextWindow + 1);
      
      // Token 计数（窗口内的唯一 token 数）
      const uniqueTokens = new Set(windowTokens.map(t => t.token)).size;
      
      heatmap.push({
        position: index,
        token: token.token,
        density,
        tokenCount: uniqueTokens,
        contextWindow
      });

      // 更新 token 的密度
      token.density = density;
    });

    return heatmap;
  }

  /**
   * 计算统计信息
   */
  private calculateStatistics(
    tokens: TokenInfo[],
    processingTime: number
  ): BPETokenizationResult['statistics'] {
    const uniqueTokens = new Set(tokens.map(t => t.token)).size;
    const subwordCount = tokens.filter(t => t.subwordParts && t.subwordParts.length > 1).length;
    const totalLength = tokens.reduce((sum, t) => sum + t.token.length, 0);

    return {
      totalTokens: tokens.length,
      uniqueTokens,
      subwordRatio: tokens.length > 0 ? subwordCount / tokens.length : 0,
      averageTokenLength: tokens.length > 0 ? totalLength / tokens.length : 0,
      processingTime
    };
  }

  /**
   * 获取模型信息
   */
  private getModelInfo(): BPETokenizationResult['modelInfo'] {
    let vocabSize = 0;
    let mergesCount = 0;
    
    try {
      if (this.tokenizer) {
        // 获取词汇表大小
        if (typeof this.tokenizer.get_vocab === 'function') {
          const vocab = this.tokenizer.get_vocab();
          if (vocab && typeof vocab === 'object') {
            vocabSize = Object.keys(vocab).length;
          }
        } else if (this.tokenizer.vocab && typeof this.tokenizer.vocab === 'object') {
          vocabSize = Object.keys(this.tokenizer.vocab).length;
        } else if (this.tokenizer.model?.vocab && typeof this.tokenizer.model.vocab === 'object') {
          vocabSize = Object.keys(this.tokenizer.model.vocab).length;
        }
        
        // 获取 BPE merges 数量
        if (this.tokenizer.model?.merges && Array.isArray(this.tokenizer.model.merges)) {
          mergesCount = this.tokenizer.model.merges.length;
        } else if (this.tokenizer.merges && Array.isArray(this.tokenizer.merges)) {
          mergesCount = this.tokenizer.merges.length;
        } else if (this.tokenizer.model?.bpe_merges && Array.isArray(this.tokenizer.model.bpe_merges)) {
          mergesCount = this.tokenizer.model.bpe_merges.length;
        } else if (this.tokenizer.bpe_merges && Array.isArray(this.tokenizer.bpe_merges)) {
          mergesCount = this.tokenizer.bpe_merges.length;
        }
        
        console.log(`[BPE Tokenizer] 模型信息: ${this.currentModel}, 词汇表大小: ${vocabSize}, BPE 合并数: ${mergesCount}`);
      }
    } catch (error) {
      console.warn('[BPE Tokenizer] 获取模型信息失败:', error);
    }
    
    return {
      name: this.currentModel,
      vocabSize,
      mergesCount
    };
  }

  /**
   * 创建空结果
   */
  private createEmptyResult(text: string, startTime: number): BPETokenizationResult {
    return {
      tokens: [],
      originalText: text,
      processingSteps: [],
      vectorWeights: [],
      densityHeatmap: [],
      statistics: {
        totalTokens: 0,
        uniqueTokens: 0,
        subwordRatio: 0,
        averageTokenLength: 0,
        processingTime: Date.now() - startTime
      },
      modelInfo: {
        name: this.currentModel,
        vocabSize: 0,
        mergesCount: 0
      }
    };
  }

  // ============= 辅助方法 =============

  private buildTrie(): void {
    this.trie = new TrieNode();
    
    if (!this.tokenizer) return;
    
    try {
      // 从 tokenizer 的词汇表构建 Trie
      const vocab = this.tokenizer.get_vocab();
      if (vocab && typeof vocab === 'object') {
        Object.entries(vocab).forEach(([token, id]) => {
          if (token && typeof id === 'number') {
            let node = this.trie;
            for (const char of token) {
              if (!node.children.has(char)) {
                node.children.set(char, new TrieNode());
              }
              node = node.children.get(char)!;
            }
            node.isEndOfWord = true;
            node.tokenId = id;
            node.metadata = {
              token,
              type: this.getTokenType(token)
            };
          }
        });
      }
    } catch (error) {
      console.warn('[BPE Tokenizer] 构建 Trie 树失败:', error);
    }
  }

  private countTrieNodes(node: TrieNode): number {
    let count = 1;
    node.children.forEach(child => {
      count += this.countTrieNodes(child);
    });
    return count;
  }

  private getTokenType(token: string): TokenInfo['type'] {
    if (/^\[(CLS|SEP|PAD|UNK|MASK|BOS|EOS)\]|<(UNK|CLS|SEP|PAD|MASK|BOS|EOS)>/i.test(token)) {
      return 'special';
    }
    if (/^[a-zA-Z]+$/.test(token)) {
      return 'english';
    }
    if (/^[\u4e00-\u9fff]+$/.test(token)) {
      return 'chinese';
    }
    if (/^[0-9]+$/.test(token)) {
      return 'number';
    }
    if (/^[.,!?:;()"'\-/\[\]{}]+$/.test(token)) {
      return 'punctuation';
    }
    // 包含子词标记的视为子词
    if (token.includes('##') || token.includes('</w>') || (token.length > 1 && /[a-zA-Z\u4e00-\u9fff]/.test(token))) {
      return 'subword';
    }
    return 'special';
  }

  private extractSubwordParts(token: string): string[] {
    // 处理 BPE 标记（如 ## 前缀）
    if (token.startsWith('##')) {
      return ['##', token.substring(2)];
    }
    if (token.includes('</w>')) {
      return token.split('</w>').filter(p => p);
    }
    // 简单的字符分割
    if (token.length <= 1) return [token];
    const parts: string[] = [];
    let current = '';
    for (const char of token) {
      if (/[a-zA-Z\u4e00-\u9fff]/.test(char)) {
        current += char;
      } else {
        if (current) {
          parts.push(current);
          current = '';
        }
        parts.push(char);
      }
    }
    if (current) parts.push(current);
    return parts.length > 1 ? parts : [token];
  }

  private getTypeWeight(type: TokenInfo['type']): number {
    const weights: Record<TokenInfo['type'], number> = {
      chinese: 1.2,
      english: 1.0,
      number: 0.8,
      punctuation: 0.5,
      special: 0.3,
      subword: 1.1
    };
    return weights[type] || 1.0;
  }

  // ============= Observability 方法 =============

  private createTrace(name: string): string {
    if (!this.observabilityEngine) return uuidv4();
    return this.observabilityEngine.createTrace({
      name,
      input: {},
      metadata: { component: 'BPETokenizer', model: this.currentModel }
    });
  }

  private finishTrace(traceId: string, status: 'SUCCESS' | 'ERROR', output: any): void {
    if (!this.observabilityEngine) return;
    this.observabilityEngine.updateTrace(traceId, {
      output,
      status: status === 'SUCCESS' ? 'SUCCESS' : 'ERROR',
      endTime: new Date()
    });
  }

  private createSpan(traceId: string, name: string, input: any): string {
    if (!this.observabilityEngine) return uuidv4();
    return this.observabilityEngine.createSpan({
      traceId,
      name,
      input,
      metadata: { component: 'BPETokenizer', model: this.currentModel }
    });
  }

  private finishSpan(spanId: string, output: any): void {
    if (!this.observabilityEngine) return;
    this.observabilityEngine.updateObservation(spanId, {
      output,
      endTime: new Date()
    });
  }

  private createEvent(traceId: string, name: string, input: any): string {
    if (!this.observabilityEngine) return uuidv4();
    return this.observabilityEngine.createEvent({
      traceId,
      name,
      input,
      metadata: { component: 'BPETokenizer', model: this.currentModel }
    });
  }
}
