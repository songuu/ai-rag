'use strict';

/**
 * 自适应实体路由 RAG (Adaptive Entity-Routing RAG)
 * 
 * 基于 LangGraph 的四层架构设计：
 * 1. 认知解析层 (Cognitive Parsing Layer) - 实体提取与意图分类
 * 2. 策略控制层 (Strategic Control Layer) - 实体校验、自适应路由、约束松弛
 * 3. 执行检索层 (Execution Layer) - 结构化/语义检索、混合重排序
 * 4. 数据基础设施层 (Data Infrastructure Layer) - 向量数据库、实体元数据存储
 */

import { ChatOllama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { getMilvusInstance, MilvusVectorStore, MilvusSearchResult, getModelDimension } from './milvus-client';

// ==================== 类型定义 ====================

/** 实体类型 */
export type EntityType = 'PERSON' | 'ORGANIZATION' | 'LOCATION' | 'PRODUCT' | 'DATE' | 'EVENT' | 'CONCEPT' | 'OTHER';

/** 逻辑操作符 */
export type LogicalOperator = 'AND' | 'OR' | 'NOT';

/** 意图类型 */
export type IntentType = 'factual' | 'conceptual' | 'comparison' | 'procedural' | 'exploratory';

/** 提取的实体 */
export interface ExtractedEntity {
  name: string;
  type: EntityType;
  value: string;
  confidence: number;
  normalized?: string;  // 归一化后的名称
  aliases?: string[];   // 同义词
}

/** 逻辑关系 */
export interface LogicalRelation {
  operator: LogicalOperator;
  entities: string[];
  description: string;
}

/** 解析结果 */
export interface ParsedQuery {
  originalQuery: string;
  entities: ExtractedEntity[];
  logicalRelations: LogicalRelation[];
  intent: IntentType;
  complexity: 'simple' | 'moderate' | 'complex';
  confidence: number;
  keywords: string[];
}

/** 校验后的实体 */
export interface ValidatedEntity extends ExtractedEntity {
  isValid: boolean;
  normalizedName: string;
  matchScore: number;
  suggestions?: string[];
}

/** 检索条件 */
export interface SearchConstraint {
  field: string;
  operator: 'eq' | 'contains' | 'in' | 'range' | 'not';
  value: string | string[] | { min?: any; max?: any };
  priority: number;  // 优先级，越高越重要
}

/** 路由决策 */
export interface RoutingDecision {
  action: 'structured_search' | 'semantic_search' | 'hybrid_search' | 'relax_constraints' | 'generate_response';
  constraints: SearchConstraint[];
  relaxedConstraints?: string[];  // 已松弛的约束
  retryCount: number;
  maxRetries: number;
  reason: string;
}

/** 检索结果 */
export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  matchType: 'structured' | 'semantic' | 'hybrid';
}

/** 重排序后的结果 */
export interface RankedResult extends SearchResult {
  rerankedScore: number;
  relevanceExplanation: string;
}

/** 工作流状态 */
export interface WorkflowState {
  query: ParsedQuery;
  validatedEntities: ValidatedEntity[];
  currentDecision: RoutingDecision;
  searchResults: SearchResult[];
  rankedResults: RankedResult[];
  finalResponse: string;
  steps: WorkflowStep[];
  totalDuration: number;
}

/** 工作流步骤 */
export interface WorkflowStep {
  step: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  duration?: number;
  details?: any;
  error?: string;
}

/** 实体元数据 */
export interface EntityMetadata {
  standardName: string;
  type: EntityType;
  aliases: string[];
  hierarchy?: string[];  // 层级关系，如 ['中国', '北京', '朝阳']
  relatedEntities?: string[];
  embedding?: number[];
}

/** 配置选项 */
export interface AdaptiveRAGConfig {
  llmModel: string;
  embeddingModel: string;
  maxRetries: number;
  constraintPriority: EntityType[];  // 约束松弛优先级（后面的先松弛）
  minResultCount: number;
  similarityThreshold: number;
  enableReranking: boolean;
  milvusCollection: string;
}

// ==================== 默认配置 ====================

const DEFAULT_CONFIG: AdaptiveRAGConfig = {
  llmModel: 'qwen2.5:7b',
  embeddingModel: 'nomic-embed-text',
  maxRetries: 3,
  constraintPriority: ['PERSON', 'ORGANIZATION', 'PRODUCT', 'EVENT', 'LOCATION', 'DATE', 'CONCEPT', 'OTHER'],
  minResultCount: 3,
  similarityThreshold: 0.6,
  enableReranking: true,
  milvusCollection: 'rag_documents',
};

// ==================== Prompts ====================

const ENTITY_EXTRACTION_PROMPT = `你是一个专业的认知解析引擎，负责将用户的自然语言查询转换为结构化的数据对象。

用户查询: {query}

## 任务说明
请深入分析用户查询，提取以下结构化信息：

1. **实体提取**：识别查询中的所有命名实体（人名、组织、地点、产品、日期、事件、概念等）
2. **逻辑关系**：识别实体之间的逻辑运算关系（AND并且、OR或者、NOT排除）
3. **意图分类**：判断用户查询的根本目的
4. **复杂度评估**：评估查询的处理难度

## 返回格式（严格JSON）
{
  "entities": [
    {
      "name": "规范化的实体名称",
      "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|DATE|EVENT|CONCEPT|OTHER",
      "value": "用户原始输入的值",
      "confidence": 0.85
    }
  ],
  "logicalRelations": [
    {
      "operator": "AND|OR|NOT",
      "entities": ["实体1", "实体2"],
      "description": "关系的自然语言描述"
    }
  ],
  "intent": "factual|conceptual|comparison|procedural|exploratory",
  "complexity": "simple|moderate|complex",
  "confidence": 0.85,
  "keywords": ["核心关键词1", "核心关键词2"]
}

## 实体类型定义
- **PERSON**: 人名、角色、职位（如：马斯克、产品经理）
- **ORGANIZATION**: 组织、公司、团队、机构（如：特斯拉、OpenAI）
- **LOCATION**: 地理位置、地点、区域（如：北京、硅谷）
- **PRODUCT**: 产品、品牌、型号（如：iPhone 15、Model S）
- **DATE**: 时间点、时间段、年份（如：2024年、去年）
- **EVENT**: 事件、活动、发布会（如：WWDC、双十一）
- **CONCEPT**: 概念、术语、技术名词（如：RAG、向量检索）
- **OTHER**: 其他无法分类的实体

## 意图类型定义
- **factual**: 寻求具体事实或数据（谁/什么/哪里/何时）
- **conceptual**: 理解概念或原理（为什么/什么意思/如何理解）
- **comparison**: 对比分析多个对象（A和B哪个好/区别是什么）
- **procedural**: 寻求操作步骤或方法（怎么做/如何实现）
- **exploratory**: 开放式探索或发现（有什么/还有哪些）

## 逻辑关系识别示例
- "北京和上海的房价" → AND关系：[北京, 上海]
- "苹果或华为的手机" → OR关系：[苹果, 华为]
- "不包括进口的汽车" → NOT关系：[进口]
- "iPhone 15在北京的价格" → 暗含AND：[iPhone 15, 北京]

## 复杂度评估标准
- **simple**: 单一实体、直接问答
- **moderate**: 2-3个实体、需要简单推理
- **complex**: 多实体、多关系、需要复杂推理或跨领域知识

请只返回严格的JSON格式，不要添加任何其他解释文字。`;

const ENTITY_RESOLUTION_PROMPT = `你是一个实体校验专家。请判断用户输入的实体是否与标准实体库中的实体匹配。

用户输入实体: {userEntity}
用户输入类型: {userType}

候选标准实体列表:
{candidates}

请判断用户输入最可能对应哪个标准实体，或者是否是一个新实体。

返回JSON格式：
{
  "isMatch": true/false,
  "matchedEntity": "匹配的标准实体名称（如果匹配）",
  "confidence": 0.0-1.0,
  "normalizedName": "归一化后的名称",
  "suggestions": ["可能的其他匹配"]
}

只返回JSON。`;

const RERANKING_PROMPT = `你是一个文档相关性评估专家。请评估以下文档与用户查询的相关性。

用户查询: {query}
查询意图: {intent}
提取的实体: {entities}

文档内容:
{document}

请评估这个文档与查询的相关性，返回JSON格式：
{
  "relevanceScore": 0.0-1.0,
  "explanation": "相关性解释",
  "matchedEntities": ["匹配的实体"],
  "keyInformation": "文档中的关键信息"
}

只返回JSON。`;

const RESPONSE_GENERATION_PROMPT = `你是一个专业的问答助手。请基于检索到的上下文回答用户的问题。

用户问题: {query}
查询意图: {intent}
提取的实体: {entities}

检索到的上下文:
{context}

请给出准确、完整的回答。如果上下文信息不足以回答问题，请诚实说明。

回答要求：
1. 准确引用上下文中的信息
2. 根据查询意图调整回答风格
3. 如果是事实性问题，给出明确答案
4. 如果是概念性问题，给出清晰解释
5. 如果信息不足，说明已知信息和未知部分`;

// ==================== 预处理器 ====================

/**
 * 实体预处理器
 * 在 LLM 调用前使用规则和词典进行预识别，提高小模型的准确率
 */
class EntityPreprocessor {
  // 常见中文地名别称映射（作为预处理词典）
  private static readonly LOCATION_ALIASES: Record<string, string> = {
    '魔都': '上海', '帝都': '北京', '妖都': '广州', '羊城': '广州',
    '蓉城': '成都', '鹏城': '深圳', '江城': '武汉', '山城': '重庆',
    '泉城': '济南', '冰城': '哈尔滨', '春城': '昆明', '榕城': '福州',
    '石城': '南京', '星城': '长沙', '花城': '广州', '雾都': '重庆',
  };

  // 预处理结果
  static preprocess(query: string): {
    normalizedQuery: string;
    preMappedEntities: { original: string; normalized: string; type: EntityType }[];
  } {
    let normalizedQuery = query;
    const preMappedEntities: { original: string; normalized: string; type: EntityType }[] = [];

    // 检测并标记地名别称
    for (const [alias, standard] of Object.entries(this.LOCATION_ALIASES)) {
      if (query.includes(alias)) {
        preMappedEntities.push({
          original: alias,
          normalized: standard,
          type: 'LOCATION',
        });
        // 在查询中添加标注，帮助 LLM 理解
        normalizedQuery = normalizedQuery.replace(alias, `${alias}(即${standard})`);
      }
    }

    return { normalizedQuery, preMappedEntities };
  }

  // 后处理：校验和修正 LLM 输出
  static postprocess(
    entities: ExtractedEntity[], 
    preMappedEntities: { original: string; normalized: string; type: EntityType }[]
  ): ExtractedEntity[] {
    const correctedEntities: ExtractedEntity[] = [];
    const processedNames = new Set<string>();

    // 首先添加预映射的实体
    for (const preEntity of preMappedEntities) {
      correctedEntities.push({
        name: preEntity.normalized,
        type: preEntity.type,
        value: preEntity.original,
        confidence: 0.95, // 预映射的高置信度
      });
      processedNames.add(preEntity.original);
      processedNames.add(preEntity.normalized);
    }

    // 处理 LLM 返回的实体
    for (const entity of entities) {
      // 检查是否已经处理过（通过预映射）
      if (processedNames.has(entity.name) || processedNames.has(entity.value)) {
        continue;
      }

      // 检查是否是别称被错误分类
      const aliasCheck = this.LOCATION_ALIASES[entity.name] || this.LOCATION_ALIASES[entity.value];
      if (aliasCheck) {
        correctedEntities.push({
          name: aliasCheck,
          type: 'LOCATION',
          value: entity.value || entity.name,
          confidence: 0.9,
        });
        processedNames.add(entity.name);
        continue;
      }

      // 保留原实体
      correctedEntities.push(entity);
      processedNames.add(entity.name);
    }

    return correctedEntities;
  }
}

// ==================== 核心类实现 ====================

/**
 * 第一层：认知解析层
 * 负责实体提取和意图分类
 */
export class CognitiveParser {
  private llm: ChatOllama;
  private modelName: string;

  constructor(model: string) {
    this.modelName = model;
    this.llm = new ChatOllama({
      model,
      temperature: 0.1,
      format: 'json',
    });
  }

  /**
   * 检测模型能力等级
   */
  private getModelCapabilityLevel(): 'low' | 'medium' | 'high' {
    const modelLower = this.modelName.toLowerCase();
    
    // 小模型（参数量 < 3B）
    if (modelLower.includes('0.5b') || modelLower.includes('1b') || modelLower.includes('2b')) {
      return 'low';
    }
    // 中等模型（3B-13B）
    if (modelLower.includes('3b') || modelLower.includes('7b') || modelLower.includes('8b')) {
      return 'medium';
    }
    // 大模型（> 13B）
    return 'high';
  }

  /**
   * 解析用户查询，提取实体和逻辑关系
   */
  async parse(query: string): Promise<ParsedQuery> {
    const capability = this.getModelCapabilityLevel();
    
    // 对于低能力模型，使用预处理增强
    const { normalizedQuery, preMappedEntities } = capability === 'low' 
      ? EntityPreprocessor.preprocess(query)
      : { normalizedQuery: query, preMappedEntities: [] };
    
    if (preMappedEntities.length > 0) {
      console.log(`[CognitiveParser] 预处理识别到 ${preMappedEntities.length} 个实体:`, 
        preMappedEntities.map(e => `${e.original}→${e.normalized}`).join(', '));
    }

    try {
      const prompt = ENTITY_EXTRACTION_PROMPT.replace('{query}', normalizedQuery);
      const response = await this.llm.invoke(prompt);
      const content = typeof response.content === 'string' 
        ? response.content 
        : JSON.stringify(response.content);
      
      const parsed = this.safeParseJson(content);
      
      // 提取 LLM 返回的实体
      let entities = (parsed.entities || []).map((e: any) => ({
        name: e.name || '',
        type: this.normalizeEntityType(e.type),
        value: e.value || e.name || '',
        confidence: parseFloat(e.confidence) || 0.8,
      }));

      // 对于低能力模型，应用后处理校验
      if (capability === 'low') {
        entities = EntityPreprocessor.postprocess(entities, preMappedEntities);
      }

      return {
        originalQuery: query,
        entities,
        logicalRelations: parsed.logicalRelations || [],
        intent: this.normalizeIntent(parsed.intent),
        complexity: parsed.complexity || 'moderate',
        confidence: parseFloat(parsed.confidence) || 0.8,
        keywords: parsed.keywords || [],
      };
    } catch (error) {
      console.error('[CognitiveParser] 解析失败:', error);
      // 降级处理：基于规则提取
      return this.fallbackParse(query, preMappedEntities);
    }
  }

  private normalizeEntityType(type: string): EntityType {
    const normalized = (type || '').toUpperCase();
    const validTypes: EntityType[] = ['PERSON', 'ORGANIZATION', 'LOCATION', 'PRODUCT', 'DATE', 'EVENT', 'CONCEPT', 'OTHER'];
    return validTypes.includes(normalized as EntityType) ? normalized as EntityType : 'OTHER';
  }

  private normalizeIntent(intent: string): IntentType {
    const normalized = (intent || '').toLowerCase();
    const validIntents: IntentType[] = ['factual', 'conceptual', 'comparison', 'procedural', 'exploratory'];
    return validIntents.includes(normalized as IntentType) ? normalized as IntentType : 'factual';
  }

  private fallbackParse(
    query: string, 
    preMappedEntities: { original: string; normalized: string; type: EntityType }[] = []
  ): ParsedQuery {
    // 基于规则的简单提取
    const entities: ExtractedEntity[] = [];
    const keywords: string[] = [];
    const processedNames = new Set<string>();

    // 首先添加预映射的实体（最高优先级）
    for (const preEntity of preMappedEntities) {
      entities.push({
        name: preEntity.normalized,
        type: preEntity.type,
        value: preEntity.original,
        confidence: 0.95,
      });
      processedNames.add(preEntity.original);
      processedNames.add(preEntity.normalized);
    }

    // 提取引号中的内容作为实体
    const quotedMatches = query.match(/["'"](.*?)["'"]/g);
    if (quotedMatches) {
      quotedMatches.forEach(match => {
        const value = match.replace(/["'"]/g, '');
        if (!processedNames.has(value)) {
          entities.push({
            name: value,
            type: 'OTHER',
            value,
            confidence: 0.7,
          });
          processedNames.add(value);
        }
      });
    }

    // 提取可能的产品名称（连续的英文+数字）
    const productMatches = query.match(/[A-Za-z]+\s*\d+(\s*[A-Za-z]*)?/g);
    if (productMatches) {
      productMatches.forEach(match => {
        const trimmed = match.trim();
        if (!processedNames.has(trimmed)) {
          entities.push({
            name: trimmed,
            type: 'PRODUCT',
            value: trimmed,
            confidence: 0.6,
          });
          processedNames.add(trimmed);
        }
      });
    }

    // 提取年份
    const yearMatches = query.match(/\d{4}年?/g);
    if (yearMatches) {
      yearMatches.forEach(match => {
        if (!processedNames.has(match)) {
          entities.push({
            name: match,
            type: 'DATE',
            value: match,
            confidence: 0.9,
          });
          processedNames.add(match);
        }
      });
    }

    // 提取关键词（去除停用词）
    const stopWords = ['的', '是', '在', '和', '与', '或', '了', '吗', '呢', '啊', '什么', '哪', '如何', '怎么', '怎样'];
    const words = query.split(/[\s,，。？！\?!]+/).filter(w => w.length > 1 && !stopWords.includes(w));
    keywords.push(...words.slice(0, 5));

    // 判断意图
    let intent: IntentType = 'factual';
    if (query.includes('是') && query.includes('么') || query.includes('是否') || query.includes('是不是')) {
      intent = 'comparison'; // "X是Y么" 类型的确认问题
    } else if (query.includes('比较') || query.includes('对比') || query.includes('区别')) {
      intent = 'comparison';
    } else if (query.includes('如何') || query.includes('怎么') || query.includes('步骤')) {
      intent = 'procedural';
    } else if (query.includes('什么是') || query.includes('解释') || query.includes('含义')) {
      intent = 'conceptual';
    }

    return {
      originalQuery: query,
      entities,
      logicalRelations: [],
      intent,
      complexity: entities.length > 2 ? 'complex' : entities.length > 0 ? 'moderate' : 'simple',
      confidence: 0.6,
      keywords,
    };
  }

  private safeParseJson(content: string): any {
    try {
      return JSON.parse(content);
    } catch {
      // 尝试提取 JSON
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          return JSON.parse(jsonMatch[0]);
        } catch {
          return {};
        }
      }
      return {};
    }
  }
}

/**
 * 第二层：策略控制层
 * 维护状态，执行校验、路由和约束松弛
 */
export class StrategyController {
  private llm: ChatOllama;
  private entityMetadataStore: EntityMetadataStore;
  private config: AdaptiveRAGConfig;

  constructor(config: AdaptiveRAGConfig, entityMetadataStore: EntityMetadataStore) {
    this.config = config;
    this.entityMetadataStore = entityMetadataStore;
    this.llm = new ChatOllama({
      model: config.llmModel,
      temperature: 0.1,
      format: 'json',
    });
  }

  /**
   * 校验实体
   */
  async validateEntities(entities: ExtractedEntity[]): Promise<ValidatedEntity[]> {
    const validated: ValidatedEntity[] = [];

    for (const entity of entities) {
      // 获取候选实体
      const candidates = await this.entityMetadataStore.findSimilar(entity.name, entity.type, 5);
      
      if (candidates.length === 0) {
        // 没有候选，直接使用原始实体
        validated.push({
          ...entity,
          isValid: true,
          normalizedName: entity.name,
          matchScore: 1.0,
        });
        continue;
      }

      // 快速规则匹配
      const exactMatch = candidates.find(c => 
        c.standardName.toLowerCase() === entity.name.toLowerCase() ||
        c.aliases.some(a => a.toLowerCase() === entity.name.toLowerCase())
      );

      if (exactMatch) {
        validated.push({
          ...entity,
          isValid: true,
          normalizedName: exactMatch.standardName,
          matchScore: 1.0,
          normalized: exactMatch.standardName,
          aliases: exactMatch.aliases,
        });
        continue;
      }

      // 使用 LLM 进行模糊匹配
      try {
        const candidatesList = candidates.map(c => 
          `- ${c.standardName} (别名: ${c.aliases.join(', ')})`
        ).join('\n');

        const prompt = ENTITY_RESOLUTION_PROMPT
          .replace('{userEntity}', entity.name)
          .replace('{userType}', entity.type)
          .replace('{candidates}', candidatesList);

        const response = await this.llm.invoke(prompt);
        const content = typeof response.content === 'string' 
          ? response.content 
          : JSON.stringify(response.content);
        const result = this.safeParseJson(content);

        validated.push({
          ...entity,
          isValid: result.isMatch !== false,
          normalizedName: result.normalizedName || result.matchedEntity || entity.name,
          matchScore: parseFloat(result.confidence) || 0.7,
          suggestions: result.suggestions || [],
        });
      } catch (error) {
        console.error('[StrategyController] 实体校验失败:', error);
        validated.push({
          ...entity,
          isValid: true,
          normalizedName: entity.name,
          matchScore: 0.5,
        });
      }
    }

    return validated;
  }

  /**
   * 路由决策
   */
  makeRoutingDecision(
    query: ParsedQuery,
    validatedEntities: ValidatedEntity[],
    previousDecision?: RoutingDecision,
    resultCount: number = 0
  ): RoutingDecision {
    const retryCount = previousDecision?.retryCount || 0;
    const relaxedConstraints = previousDecision?.relaxedConstraints || [];

    // 如果有结果，直接生成响应
    if (resultCount >= this.config.minResultCount) {
      return {
        action: 'generate_response',
        constraints: previousDecision?.constraints || [],
        relaxedConstraints,
        retryCount,
        maxRetries: this.config.maxRetries,
        reason: `找到 ${resultCount} 个相关结果，准备生成回答`,
      };
    }

    // 如果已达到最大重试次数，降级为纯语义检索
    if (retryCount >= this.config.maxRetries) {
      return {
        action: 'semantic_search',
        constraints: [],
        relaxedConstraints,
        retryCount,
        maxRetries: this.config.maxRetries,
        reason: '多次尝试后仍无结果，降级为纯语义检索',
      };
    }

    // 如果是概念性问题，直接使用语义检索
    if (query.intent === 'conceptual' || query.intent === 'exploratory') {
      return {
        action: 'semantic_search',
        constraints: [],
        relaxedConstraints: [],
        retryCount: 0,
        maxRetries: this.config.maxRetries,
        reason: '概念性/探索性问题，使用语义检索',
      };
    }

    // 构建约束条件
    const constraints: SearchConstraint[] = validatedEntities
      .filter(e => e.isValid && !relaxedConstraints.includes(e.type))
      .map((entity, index) => ({
        field: this.getFieldNameForType(entity.type),
        operator: 'contains' as const,
        value: entity.normalizedName,
        priority: this.config.constraintPriority.indexOf(entity.type),
      }))
      .sort((a, b) => a.priority - b.priority);

    // 如果没有约束或者之前检索无结果，进行约束松弛
    if (constraints.length === 0 || (previousDecision && resultCount === 0)) {
      // 找到优先级最低的约束进行松弛
      const typeToRelax = this.findLowestPriorityType(validatedEntities, relaxedConstraints);
      
      if (typeToRelax && retryCount < this.config.maxRetries) {
        return {
          action: 'relax_constraints',
          constraints: constraints.filter(c => c.field !== this.getFieldNameForType(typeToRelax)),
          relaxedConstraints: [...relaxedConstraints, typeToRelax],
          retryCount: retryCount + 1,
          maxRetries: this.config.maxRetries,
          reason: `移除 ${typeToRelax} 约束，进行更宽泛的检索`,
        };
      }
    }

    // 有约束条件，使用结构化检索
    if (constraints.length > 0) {
      return {
        action: 'structured_search',
        constraints,
        relaxedConstraints,
        retryCount,
        maxRetries: this.config.maxRetries,
        reason: `使用 ${constraints.length} 个过滤条件进行结构化检索`,
      };
    }

    // 默认使用混合检索
    return {
      action: 'hybrid_search',
      constraints: [],
      relaxedConstraints,
      retryCount,
      maxRetries: this.config.maxRetries,
      reason: '无有效约束，使用混合检索',
    };
  }

  private findLowestPriorityType(entities: ValidatedEntity[], relaxed: string[]): EntityType | null {
    const unreleaxedTypes = entities
      .filter(e => e.isValid && !relaxed.includes(e.type))
      .map(e => e.type);

    if (unreleaxedTypes.length === 0) return null;

    // 按优先级排序（低优先级在前）
    const sorted = [...unreleaxedTypes].sort((a, b) => 
      this.config.constraintPriority.indexOf(b) - this.config.constraintPriority.indexOf(a)
    );

    return sorted[0];
  }

  private getFieldNameForType(type: EntityType): string {
    const mapping: Record<EntityType, string> = {
      PERSON: 'person',
      ORGANIZATION: 'organization',
      LOCATION: 'location',
      PRODUCT: 'product',
      DATE: 'date',
      EVENT: 'event',
      CONCEPT: 'concept',
      OTHER: 'content',
    };
    return mapping[type] || 'content';
  }

  private safeParseJson(content: string): any {
    try {
      return JSON.parse(content);
    } catch {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          return JSON.parse(jsonMatch[0]);
        } catch {
          return {};
        }
      }
      return {};
    }
  }
}

/**
 * 第三层：执行检索层
 * 执行具体的检索操作
 */
export class SearchExecutor {
  private embeddings: OllamaEmbeddings;
  private config: AdaptiveRAGConfig;
  private llm: ChatOllama;
  private milvus: MilvusVectorStore | null = null;

  constructor(config: AdaptiveRAGConfig) {
    this.config = config;
    this.embeddings = new OllamaEmbeddings({
      model: config.embeddingModel,
    });
    this.llm = new ChatOllama({
      model: config.llmModel,
      temperature: 0.1,
    });
  }

  /**
   * 获取或初始化 Milvus 实例
   * 注意：前端已确保选择的 embedding 模型与集合维度兼容
   */
  private async getMilvusClient(): Promise<MilvusVectorStore> {
    if (!this.milvus) {
      // 使用配置的 embedding 模型维度
      const dimension = getModelDimension(this.config.embeddingModel) || 768;
      
      this.milvus = getMilvusInstance({
        collectionName: this.config.milvusCollection,
        embeddingDimension: dimension,
      });
      
      // 连接并验证维度
      try {
        await this.milvus.connect();
        const stats = await this.milvus.getCollectionStats();
        const collectionDimension = stats?.embeddingDimension;
        
        if (collectionDimension && collectionDimension !== dimension) {
          console.error(`[SearchExecutor] ⚠️ 维度不匹配: 模型 ${this.config.embeddingModel} (${dimension}D) vs 集合 (${collectionDimension}D)`);
          console.error(`[SearchExecutor] 请在前端选择与知识库兼容的 embedding 模型`);
        }
      } catch (error) {
        console.log('[SearchExecutor] 无法验证集合维度，继续使用配置的维度');
      }
    }
    return this.milvus;
  }

  /**
   * 结构化检索（带过滤条件）
   */
  async structuredSearch(
    query: string,
    constraints: SearchConstraint[],
    topK: number = 10
  ): Promise<SearchResult[]> {
    try {
      // 构建 Milvus 过滤表达式
      const filterExpr = this.buildFilterExpression(constraints);
      
      // 将实体名称加入查询文本以增强语义搜索
      // 这样向量搜索能更好地找到包含这些实体的文档
      const entityValues = constraints
        .filter(c => c.operator === 'contains' && c.value)
        .map(c => String(c.value));
      
      const enhancedQuery = entityValues.length > 0
        ? `${query} ${entityValues.join(' ')}`
        : query;
      
      console.log(`[SearchExecutor] 增强查询: "${enhancedQuery.substring(0, 100)}..."`);
      
      // 生成查询向量
      const queryVector = await this.embeddings.embedQuery(enhancedQuery);

      // 获取 Milvus 客户端并执行搜索
      const milvus = await this.getMilvusClient();
      const results = await milvus.search(
        queryVector,
        topK,
        this.config.similarityThreshold,
        filterExpr || undefined
      );

      // 对结果进行实体匹配后处理（提升包含实体的文档得分）
      return results.map((r: MilvusSearchResult) => {
        let boostedScore = r.score;
        
        // 检查内容中是否包含目标实体
        if (entityValues.length > 0) {
          const contentLower = r.content.toLowerCase();
          const matchedEntities = entityValues.filter(e => 
            contentLower.includes(e.toLowerCase())
          );
          
          // 每匹配一个实体，提升 10% 的得分
          if (matchedEntities.length > 0) {
            boostedScore = Math.min(1.0, r.score * (1 + 0.1 * matchedEntities.length));
          }
        }
        
        return {
          id: r.id,
          content: r.content,
          score: boostedScore,
          metadata: r.metadata || {},
          matchType: 'structured' as const,
        };
      });
    } catch (error) {
      console.error('[SearchExecutor] 结构化检索失败:', error);
      return [];
    }
  }

  /**
   * 语义检索（纯向量搜索）
   */
  async semanticSearch(query: string, topK: number = 10): Promise<SearchResult[]> {
    try {
      const queryVector = await this.embeddings.embedQuery(query);

      // 获取 Milvus 客户端并执行搜索
      const milvus = await this.getMilvusClient();
      const results = await milvus.search(
        queryVector,
        topK,
        this.config.similarityThreshold
      );

      return results.map((r: MilvusSearchResult) => ({
        id: r.id,
        content: r.content,
        score: r.score,
        metadata: r.metadata || {},
        matchType: 'semantic' as const,
      }));
    } catch (error) {
      console.error('[SearchExecutor] 语义检索失败:', error);
      return [];
    }
  }

  /**
   * 混合检索
   */
  async hybridSearch(
    query: string,
    constraints: SearchConstraint[],
    topK: number = 10
  ): Promise<SearchResult[]> {
    // 并行执行结构化和语义检索
    const [structuredResults, semanticResults] = await Promise.all([
      this.structuredSearch(query, constraints, topK),
      this.semanticSearch(query, topK),
    ]);

    // 合并去重
    const resultMap = new Map<string, SearchResult>();
    
    structuredResults.forEach(r => {
      resultMap.set(r.id, { ...r, matchType: 'hybrid' });
    });
    
    semanticResults.forEach(r => {
      if (resultMap.has(r.id)) {
        // 已存在，取更高分数
        const existing = resultMap.get(r.id)!;
        if (r.score > existing.score) {
          resultMap.set(r.id, { ...r, matchType: 'hybrid' });
        }
      } else {
        resultMap.set(r.id, { ...r, matchType: 'hybrid' });
      }
    });

    // 按分数排序
    return Array.from(resultMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * 重排序
   */
  async rerank(
    results: SearchResult[],
    query: ParsedQuery,
    topK: number = 5
  ): Promise<RankedResult[]> {
    if (!this.config.enableReranking || results.length === 0) {
      return results.map(r => ({
        ...r,
        rerankedScore: r.score,
        relevanceExplanation: '未启用重排序',
      }));
    }

    const rankedResults: RankedResult[] = [];

    for (const result of results.slice(0, Math.min(results.length, 10))) {
      try {
        const prompt = RERANKING_PROMPT
          .replace('{query}', query.originalQuery)
          .replace('{intent}', query.intent)
          .replace('{entities}', query.entities.map(e => e.name).join(', '))
          .replace('{document}', result.content.substring(0, 1500));

        const response = await this.llm.invoke(prompt);
        const content = typeof response.content === 'string' 
          ? response.content 
          : JSON.stringify(response.content);
        
        const parsed = this.safeParseJson(content);
        
        rankedResults.push({
          ...result,
          rerankedScore: parseFloat(parsed.relevanceScore) || result.score,
          relevanceExplanation: parsed.explanation || '',
        });
      } catch (error) {
        console.error('[SearchExecutor] 重排序失败:', error);
        rankedResults.push({
          ...result,
          rerankedScore: result.score,
          relevanceExplanation: '重排序失败',
        });
      }
    }

    return rankedResults
      .sort((a, b) => b.rerankedScore - a.rerankedScore)
      .slice(0, topK);
  }

  private buildFilterExpression(constraints: SearchConstraint[]): string {
    if (constraints.length === 0) return '';

    const expressions = constraints.map(c => {
      switch (c.operator) {
        case 'eq':
          return `${c.field} == "${c.value}"`;
        case 'contains':
          // Milvus 不支持 LIKE '%xxx%' 模式，只支持前缀匹配 'xxx%' 或精确匹配
          // 对于 content 字段使用前缀匹配，其他字段跳过（依赖向量语义搜索）
          if (c.field === 'content') {
            // 使用前缀匹配（Milvus 支持）
            return `${c.field} like "${c.value}%"`;
          }
          // 对于实体字段（person, organization 等），跳过 filter
          // 因为这些字段可能不存在于 Milvus schema 中
          // 依赖向量搜索的语义相似度来召回相关文档
          console.log(`[SearchExecutor] 跳过 contains 约束: ${c.field}="${c.value}" (依赖语义搜索)`);
          return '';
        case 'in':
          return `${c.field} in [${(c.value as string[]).map(v => `"${v}"`).join(', ')}]`;
        case 'not':
          return `${c.field} != "${c.value}"`;
        default:
          return '';
      }
    }).filter(e => e);

    return expressions.join(' && ');
  }

  private safeParseJson(content: string): any {
    try {
      return JSON.parse(content);
    } catch {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          return JSON.parse(jsonMatch[0]);
        } catch {
          return {};
        }
      }
      return {};
    }
  }
}

/**
 * 第四层：数据基础设施层 - 实体元数据存储
 */
export class EntityMetadataStore {
  private entities: Map<string, EntityMetadata> = new Map();
  private embeddings: OllamaEmbeddings;

  constructor(embeddingModel: string) {
    this.embeddings = new OllamaEmbeddings({ model: embeddingModel });
    // 初始化一些常见的同义词映射
    this.initializeDefaultMappings();
  }

  private initializeDefaultMappings() {
    // 地点同义词
    this.addEntity({
      standardName: '上海',
      type: 'LOCATION',
      aliases: ['魔都', 'Shanghai', '沪'],
      hierarchy: ['中国', '上海'],
    });
    this.addEntity({
      standardName: '北京',
      type: 'LOCATION',
      aliases: ['帝都', 'Beijing', '京'],
      hierarchy: ['中国', '北京'],
    });
    this.addEntity({
      standardName: '深圳',
      type: 'LOCATION',
      aliases: ['鹏城', 'Shenzhen'],
      hierarchy: ['中国', '广东', '深圳'],
    });

    // 公司同义词
    this.addEntity({
      standardName: 'Apple',
      type: 'ORGANIZATION',
      aliases: ['苹果', '苹果公司', 'Apple Inc.', 'AAPL'],
    });
    this.addEntity({
      standardName: 'Google',
      type: 'ORGANIZATION',
      aliases: ['谷歌', 'Alphabet', 'GOOG'],
    });
    this.addEntity({
      standardName: 'Microsoft',
      type: 'ORGANIZATION',
      aliases: ['微软', 'MS', 'MSFT'],
    });
    this.addEntity({
      standardName: 'Tesla',
      type: 'ORGANIZATION',
      aliases: ['特斯拉', 'TSLA'],
    });
    this.addEntity({
      standardName: 'SpaceX',
      type: 'ORGANIZATION',
      aliases: ['太空探索技术公司', 'Space Exploration Technologies Corp.'],
    });

    // 人物同义词
    this.addEntity({
      standardName: 'Elon Musk',
      type: 'PERSON',
      aliases: ['马斯克', '埃隆·马斯克', '老马', 'Musk'],
    });
    this.addEntity({
      standardName: 'Tim Cook',
      type: 'PERSON',
      aliases: ['库克', '蒂姆·库克'],
    });

    // 产品同义词
    this.addEntity({
      standardName: 'iPhone 15',
      type: 'PRODUCT',
      aliases: ['iPhone15', 'iPhone 15 Pro', 'iPhone 15 Pro Max'],
    });
    this.addEntity({
      standardName: 'ChatGPT',
      type: 'PRODUCT',
      aliases: ['GPT', 'GPT-4', 'GPT-4o', 'OpenAI GPT'],
    });
  }

  addEntity(metadata: EntityMetadata): void {
    this.entities.set(metadata.standardName.toLowerCase(), metadata);
  }

  async findSimilar(name: string, type: EntityType, topK: number = 5): Promise<EntityMetadata[]> {
    const candidates: EntityMetadata[] = [];
    const lowerName = name.toLowerCase();

    // 精确匹配
    if (this.entities.has(lowerName)) {
      candidates.push(this.entities.get(lowerName)!);
    }

    // 别名匹配
    for (const [, metadata] of this.entities) {
      if (metadata.type === type || type === 'OTHER') {
        if (metadata.aliases.some(a => a.toLowerCase() === lowerName)) {
          if (!candidates.find(c => c.standardName === metadata.standardName)) {
            candidates.push(metadata);
          }
        }
      }
    }

    // 模糊匹配（编辑距离）
    for (const [key, metadata] of this.entities) {
      if (candidates.length >= topK) break;
      if (candidates.find(c => c.standardName === metadata.standardName)) continue;

      const similarity = this.calculateSimilarity(lowerName, key);
      if (similarity > 0.6) {
        candidates.push(metadata);
      }
    }

    return candidates.slice(0, topK);
  }

  private calculateSimilarity(a: string, b: string): number {
    // 简单的 Jaccard 相似度
    const setA = new Set(a.split(''));
    const setB = new Set(b.split(''));
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    return intersection.size / union.size;
  }

  getAllEntities(): EntityMetadata[] {
    return Array.from(this.entities.values());
  }

  getEntitiesByType(type: EntityType): EntityMetadata[] {
    return Array.from(this.entities.values()).filter(e => e.type === type);
  }
}

/**
 * 响应生成器
 */
export class ResponseGenerator {
  private llm: ChatOllama;

  constructor(model: string) {
    this.llm = new ChatOllama({
      model,
      temperature: 0.7,
    });
  }

  async generate(
    query: ParsedQuery,
    results: RankedResult[]
  ): Promise<string> {
    if (results.length === 0) {
      return '抱歉，未能找到与您问题相关的信息。请尝试使用不同的关键词或更简洁的表述。';
    }

    const context = results
      .slice(0, 5)
      .map((r, i) => `[文档${i + 1}] (相关度: ${(r.rerankedScore * 100).toFixed(1)}%)\n${r.content}`)
      .join('\n\n---\n\n');

    const prompt = RESPONSE_GENERATION_PROMPT
      .replace('{query}', query.originalQuery)
      .replace('{intent}', query.intent)
      .replace('{entities}', query.entities.map(e => `${e.name}(${e.type})`).join(', '))
      .replace('{context}', context);

    try {
      const response = await this.llm.invoke(prompt);
      return typeof response.content === 'string' 
        ? response.content 
        : JSON.stringify(response.content);
    } catch (error) {
      console.error('[ResponseGenerator] 生成失败:', error);
      return '抱歉，生成回答时出现错误。请稍后重试。';
    }
  }
}

/**
 * 主控制器 - 自适应实体路由 RAG
 */
export class AdaptiveEntityRAG {
  private config: AdaptiveRAGConfig;
  private cognitiveParser: CognitiveParser;
  private strategyController: StrategyController;
  private searchExecutor: SearchExecutor;
  private entityMetadataStore: EntityMetadataStore;
  private responseGenerator: ResponseGenerator;

  constructor(config: Partial<AdaptiveRAGConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    this.entityMetadataStore = new EntityMetadataStore(this.config.embeddingModel);
    this.cognitiveParser = new CognitiveParser(this.config.llmModel);
    this.strategyController = new StrategyController(this.config, this.entityMetadataStore);
    this.searchExecutor = new SearchExecutor(this.config);
    this.responseGenerator = new ResponseGenerator(this.config.llmModel);
  }

  /**
   * 执行完整的 RAG 流程
   */
  async query(question: string, topK: number = 5): Promise<WorkflowState> {
    const startTime = Date.now();
    const steps: WorkflowStep[] = [];

    let state: WorkflowState = {
      query: {} as ParsedQuery,
      validatedEntities: [],
      currentDecision: {} as RoutingDecision,
      searchResults: [],
      rankedResults: [],
      finalResponse: '',
      steps: [],
      totalDuration: 0,
    };

    try {
      // Step 1: 认知解析
      const parseStep = this.createStep('认知解析 (实体提取)');
      steps.push(parseStep);
      parseStep.status = 'running';
      
      const parseStart = Date.now();
      state.query = await this.cognitiveParser.parse(question);
      parseStep.duration = Date.now() - parseStart;
      parseStep.status = 'completed';
      parseStep.details = {
        entities: state.query.entities.length,
        intent: state.query.intent,
        complexity: state.query.complexity,
      };

      // Step 2: 实体校验
      const validateStep = this.createStep('实体校验与归一化');
      steps.push(validateStep);
      validateStep.status = 'running';

      const validateStart = Date.now();
      state.validatedEntities = await this.strategyController.validateEntities(state.query.entities);
      validateStep.duration = Date.now() - validateStart;
      validateStep.status = 'completed';
      validateStep.details = {
        validated: state.validatedEntities.filter(e => e.isValid).length,
        total: state.validatedEntities.length,
      };

      // Step 3: 路由决策与检索循环
      let retryCount = 0;
      let results: SearchResult[] = [];

      while (retryCount <= this.config.maxRetries) {
        // 做出路由决策
        const routingStep = this.createStep(`路由决策 (尝试 ${retryCount + 1})`);
        steps.push(routingStep);
        routingStep.status = 'running';

        state.currentDecision = this.strategyController.makeRoutingDecision(
          state.query,
          state.validatedEntities,
          retryCount > 0 ? state.currentDecision : undefined,
          results.length
        );

        routingStep.status = 'completed';
        routingStep.details = {
          action: state.currentDecision.action,
          reason: state.currentDecision.reason,
          constraintCount: state.currentDecision.constraints.length,
        };

        // 如果决定生成响应，跳出循环
        if (state.currentDecision.action === 'generate_response') {
          break;
        }

        // 执行检索
        const searchStep = this.createStep(`执行${this.getSearchTypeName(state.currentDecision.action)}`);
        steps.push(searchStep);
        searchStep.status = 'running';

        const searchStart = Date.now();

        switch (state.currentDecision.action) {
          case 'structured_search':
            results = await this.searchExecutor.structuredSearch(
              question,
              state.currentDecision.constraints,
              topK * 2
            );
            break;
          case 'semantic_search':
            results = await this.searchExecutor.semanticSearch(question, topK * 2);
            break;
          case 'hybrid_search':
            results = await this.searchExecutor.hybridSearch(
              question,
              state.currentDecision.constraints,
              topK * 2
            );
            break;
          case 'relax_constraints':
            // 约束松弛后重新进行结构化检索
            results = await this.searchExecutor.structuredSearch(
              question,
              state.currentDecision.constraints,
              topK * 2
            );
            break;
        }

        searchStep.duration = Date.now() - searchStart;
        searchStep.status = 'completed';
        searchStep.details = {
          resultCount: results.length,
          matchType: state.currentDecision.action,
        };

        state.searchResults = results;

        // 如果有结果或者已经是语义检索，跳出循环
        if (results.length >= this.config.minResultCount || state.currentDecision.action === 'semantic_search') {
          break;
        }

        retryCount++;
      }

      // Step 4: 重排序
      if (state.searchResults.length > 0) {
        const rerankStep = this.createStep('混合重排序');
        steps.push(rerankStep);
        rerankStep.status = this.config.enableReranking ? 'running' : 'skipped';

        if (this.config.enableReranking) {
          const rerankStart = Date.now();
          state.rankedResults = await this.searchExecutor.rerank(state.searchResults, state.query, topK);
          rerankStep.duration = Date.now() - rerankStart;
          rerankStep.status = 'completed';
          rerankStep.details = {
            inputCount: state.searchResults.length,
            outputCount: state.rankedResults.length,
          };
        } else {
          state.rankedResults = state.searchResults.map(r => ({
            ...r,
            rerankedScore: r.score,
            relevanceExplanation: '未启用重排序',
          }));
        }
      }

      // Step 5: 生成响应
      const generateStep = this.createStep('生成响应');
      steps.push(generateStep);
      generateStep.status = 'running';

      const generateStart = Date.now();
      state.finalResponse = await this.responseGenerator.generate(state.query, state.rankedResults);
      generateStep.duration = Date.now() - generateStart;
      generateStep.status = 'completed';

    } catch (error) {
      console.error('[AdaptiveEntityRAG] 查询失败:', error);
      const errorStep = steps.find(s => s.status === 'running');
      if (errorStep) {
        errorStep.status = 'failed';
        errorStep.error = error instanceof Error ? error.message : String(error);
      }
      state.finalResponse = `处理查询时出错: ${error instanceof Error ? error.message : '未知错误'}`;
    }

    state.steps = steps;
    state.totalDuration = Date.now() - startTime;

    return state;
  }

  private createStep(name: string): WorkflowStep {
    return {
      step: name,
      status: 'pending',
    };
  }

  private getSearchTypeName(action: string): string {
    const names: Record<string, string> = {
      structured_search: '结构化检索',
      semantic_search: '语义检索',
      hybrid_search: '混合检索',
      relax_constraints: '松弛约束检索',
    };
    return names[action] || '检索';
  }

  /**
   * 获取实体元数据存储
   */
  getEntityMetadataStore(): EntityMetadataStore {
    return this.entityMetadataStore;
  }

  /**
   * 添加自定义实体映射
   */
  addEntityMapping(metadata: EntityMetadata): void {
    this.entityMetadataStore.addEntity(metadata);
  }
}

// 导出默认实例创建函数
export function createAdaptiveEntityRAG(config?: Partial<AdaptiveRAGConfig>): AdaptiveEntityRAG {
  return new AdaptiveEntityRAG(config);
}
