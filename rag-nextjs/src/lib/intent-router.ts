/**
 * 意图路由器 (Semantic Router)
 * 
 * 基于 LangGraph 的智能意图分类系统
 * 
 * 三条车道:
 * - Lane 1 (Fast Track): 闲聊/通用问题，0 IO，< 1秒
 * - Lane 2 (Standard RAG): 知识库问答，标准 RAG，3-5秒
 * - Lane 3 (Reasoning Agent): 复杂推理，深度分析，15-60秒
 */

import { StateGraph, Annotation, END, START } from '@langchain/langgraph';
import { Ollama } from '@langchain/ollama';

// ==================== 类型定义 ====================

/** 意图类型 */
export type IntentType = 'chat' | 'fast_rag' | 'reasoning';

/** 意图分类结果 */
export interface IntentClassification {
  intent: IntentType;
  confidence: number;
  reasoning: string;
  keywords: string[];
  complexity: 'low' | 'medium' | 'high';
  requiresRetrieval: boolean;
  requiresReasoning: boolean;
  suggestedLane: 1 | 2 | 3;
  estimatedTime: string;
}

/** 路由状态 */
export interface RouterState {
  query: string;
  classification: IntentClassification | null;
  routerModel: string;
  startTime: number;
  error?: string;
}

/** 路由配置 */
export interface RouterConfig {
  routerModel?: string;  // 用于路由的轻量级模型
  timeout?: number;      // 路由超时（毫秒）
}

// ==================== 意图分类提示词 ====================

const CLASSIFICATION_PROMPT = `你是一个智能意图分类器。分析用户查询并将其分类为以下三种类型之一：

## 分类标准

### 1. chat (闲聊/通用)
- 打招呼、问候: "你好", "早上好", "你是谁"
- 通用知识: "什么是AI", "今天星期几"
- 写作请求: "帮我写封邮件", "写一首诗"
- 不需要查询知识库的问题

### 2. fast_rag (快速知识库问答)
- 事实性问题: "A公司的2024年营收是多少？"
- 简单查询: "总结一下这份文档"
- 定义查询: "什么是RAG系统？"
- 答案直接在文档中，找到就能答

### 3. reasoning (复杂推理)
- 对比分析: "对比A公司和B公司过去三年的增长策略异同"
- 假设推演: "如果不考虑汇率影响，这个项目的实际收益如何？"
- 多源综合: "综合分析市场趋势和公司财报，给出投资建议"
- 需要逻辑推演、多步骤思考

## 用户查询
"{query}"

## 输出格式
请严格按照以下 JSON 格式输出（不要添加任何其他内容）:
{{
  "intent": "chat" 或 "fast_rag" 或 "reasoning",
  "confidence": 0.0到1.0之间的数字,
  "reasoning": "分类理由（简短）",
  "keywords": ["关键词1", "关键词2"],
  "complexity": "low" 或 "medium" 或 "high",
  "requiresRetrieval": true或false,
  "requiresReasoning": true或false
}}`;

// ==================== LangGraph 状态定义 ====================

const RouterAnnotation = Annotation.Root({
  query: Annotation<string>(),
  classification: Annotation<IntentClassification | null>({ default: () => null }),
  routerModel: Annotation<string>({ default: () => 'llama3.2' }),
  startTime: Annotation<number>({ default: () => Date.now() }),
  error: Annotation<string | undefined>()
});

// ==================== 路由节点 ====================

/**
 * 意图分类节点
 * 使用轻量级模型快速分类用户意图
 */
async function classifyIntentNode(
  state: typeof RouterAnnotation.State
): Promise<Partial<typeof RouterAnnotation.State>> {
  const nodeStartTime = Date.now();
  
  console.log(`\n${'='.repeat(60)}`);
  console.log(`[INTENT_ROUTER] 开始意图分类`);
  console.log(`[INTENT_ROUTER] 查询: "${state.query}"`);
  console.log(`[INTENT_ROUTER] 路由模型: ${state.routerModel}`);
  console.log(`${'='.repeat(60)}`);

  try {
    // 快速规则匹配（极速路径）
    const quickMatch = quickIntentMatch(state.query);
    if (quickMatch) {
      console.log(`[INTENT_ROUTER] 快速匹配成功: ${quickMatch.intent}`);
      console.log(`[INTENT_ROUTER] 耗时: ${Date.now() - nodeStartTime}ms`);
      return { classification: quickMatch };
    }

    // 使用 LLM 进行深度分类
    const llm = new Ollama({
      model: state.routerModel,
      baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
    });

    const prompt = CLASSIFICATION_PROMPT.replace('{query}', escapeBraces(state.query));
    const response = await llm.invoke(prompt);

    // 解析 JSON 响应
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('无法解析分类结果');
    }

    const parsed = JSON.parse(jsonMatch[0]);
    
    // 构建分类结果
    const classification: IntentClassification = {
      intent: parsed.intent || 'fast_rag',
      confidence: parsed.confidence || 0.7,
      reasoning: parsed.reasoning || '默认分类',
      keywords: parsed.keywords || [],
      complexity: parsed.complexity || 'medium',
      requiresRetrieval: parsed.requiresRetrieval !== false,
      requiresReasoning: parsed.requiresReasoning || false,
      suggestedLane: getSuggestedLane(parsed.intent),
      estimatedTime: getEstimatedTime(parsed.intent)
    };

    const duration = Date.now() - nodeStartTime;
    console.log(`[INTENT_ROUTER] 分类完成:`);
    console.log(`  - 意图: ${classification.intent}`);
    console.log(`  - 置信度: ${(classification.confidence * 100).toFixed(0)}%`);
    console.log(`  - 车道: Lane ${classification.suggestedLane}`);
    console.log(`  - 预计耗时: ${classification.estimatedTime}`);
    console.log(`[INTENT_ROUTER] 路由耗时: ${duration}ms`);

    return { classification };

  } catch (error) {
    console.error('[INTENT_ROUTER] 分类错误:', error);
    
    // 降级到默认分类
    const fallbackClassification: IntentClassification = {
      intent: 'fast_rag',
      confidence: 0.5,
      reasoning: '分类失败，降级到标准 RAG',
      keywords: [],
      complexity: 'medium',
      requiresRetrieval: true,
      requiresReasoning: false,
      suggestedLane: 2,
      estimatedTime: '3-5秒'
    };

    return { 
      classification: fallbackClassification,
      error: error instanceof Error ? error.message : '分类失败'
    };
  }
}

// ==================== 辅助函数 ====================

/**
 * 快速规则匹配（无需 LLM）
 * 用于极速识别明显的意图类型
 */
function quickIntentMatch(query: string): IntentClassification | null {
  const q = query.toLowerCase().trim();
  
  // 闲聊模式 - 极速匹配
  const chatPatterns = [
    /^(你好|您好|hi|hello|hey|嗨|哈喽)/i,
    /^(早上好|下午好|晚上好|早安|晚安)/i,
    /^(你是谁|你叫什么|介绍一下你自己)/i,
    /^(谢谢|感谢|多谢)/i,
    /^(再见|拜拜|bye|goodbye)/i,
    /^(帮我写|写一个|写一篇|写一封)/i,
    /^(讲个笑话|说个笑话|来个笑话)/i,
  ];

  for (const pattern of chatPatterns) {
    if (pattern.test(q)) {
      return {
        intent: 'chat',
        confidence: 0.95,
        reasoning: '规则匹配: 闲聊/通用请求',
        keywords: [],
        complexity: 'low',
        requiresRetrieval: false,
        requiresReasoning: false,
        suggestedLane: 1,
        estimatedTime: '< 1秒'
      };
    }
  }

  // 复杂推理模式 - 关键词匹配
  const reasoningKeywords = [
    '对比', '比较', '分析', '综合', '推断', '推理',
    '如果', '假设', '假如', '倘若',
    '为什么会', '原因是什么', '背后的逻辑',
    '趋势', '预测', '评估', '建议',
    '异同', '优劣', '利弊'
  ];

  const hasReasoningKeyword = reasoningKeywords.some(kw => q.includes(kw));
  const isLongQuery = q.length > 50;
  const hasMultipleQuestions = (q.match(/？|\?/g) || []).length > 1;

  if (hasReasoningKeyword && (isLongQuery || hasMultipleQuestions)) {
    return {
      intent: 'reasoning',
      confidence: 0.85,
      reasoning: '规则匹配: 包含推理关键词且问题复杂',
      keywords: reasoningKeywords.filter(kw => q.includes(kw)),
      complexity: 'high',
      requiresRetrieval: true,
      requiresReasoning: true,
      suggestedLane: 3,
      estimatedTime: '15-60秒'
    };
  }

  // 无法快速匹配，需要 LLM 判断
  return null;
}

/**
 * 获取建议的车道
 */
function getSuggestedLane(intent: IntentType): 1 | 2 | 3 {
  switch (intent) {
    case 'chat': return 1;
    case 'fast_rag': return 2;
    case 'reasoning': return 3;
    default: return 2;
  }
}

/**
 * 获取预计耗时
 */
function getEstimatedTime(intent: IntentType): string {
  switch (intent) {
    case 'chat': return '< 1秒';
    case 'fast_rag': return '3-5秒';
    case 'reasoning': return '15-60秒';
    default: return '3-5秒';
  }
}

/**
 * 转义大括号
 */
function escapeBraces(text: string): string {
  return text.replace(/\{/g, '{{').replace(/\}/g, '}}');
}

// ==================== 构建路由图 ====================

/**
 * 构建意图路由图
 */
export function buildIntentRouterGraph() {
  const workflow = new StateGraph(RouterAnnotation)
    .addNode('classify', classifyIntentNode)
    .addEdge(START, 'classify')
    .addEdge('classify', END);

  return workflow.compile();
}

// ==================== 主执行函数 ====================

/**
 * 执行意图路由
 */
export async function routeIntent(
  query: string,
  config?: RouterConfig
): Promise<IntentClassification> {
  const startTime = Date.now();
  
  const initialState: Partial<typeof RouterAnnotation.State> = {
    query,
    routerModel: config?.routerModel || 'llama3.2',
    startTime
  };

  try {
    const graph = buildIntentRouterGraph();
    const result = await graph.invoke(initialState);

    const routingTime = Date.now() - startTime;
    console.log(`\n[INTENT_ROUTER] 总路由耗时: ${routingTime}ms\n`);

    if (result.classification) {
      return result.classification;
    }

    // 默认返回 fast_rag
    return {
      intent: 'fast_rag',
      confidence: 0.5,
      reasoning: '默认分类',
      keywords: [],
      complexity: 'medium',
      requiresRetrieval: true,
      requiresReasoning: false,
      suggestedLane: 2,
      estimatedTime: '3-5秒'
    };

  } catch (error) {
    console.error('[INTENT_ROUTER] 执行错误:', error);
    
    return {
      intent: 'fast_rag',
      confidence: 0.5,
      reasoning: '路由失败，降级到标准 RAG',
      keywords: [],
      complexity: 'medium',
      requiresRetrieval: true,
      requiresReasoning: false,
      suggestedLane: 2,
      estimatedTime: '3-5秒'
    };
  }
}

// ==================== 车道处理器类型 ====================

export interface LaneHandler {
  lane: 1 | 2 | 3;
  name: string;
  description: string;
  execute: (query: string, config: any) => AsyncGenerator<any, void, unknown>;
}

export interface LaneResult {
  lane: 1 | 2 | 3;
  laneName: string;
  answer: string;
  thinkingProcess?: any[];
  retrievalStats?: any;
  totalDuration: number;
}
