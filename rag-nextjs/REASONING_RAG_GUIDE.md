# Reasoning RAG 架构设计指南

基于推理模型 (DeepSeek-R1/Qwen3) 的高级 RAG 系统完整架构设计。

## 📋 目录

1. [系统概述](#系统概述)
2. [三层架构](#三层架构)
3. [核心组件](#核心组件)
4. [API 参考](#api-参考)
5. [使用示例](#使用示例)
6. [与其他 RAG 模式对比](#与其他-rag-模式对比)

---

## 系统概述

Reasoning RAG 是一个专为推理模型设计的高级检索增强生成系统，支持：

- **思维链可视化** - 展示模型的完整推理过程
- **混合检索** - Dense + BM25 双路召回
- **深度重排序** - LLM 相关性精排
- **智能编排** - 意图识别与工具调用

### 支持的推理模型

| 模型 | 参数量 | 上下文长度 | 特性 |
|------|--------|-----------|------|
| DeepSeek-R1 7B | 7B | 32K | 支持 `<think>` 标签 |
| DeepSeek-R1 14B | 14B | 32K | 更强推理能力 |
| DeepSeek-R1 32B | 32B | 64K | 顶级推理能力 |
| Qwen3 8B | 8B | 32K | 中文优化 |
| Qwen3 14B | 14B | 32K | 更强中文推理 |

---

## 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Graph State (状态层)                      │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │    Messages     │  │          Scratchpad             │   │
│  │  (对话历史)      │  │        (思维链片段)              │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Cognitive Layer (认知层)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Reasoning Orchestrator                  │   │
│  │                    (编排器)                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ 意图识别 │→│ 决策分析 │→│ 输出: Tool/Generate│   │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Tool Execution Layer (执行层)                 │
│  ┌─────────────┐  ┌─────────────────────────────────────┐   │
│  │Tool Gateway │→│        Hybrid Retrieval              │   │
│  │ (安全检查)   │  │  ┌─────────┐    ┌──────────┐       │   │
│  └─────────────┘  │  │  Dense  │ +  │  BM25   │        │   │
│                   │  │ (Milvus)│    │ (Sparse)│        │   │
│                   │  └─────────┘    └──────────┘       │   │
│                   └─────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Reranker                          │   │
│  │  Top-50 + Top-50 → LLM Score → Top-5                │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Formatter                          │   │
│  │  清洗 → 去噪 → XML/Markdown 格式化                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Generator (生成器)                      │
│        推理模型 + 格式化上下文 → 最终回答                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. Graph State (全局精细化状态)

```typescript
interface ReasoningRAGState {
  // OpenAI 标准格式消息
  messages: BaseMessage[];
  
  // 思维链存储 (用于调试和审计)
  scratchpad: ThinkingStep[];
  
  // 用户原始输入
  originalQuery: string;
  
  // 配置
  config: {
    reasoningModel: string;      // 推理模型
    embeddingModel: string;      // 嵌入模型
    topK: number;                // 检索数量
    rerankTopK: number;          // 重排后保留数量
    enableBM25: boolean;         // 启用 BM25
    enableRerank: boolean;       // 启用重排序
  };
  
  // 编排器决策
  orchestratorDecision?: OrchestratorDecision;
  
  // 检索结果
  retrievalResult?: HybridRetrievalResult;
  
  // 最终答案
  finalAnswer: string;
}
```

### 2. Cognitive Layer (认知层)

**Orchestrator (编排器)** 是系统的大脑：

```typescript
interface OrchestratorDecision {
  action: 'tool_call' | 'generate' | 'clarify';
  intent: string;           // 意图类型
  confidence: number;       // 置信度
  reasoning: string;        // 推理过程
  toolCalls?: ToolCall[];   // 工具调用
}
```

工作流程：
1. **意图识别** - 分析用户问题类型
2. **决策分析** - 判断是否需要检索
3. **输出分支**：
   - `tool_call` → 调用检索工具
   - `generate` → 直接生成回答
   - `clarify` → 需要用户澄清

### 3. Tool Execution Layer (执行层)

#### Tool Gateway (工具网关)

- 拦截工具调用
- 参数验证
- 安全检查 (防 SQL 注入等)

#### Hybrid Retrieval (混合检索)

| 检索方式 | 引擎 | 优势 |
|---------|------|------|
| Dense | Milvus | 语义理解 ("苹果" → "水果") |
| Sparse | BM25 | 关键词精确匹配 ("iPhone 16 Pro") |

**RRF 融合算法**：
```
score = Σ 1/(k + rank) * weight
Dense weight: 0.6
Sparse weight: 0.4
```

#### Reranker (重排序)

1. Dense Top-50 + BM25 Top-50 = 100 候选
2. LLM 深度评分
3. 保留 Top-5 高质量结果

#### Formatter (格式化器)

```xml
<retrieved_documents>
  <document id="1" score="0.95" source="hybrid">
    <content>清洗后的文档内容...</content>
    <metadata>{"source": "file.pdf"}</metadata>
  </document>
</retrieved_documents>
```

---

## API 参考

### POST /api/reasoning-rag

执行 Reasoning RAG 查询。

**请求体**：

```json
{
  "query": "什么是 RAG 系统？",
  "config": {
    "reasoningModel": "deepseek-r1:7b",
    "embeddingModel": "nomic-embed-text",
    "topK": 50,
    "rerankTopK": 5,
    "enableBM25": true,
    "enableRerank": true,
    "temperature": 0.7
  }
}
```

**响应**：

```json
{
  "success": true,
  "data": {
    "query": "什么是 RAG 系统？",
    "answer": "RAG (Retrieval-Augmented Generation) 是...",
    "thinkingProcess": [
      {
        "id": "think-1",
        "type": "reasoning",
        "content": "用户在询问技术概念...",
        "confidence": 0.9
      }
    ],
    "retrieval": {
      "statistics": {
        "denseCount": 50,
        "sparseCount": 35,
        "finalCount": 5,
        "totalTime": 1250
      }
    },
    "workflow": {
      "totalDuration": 3500,
      "decisionPath": ["orchestrator:tool_call", "hybrid_retrieval:50_docs", "reranker:5_docs"]
    }
  }
}
```

### GET /api/reasoning-rag?action=models

获取可用的推理模型列表。

---

## 使用示例

### 基本使用

```typescript
const response = await fetch('/api/reasoning-rag', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: '如何优化向量检索的准确性？',
    config: {
      reasoningModel: 'deepseek-r1:7b',
      enableBM25: true,
      enableRerank: true
    }
  })
});

const { data } = await response.json();
console.log('回答:', data.answer);
console.log('思维链:', data.thinkingProcess);
```

### 在 React 中使用

```tsx
import ReasoningRAGVisualizer from '@/components/ReasoningRAGVisualizer';

function MyComponent() {
  const [response, setResponse] = useState(null);
  
  return (
    <ReasoningRAGVisualizer
      query={response?.query}
      answer={response?.answer}
      thinkingProcess={response?.thinkingProcess}
      retrieval={response?.retrieval}
      workflow={response?.workflow}
    />
  );
}
```

---

## 与其他 RAG 模式对比

| 特性 | Reasoning RAG | Agentic RAG | Self-Corrective RAG |
|------|--------------|-------------|---------------------|
| **核心模型** | 推理模型 (R1/Qwen3) | 通用 LLM | 通用 LLM |
| **思维链** | ✅ 完整展示 | ❌ | ❌ |
| **混合检索** | ✅ Dense + BM25 | Dense Only | Dense Only |
| **重排序** | ✅ LLM Rerank | ❌ | ❌ |
| **智能编排** | ✅ Orchestrator | ✅ 多节点 | ❌ 固定流程 |
| **质量闭环** | ❌ | ✅ 幻觉检查 | ✅ Grader 循环 |
| **适用场景** | 复杂推理问题 | 多轮对话 | 检索质量敏感 |

### 选择建议

- **Reasoning RAG**: 需要深度推理、需要看到思考过程
- **Agentic RAG**: 需要多轮交互、复杂工作流
- **Self-Corrective RAG**: 检索质量要求高、需要自动修正

---

## 性能优化建议

1. **BM25 索引优化**: 对于大量文档，建议使用专业的 BM25 引擎
2. **Rerank 批处理**: 批量评估可显著降低延迟
3. **模型选择**: 7B 模型适合实时应用，32B 适合离线分析
4. **缓存策略**: 对常见查询缓存检索结果

---

## 更新日志

- **v1.0.0** (2026-01-20): 初始版本
  - 三层架构设计
  - 混合检索 (Dense + BM25)
  - LLM 重排序
  - 思维链可视化
