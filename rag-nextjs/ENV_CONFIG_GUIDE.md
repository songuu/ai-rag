# 环境变量配置指南

## 快速开始

创建 `.env.local` 文件并配置以下环境变量：

```bash
# 模型提供商选择 (必填)
MODEL_PROVIDER=ollama  # 可选: ollama | openai | azure | custom
```

## 主开关

`MODEL_PROVIDER` 环境变量控制系统使用本地 Ollama 还是生产 API：

| 值 | 说明 |
|---|---|
| `ollama` | 使用本地 Ollama 服务 (默认) |
| `openai` | 使用 OpenAI API |
| `azure` | 使用 Azure OpenAI 服务 |
| `custom` | 使用自定义 OpenAI 兼容 API (如 DeepSeek, 智谱等) |

## Ollama 配置 (本地模式)

当 `MODEL_PROVIDER=ollama` 时使用：

```bash
# Ollama 服务地址
OLLAMA_BASE_URL=http://localhost:11434

# LLM 模型 (对话/生成)
# 推荐: llama3.1, qwen2.5, glm-4
OLLAMA_LLM_MODEL=llama3.1

# Embedding 模型 (向量嵌入)
# 推荐: nomic-embed-text (768维), bge-m3 (1024维)
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# 推理模型 (复杂推理任务)
# 推荐: deepseek-r1, qwen3
OLLAMA_REASONING_MODEL=deepseek-r1
```

## OpenAI 配置

当 `MODEL_PROVIDER=openai` 时使用：

```bash
# API Key (必填)
OPENAI_API_KEY=sk-xxxxx

# API 基础地址 (可选，用于代理)
OPENAI_BASE_URL=https://api.openai.com/v1

# LLM 模型
OPENAI_LLM_MODEL=gpt-4o-mini

# Embedding 模型
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# 推理模型
OPENAI_REASONING_MODEL=gpt-4o
```

## Azure OpenAI 配置

当 `MODEL_PROVIDER=azure` 时使用：

```bash
# API Key (必填)
AZURE_OPENAI_API_KEY=xxxxx

# 终端地址 (必填)
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com

# 部署名称
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## 自定义 API 配置

当 `MODEL_PROVIDER=custom` 时使用，支持 OpenAI 兼容的第三方 API：

```bash
# API Key (必填)
CUSTOM_API_KEY=sk-xxxxx

# API 基础地址 (必填)
# DeepSeek: https://api.deepseek.com
# 智谱: https://open.bigmodel.cn/api/paas/v4
# 月之暗面: https://api.moonshot.cn/v1
CUSTOM_BASE_URL=https://api.deepseek.com

# 模型名称
CUSTOM_LLM_MODEL=deepseek-chat
CUSTOM_EMBEDDING_MODEL=
```

## Milvus 配置

Milvus 向量数据库支持两种部署模式：

### 主开关

```bash
# Milvus 提供商选择
MILVUS_PROVIDER=local  # 可选: local | zilliz
```

| 值 | 说明 |
|---|---|
| `local` | 使用本地自建 Milvus 服务 (默认) |
| `zilliz` | 使用 Zilliz Cloud 托管服务 |

### 本地 Milvus 配置 (local)

当 `MILVUS_PROVIDER=local` 时使用：

```bash
# Milvus 服务地址
MILVUS_LOCAL_ADDRESS=localhost:19530

# 认证（可选）
MILVUS_LOCAL_USERNAME=
MILVUS_LOCAL_PASSWORD=
```

### Zilliz Cloud 配置 (zilliz)

当 `MILVUS_PROVIDER=zilliz` 时使用：

```bash
# Zilliz Cloud 集群端点 (必填)
# ⚠️ 注意：不需要 https:// 前缀，SDK 会自动处理
# 格式: in01-xxx.api.region.zillizcloud.com:443
# 从 Zilliz Cloud 控制台 -> 集群 -> Connect -> Node.js 获取
MILVUS_ZILLIZ_ENDPOINT=in01-xxx.api.gcp-us-west1.zillizcloud.com:443

# API Token (必填)
# 从 Zilliz Cloud 控制台 -> API Keys 获取
MILVUS_ZILLIZ_TOKEN=your_api_key_here

# 是否为 Serverless 实例
MILVUS_ZILLIZ_SERVERLESS=false
```

### 通用默认配置

```bash
# 默认数据库
MILVUS_DEFAULT_DATABASE=default

# 默认集合名称
MILVUS_DEFAULT_COLLECTION=rag_documents

# 默认向量维度
MILVUS_DEFAULT_DIMENSION=768

# 默认索引类型
MILVUS_DEFAULT_INDEX_TYPE=IVF_FLAT

# 默认距离度量
MILVUS_DEFAULT_METRIC_TYPE=COSINE
```

## 完整配置示例

### 示例 1: 本地开发 (Ollama + 本地 Milvus)

```bash
# 模型配置
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_REASONING_MODEL=deepseek-r1

# Milvus 配置
MILVUS_PROVIDER=local
MILVUS_LOCAL_ADDRESS=localhost:19530
```

### 示例 2: 生产环境 (OpenAI + Zilliz Cloud)

```bash
# 模型配置
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxx
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_REASONING_MODEL=gpt-4o

# Milvus 配置 - 使用 Zilliz Cloud
MILVUS_PROVIDER=zilliz
MILVUS_ZILLIZ_ENDPOINT=https://in01-xxx.aws-us-west-2.vectordb.zillizcloud.com:19530
MILVUS_ZILLIZ_TOKEN=xxxxx
MILVUS_DEFAULT_DIMENSION=1536  # OpenAI embedding 维度
```

### 示例 3: 使用 DeepSeek API

```bash
MODEL_PROVIDER=custom
CUSTOM_API_KEY=sk-xxxxx
CUSTOM_BASE_URL=https://api.deepseek.com
CUSTOM_LLM_MODEL=deepseek-chat
OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # DeepSeek 不提供 Embedding，仍用 Ollama

# Milvus 配置
MILVUS_PROVIDER=local
MILVUS_LOCAL_ADDRESS=localhost:19530
```

### 示例 4: 混合模式 (本地 Ollama + Zilliz Cloud)

```bash
# 模型配置 - 使用本地 Ollama
MODEL_PROVIDER=ollama
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Milvus 配置 - 使用云端 Zilliz
MILVUS_PROVIDER=zilliz
MILVUS_ZILLIZ_ENDPOINT=https://in01-xxx.vectordb.zillizcloud.com:19530
MILVUS_ZILLIZ_TOKEN=xxxxx
```

## 模型维度参考

| 模型 | 维度 | 提供商 |
|------|------|--------|
| nomic-embed-text | 768 | Ollama |
| bge-m3 | 1024 | Ollama |
| all-minilm | 384 | Ollama |
| qwen3-embedding | 1024 | Ollama |
| text-embedding-3-small | 1536 | OpenAI |
| text-embedding-3-large | 3072 | OpenAI |
| text-embedding-ada-002 | 1536 | OpenAI |

## API 使用

### 获取当前配置

```typescript
import { getModelFactory, getConfigSummary } from '@/lib/model-config';

// 获取配置摘要
const summary = getConfigSummary();
console.log(summary);
// { provider: 'ollama', llmModel: 'llama3.1', ... }
```

### 创建模型实例

```typescript
import { createLLM, createEmbedding, createReasoningModel } from '@/lib/model-config';

// 创建 LLM (会根据 MODEL_PROVIDER 自动选择)
const llm = createLLM();

// 创建指定模型的 LLM
const llm2 = createLLM('gpt-4o');

// 创建 Embedding 模型
const embedding = createEmbedding();

// 创建推理模型
const reasoning = createReasoningModel();
```

### 动态注册模型

```typescript
import { getModelFactory } from '@/lib/model-config';

const factory = getModelFactory();

// 注册自定义模型
factory.registerModel({
  id: 'my-custom-model',
  type: 'llm',
  config: {
    provider: 'custom',
    modelName: 'my-model',
    apiKey: 'xxx',
    baseUrl: 'https://api.example.com',
  },
  description: '我的自定义模型',
});

// 获取所有注册的模型
const models = factory.getRegisteredModels();
```

### 验证配置

```typescript
import { getModelFactory } from '@/lib/model-config';

const factory = getModelFactory();
const validation = factory.validateConfig();

if (!validation.valid) {
  console.error('配置错误:', validation.errors);
}
```
