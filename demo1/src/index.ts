import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import fs from "fs";
import path from "path";

// 1. 配置常量
const OLLAMA_BASE_URL = "http://localhost:11434";
const LLM_MODEL = "llama3.1";
const EMBEDDING_MODEL = "nomic-embed-text";
const DOCS_PATH = "./data";        // 存放知识库文件的文件夹

// 简单的内存向量存储实现
class SimpleMemoryVectorStore {
  private documents: Document[] = [];
  private embeddings: number[][] = [];
  
  constructor(private embeddingModel: OllamaEmbeddings) {}
  
  async addDocuments(docs: Document[]) {
    this.documents.push(...docs);
    
    // 为每个文档生成嵌入
    for (const doc of docs) {
      const embedding = await this.embeddingModel.embedQuery(doc.pageContent);
      this.embeddings.push(embedding);
    }
  }
  
  async similaritySearch(query: string, k: number = 3): Promise<Document[]> {
    if (this.documents.length === 0) {
      return [];
    }
    
    // 生成查询的嵌入
    const queryEmbedding = await this.embeddingModel.embedQuery(query);
    
    // 计算余弦相似度
    const similarities = this.embeddings.map((embedding, index) => ({
      index,
      similarity: this.cosineSimilarity(queryEmbedding, embedding)
    }));
    
    // 按相似度排序并返回前k个
    similarities.sort((a, b) => b.similarity - a.similarity);
    
    return similarities.slice(0, k).map(item => this.documents[item.index]);
  }
  
  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }
}

class LocalRAGSystem {
  private llm: Ollama;
  private embeddings: OllamaEmbeddings;
  private vectorStore!: SimpleMemoryVectorStore;

  constructor() {
    this.llm = new Ollama({
      baseUrl: OLLAMA_BASE_URL,
      model: LLM_MODEL,
      temperature: 0, // 设置为0保证回答的稳定性
    });

    this.embeddings = new OllamaEmbeddings({
      baseUrl: OLLAMA_BASE_URL,
      model: EMBEDDING_MODEL,
    });
  }

  /**
   * 第一步：加载并处理文档，存入向量数据库
   */
  async initializeDatabase() {
    console.log("--- 正在初始化向量数据库 ---");

    // 简化的文档加载 - 从文本文件加载
    const documents: Document[] = [];
    
    // 检查数据目录是否存在
    if (!fs.existsSync(DOCS_PATH)) {
      console.log(`数据目录 ${DOCS_PATH} 不存在，创建示例文档...`);
      fs.mkdirSync(DOCS_PATH, { recursive: true });
      
      // 创建示例文档
      const sampleDoc = `
这是一个示例知识库文档。

核心技术架构：
- 使用 LangChain 构建 RAG 系统
- Ollama 作为本地 LLM 服务
- 向量数据库存储文档嵌入
- 支持多种文档格式

项目负责人：
- 姓名：张三
- 邮箱：zhangsan@example.com
- 电话：138-0000-0000

技术特点：
1. 本地部署，数据安全
2. 支持中文问答
3. 可扩展的文档格式支持
4. 高效的向量检索
      `;
      
      fs.writeFileSync(path.join(DOCS_PATH, "sample.txt"), sampleDoc, "utf8");
      console.log("已创建示例文档");
    }

    // 加载文本文件
    const files = fs.readdirSync(DOCS_PATH).filter(file => file.endsWith('.txt'));
    
    for (const file of files) {
      const filePath = path.join(DOCS_PATH, file);
      const content = fs.readFileSync(filePath, "utf8");
      documents.push(new Document({
        pageContent: content,
        metadata: { source: file }
      }));
    }

    console.log(`已加载 ${documents.length} 个文档`);

    // 文本切分
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600,
      chunkOverlap: 100,
    });

    const splitDocs = await splitter.splitDocuments(documents);
    console.log(`切分为 ${splitDocs.length} 个文本块`);

    // 存储到内存向量数据库
    this.vectorStore = new SimpleMemoryVectorStore(this.embeddings);
    await this.vectorStore.addDocuments(splitDocs);

    console.log("向量数据库初始化完成。");
  }

  /**
   * 第二步：执行问答
   */
  async ask(question: string) {
    console.log(`\n用户问题: ${question}`);

    // 1. 检索相似内容
    // k: 3 表示获取最相关的3个片段
    const relevantDocs = await this.vectorStore.similaritySearch(question, 3);
    const context = relevantDocs.map((d) => d.pageContent).join("\n---\n");

    // 2. 构造 Prompt 模板
    const prompt = ChatPromptTemplate.fromTemplate(`
      你是一个专业的知识库助手。请根据下方提供的上下文信息来回答用户的问题。
      
      【上下文信息】：
      {context}
      
      【用户问题】：
      {question}
      
      如果上下文信息中不包含答案，请礼貌地说明你不知道，不要胡乱编造。
      请使用中文回答。
    `);

    // 3. 运行链式调用
    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());

    const result = await chain.invoke({
      context: context,
      question: question,
    });

    console.log(`AI 回答: \n${result}\n`);
  }
}

// --- 运行示例 ---
async function main() {
  const rag = new LocalRAGSystem();
  
  // 如果是第一次运行，执行初始化逻辑
  await rag.initializeDatabase();

  // 模拟提问
  await rag.ask("这份文档里提到的核心技术架构是什么？");
  await rag.ask("项目负责人的联系方式是多少？");
}

main().catch(console.error);