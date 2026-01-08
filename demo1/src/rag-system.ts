import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import fs from "fs";
import path from "path";

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
  
  getDocumentCount(): number {
    return this.documents.length;
  }
  
  clear() {
    this.documents = [];
    this.embeddings = [];
  }
}

export class LocalRAGSystem {
  private llm: Ollama;
  private embeddings: OllamaEmbeddings;
  private vectorStore!: SimpleMemoryVectorStore;
  private isInitialized = false;

  constructor(
    private config: {
      ollamaBaseUrl?: string;
      llmModel?: string;
      embeddingModel?: string;
      docsPath?: string;
    } = {}
  ) {
    const {
      ollamaBaseUrl = "http://localhost:11434",
      llmModel = "llama3.1",
      embeddingModel = "nomic-embed-text",
    } = config;

    this.llm = new Ollama({
      baseUrl: ollamaBaseUrl,
      model: llmModel,
      temperature: 0,
    });

    this.embeddings = new OllamaEmbeddings({
      baseUrl: ollamaBaseUrl,
      model: embeddingModel,
    });

    this.vectorStore = new SimpleMemoryVectorStore(this.embeddings);
  }

  /**
   * 初始化数据库，加载文档
   */
  async initializeDatabase(docsPath: string = "./data"): Promise<void> {
    console.log("--- 正在初始化向量数据库 ---");

    // 检查数据目录是否存在
    if (!fs.existsSync(docsPath)) {
      console.log(`数据目录 ${docsPath} 不存在，创建示例文档...`);
      fs.mkdirSync(docsPath, { recursive: true });
      
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

系统功能：
- 文档上传和处理
- 智能问答
- 相似度搜索
- Web 界面交互
      `;
      
      fs.writeFileSync(path.join(docsPath, "sample.txt"), sampleDoc, "utf8");
      console.log("已创建示例文档");
    }

    await this.loadDocumentsFromDirectory(docsPath);
    this.isInitialized = true;
    console.log("向量数据库初始化完成。");
  }

  /**
   * 从目录加载文档
   */
  async loadDocumentsFromDirectory(docsPath: string): Promise<void> {
    const documents: Document[] = [];
    
    // 加载文本文件
    const files = fs.readdirSync(docsPath).filter(file => file.endsWith('.txt'));
    
    for (const file of files) {
      const filePath = path.join(docsPath, file);
      const content = fs.readFileSync(filePath, "utf8");
      documents.push(new Document({
        pageContent: content,
        metadata: { source: file, path: filePath }
      }));
    }

    console.log(`已加载 ${documents.length} 个文档`);

    if (documents.length > 0) {
      // 文本切分
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 600,
        chunkOverlap: 100,
      });

      const splitDocs = await splitter.splitDocuments(documents);
      console.log(`切分为 ${splitDocs.length} 个文本块`);

      // 添加到向量存储
      await this.vectorStore.addDocuments(splitDocs);
    }
  }

  /**
   * 添加单个文档
   */
  async addDocument(content: string, metadata: any = {}): Promise<void> {
    const doc = new Document({
      pageContent: content,
      metadata
    });

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600,
      chunkOverlap: 100,
    });

    const splitDocs = await splitter.splitDocuments([doc]);
    await this.vectorStore.addDocuments(splitDocs);
  }

  /**
   * 保存上传的文件到数据目录
   */
  async saveUploadedFile(filename: string, content: string, docsPath: string = "./data"): Promise<void> {
    if (!fs.existsSync(docsPath)) {
      fs.mkdirSync(docsPath, { recursive: true });
    }
    
    const filePath = path.join(docsPath, filename);
    fs.writeFileSync(filePath, content, "utf8");
    
    // 重新加载文档
    await this.addDocument(content, { source: filename, path: filePath });
  }

  /**
   * 执行问答
   */
  async ask(question: string): Promise<string> {
    if (!this.isInitialized) {
      throw new Error("RAG 系统尚未初始化，请先调用 initializeDatabase()");
    }

    console.log(`用户问题: ${question}`);

    // 1. 检索相似内容
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
      请使用中文回答，回答要简洁明了。
    `);

    // 3. 运行链式调用
    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());

    const result = await chain.invoke({
      context: context,
      question: question,
    });

    console.log(`AI 回答: \n${result}\n`);
    return result;
  }

  /**
   * 获取系统状态
   */
  getStatus(): { initialized: boolean; documentCount: number } {
    return {
      initialized: this.isInitialized,
      documentCount: this.vectorStore.getDocumentCount()
    };
  }

  /**
   * 清空向量存储
   */
  clearDatabase(): void {
    this.vectorStore.clear();
    this.isInitialized = false;
  }
}