import { LocalRAGSystem } from './rag-system';

// 全局 RAG 系统实例
let ragSystemInstance: LocalRAGSystem | null = null;

export async function getRagSystem(): Promise<LocalRAGSystem> {
  if (!ragSystemInstance) {
    ragSystemInstance = new LocalRAGSystem({
      ollamaBaseUrl: "http://localhost:11434",
      llmModel: "llama3.1",
      embeddingModel: "nomic-embed-text",
    });

    // 初始化数据库
    await ragSystemInstance.initializeDatabase();
  }

  return ragSystemInstance;
}

export function resetRagSystem() {
  ragSystemInstance = null;
}