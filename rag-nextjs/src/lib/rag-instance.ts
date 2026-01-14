import { LocalRAGSystem } from './rag-system';

// 使用 globalThis 来确保在 Next.js 热重载时保持单例
// 这是 Next.js 推荐的方式来保持服务器端的单例
declare global {
  // eslint-disable-next-line no-var
  var ragSystemInstance: LocalRAGSystem | undefined;
  // eslint-disable-next-line no-var
  var ragSystemInitPromise: Promise<LocalRAGSystem> | undefined;
}

export async function getRagSystem(): Promise<LocalRAGSystem> {
  // 如果已经有初始化好的实例，直接返回
  if (globalThis.ragSystemInstance) {
    return globalThis.ragSystemInstance;
  }

  // 如果正在初始化，等待初始化完成
  if (globalThis.ragSystemInitPromise) {
    return globalThis.ragSystemInitPromise;
  }

  // 创建初始化 Promise 并存储到 globalThis
  globalThis.ragSystemInitPromise = (async () => {
    try {
      console.log('[RAG Instance] Creating new RAG system instance...');
      
      const instance = new LocalRAGSystem({
        ollamaBaseUrl: "http://localhost:11434",
        llmModel: "llama3.1",
        embeddingModel: "nomic-embed-text",
      });

      // 初始化数据库
      await instance.initializeDatabase();
      
      // 存储实例到 globalThis
      globalThis.ragSystemInstance = instance;
      
      console.log('[RAG Instance] RAG system instance initialized successfully');
      
      return instance;
    } catch (error) {
      // 如果初始化失败，清除 Promise 以便下次重试
      globalThis.ragSystemInitPromise = undefined;
      throw error;
    }
  })();

  return globalThis.ragSystemInitPromise;
}

export function resetRagSystem() {
  console.log('[RAG Instance] Resetting RAG system instance...');
  globalThis.ragSystemInstance = undefined;
  globalThis.ragSystemInitPromise = undefined;
}

// 获取当前实例（不创建新实例）
export function getCurrentRagSystem(): LocalRAGSystem | undefined {
  return globalThis.ragSystemInstance;
}