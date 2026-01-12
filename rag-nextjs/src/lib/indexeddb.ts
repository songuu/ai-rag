// IndexedDB 工具类，用于存储历史对话

export interface ConversationMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  traceId?: string;
  retrievalDetails?: any;
  queryAnalysis?: any;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ConversationMessage[];
  createdAt: Date;
  updatedAt: Date;
}

class IndexedDBManager {
  private dbName = 'RAGConversations';
  private dbVersion = 1;
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => {
        reject(new Error('无法打开 IndexedDB'));
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // 创建对话存储对象
        if (!db.objectStoreNames.contains('conversations')) {
          const conversationStore = db.createObjectStore('conversations', {
            keyPath: 'id'
          });
          conversationStore.createIndex('updatedAt', 'updatedAt', { unique: false });
          conversationStore.createIndex('createdAt', 'createdAt', { unique: false });
        }

        // 创建消息存储对象
        if (!db.objectStoreNames.contains('messages')) {
          const messageStore = db.createObjectStore('messages', {
            keyPath: 'id'
          });
          messageStore.createIndex('conversationId', 'conversationId', { unique: false });
          messageStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };
    });
  }

  async saveConversation(conversation: Conversation): Promise<void> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['conversations'], 'readwrite');
      const store = transaction.objectStore('conversations');
      const request = store.put(conversation);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('保存对话失败'));
    });
  }

  async getAllConversations(): Promise<Conversation[]> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['conversations'], 'readonly');
      const store = transaction.objectStore('conversations');
      const index = store.index('updatedAt');
      const request = index.openCursor(null, 'prev'); // 降序排列

      const conversations: Conversation[] = [];

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
        if (cursor) {
          conversations.push(cursor.value);
          cursor.continue();
        } else {
          resolve(conversations);
        }
      };

      request.onerror = () => reject(new Error('获取对话列表失败'));
    });
  }

  async getConversation(id: string): Promise<Conversation | null> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['conversations'], 'readonly');
      const store = transaction.objectStore('conversations');
      const request = store.get(id);

      request.onsuccess = () => {
        resolve(request.result || null);
      };

      request.onerror = () => reject(new Error('获取对话失败'));
    });
  }

  async deleteConversation(id: string): Promise<void> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['conversations'], 'readwrite');
      const store = transaction.objectStore('conversations');
      const request = store.delete(id);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('删除对话失败'));
    });
  }

  async addMessageToConversation(
    conversationId: string,
    message: ConversationMessage
  ): Promise<void> {
    if (!this.db) await this.init();

    return new Promise(async (resolve, reject) => {
      // 先获取对话
      const conversation = await this.getConversation(conversationId);
      if (!conversation) {
        reject(new Error('对话不存在'));
        return;
      }

      // 添加消息
      conversation.messages.push(message);
      conversation.updatedAt = new Date();

      // 保存对话
      const transaction = this.db!.transaction(['conversations'], 'readwrite');
      const store = transaction.objectStore('conversations');
      const request = store.put(conversation);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('添加消息失败'));
    });
  }

  async createNewConversation(title: string): Promise<Conversation> {
    if (!this.db) await this.init();

    const conversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    await this.saveConversation(conversation);
    return conversation;
  }

  async deleteAllConversations(): Promise<void> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['conversations'], 'readwrite');
      const store = transaction.objectStore('conversations');
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('删除所有对话失败'));
    });
  }

  async getLatestConversation(): Promise<Conversation | null> {
    if (!this.db) await this.init();

    const conversations = await this.getAllConversations();
    return conversations.length > 0 ? conversations[0] : null;
  }
}

export const dbManager = new IndexedDBManager();