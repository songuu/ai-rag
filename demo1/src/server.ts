import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import fs from "fs";
import { createServer } from "http";
import { Server } from "socket.io";
import { LocalRAGSystem } from "./rag-system";
import { type Trace } from "./observability";

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});
const PORT = process.env.PORT || 3000;

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "../public")));

// 配置文件上传
const upload = multer({
  dest: "uploads/",
  fileFilter: (req, file, cb) => {
    // 只允许文本文件
    if (file.mimetype === "text/plain" || file.originalname.endsWith(".txt")) {
      cb(null, true);
    } else {
      cb(new Error("只支持 .txt 文本文件"));
    }
  },
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB 限制
    files: 10, // 最多10个文件
  },
});

// 初始化 RAG 系统（带实时监控回调）
const ragSystem = new LocalRAGSystem({
  ollamaBaseUrl: "http://localhost:11434",
  llmModel: "llama3.1",
  embeddingModel: "nomic-embed-text",
  onVectorizationProgress: (progress) => {
    // 向所有连接的客户端发送向量化进度
    io.emit('vectorization-progress', progress);
    console.log(`向量化进度: ${progress.current}/${progress.total} - ${progress.document}`);
  },
  onRetrievalDetails: (details) => {
    // 向所有连接的客户端发送检索详情
    io.emit('retrieval-details', details);
    console.log(`检索完成: 查询="${details.query}", 找到${details.searchResults.length}个结果`);
  },
  onQueryVectorizationProgress: (progress) => {
    // 向所有连接的客户端发送查询向量化进度
    io.emit('query-vectorization-progress', progress);
    console.log(`查询向量化: ${progress.status} - "${progress.query}"`);
  },
  onTraceUpdate: (trace) => {
    // 🎯 Langfuse 风格的 Trace 更新
    io.emit('trace-update', trace);
    console.log(`🔍 Trace 更新: ${trace.name} [${trace.status}] - ${trace.observations.length} observations`);
  }
});

// 启动时初始化数据库
let isSystemReady = false;

async function initializeSystem() {
  try {
    console.log("正在初始化 RAG 系统...");
    io.emit('system-status', { status: 'initializing', message: '正在初始化 RAG 系统...' });
    
    await ragSystem.initializeDatabase("./data");
    isSystemReady = true;
    
    console.log("RAG 系统初始化完成！");
    io.emit('system-status', { status: 'ready', message: 'RAG 系统初始化完成！' });
  } catch (error) {
    console.error("RAG 系统初始化失败:", error);
    io.emit('system-status', { status: 'error', message: `初始化失败: ${error}` });
  }
}

// WebSocket 连接处理
io.on('connection', (socket) => {
  console.log('客户端连接:', socket.id);
  
  // 发送当前系统状态
  socket.emit('system-status', { 
    status: isSystemReady ? 'ready' : 'initializing', 
    message: isSystemReady ? '系统就绪' : '系统初始化中...',
    ragStatus: ragSystem.getStatus()
  });
  
  // 处理客户端断开连接
  socket.on('disconnect', () => {
    console.log('客户端断开连接:', socket.id);
  });
  
  // 处理客户端请求系统状态
  socket.on('request-status', () => {
    socket.emit('system-status', { 
      status: isSystemReady ? 'ready' : 'initializing', 
      message: isSystemReady ? '系统就绪' : '系统初始化中...',
      ragStatus: ragSystem.getStatus()
    });
  });
});

// API 路由

// 健康检查
app.get("/api/health", (req, res) => {
  const status = ragSystem.getStatus();
  res.json({
    status: "ok",
    ragSystem: {
      ready: isSystemReady,
      ...status,
    },
    timestamp: new Date().toISOString(),
  });
});

// 问答接口（增强版）
app.post("/api/ask", async (req, res) => {
  try {
    if (!isSystemReady) {
      return res.status(503).json({
        error: "RAG 系统尚未就绪，请稍后再试",
      });
    }

    const { 
      question, 
      topK = 3, 
      similarityThreshold = 0.0 
    } = req.body;

    if (!question || typeof question !== "string") {
      return res.status(400).json({
        error: "请提供有效的问题",
      });
    }

    // 使用增强的问答方法
    const result = await ragSystem.askWithDetails(question.trim(), {
      topK: parseInt(topK),
      similarityThreshold: parseFloat(similarityThreshold),
      userId: req.body.userId,
      sessionId: req.body.sessionId
    });

    res.json({
      question,
      answer: result.answer,
      retrievalDetails: {
        searchResults: result.retrievalDetails.searchResults.map(r => ({
          document: {
            content: r.document.pageContent,
            metadata: r.document.metadata
          },
          similarity: r.similarity,
          index: r.index
        })),
        queryEmbedding: result.retrievalDetails.queryEmbedding.slice(0, 10), // 只返回前10维用于显示
        threshold: result.retrievalDetails.threshold,
        topK: result.retrievalDetails.topK,
        totalDocuments: result.retrievalDetails.totalDocuments,
        searchTime: result.retrievalDetails.searchTime
      },
      context: result.context,
      traceId: result.traceId,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("问答处理错误:", error);
    res.status(500).json({
      error: "处理问题时发生错误",
      details: error instanceof Error ? error.message : "未知错误",
    });
  }
});

// 文件上传接口（支持多文件）
app.post("/api/upload", upload.array("files", 10), async (req, res) => {
  try {
    const files = req.files as Express.Multer.File[];
    
    if (!files || files.length === 0) {
      return res.status(400).json({
        error: "请选择要上传的文件",
      });
    }

    const uploadResults = [];
    const errors = [];

    // 处理每个文件
    for (const file of files) {
      try {
        // 读取上传的文件内容
        const fileContent = fs.readFileSync(file.path, "utf8");
        const originalName = file.originalname;

        // 保存到数据目录并添加到 RAG 系统
        await ragSystem.saveUploadedFile(originalName, fileContent, "./data");

        // 清理临时文件
        fs.unlinkSync(file.path);

        uploadResults.push({
          filename: originalName,
          size: file.size,
          status: "success"
        });

        console.log(`文件上传成功: ${originalName}`);
      } catch (error) {
        console.error(`文件 ${file.originalname} 上传失败:`, error);
        
        // 清理临时文件
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }

        errors.push({
          filename: file.originalname,
          error: error instanceof Error ? error.message : "未知错误"
        });
      }
    }

    res.json({
      message: `成功上传 ${uploadResults.length} 个文件${errors.length > 0 ? `，${errors.length} 个文件失败` : ''}`,
      results: uploadResults,
      errors: errors,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("文件上传错误:", error);
    
    // 清理所有临时文件
    const files = req.files as Express.Multer.File[];
    if (files) {
      files.forEach(file => {
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      });
    }

    res.status(500).json({
      error: "文件上传失败",
      details: error instanceof Error ? error.message : "未知错误",
    });
  }
});

// 获取已上传的文件列表
app.get("/api/files", (req, res) => {
  try {
    const dataDir = "./data";
    
    if (!fs.existsSync(dataDir)) {
      return res.json({ files: [] });
    }

    const files = fs.readdirSync(dataDir)
      .filter(file => file.endsWith('.txt'))
      .map(file => {
        const filePath = path.join(dataDir, file);
        const stats = fs.statSync(filePath);
        return {
          name: file,
          size: stats.size,
          modified: stats.mtime.toISOString(),
        };
      });

    res.json({ files });
  } catch (error) {
    console.error("获取文件列表错误:", error);
    res.status(500).json({
      error: "获取文件列表失败",
    });
  }
});

// 删除文件接口
app.delete("/api/files/:filename", (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join("./data", filename);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({
        error: "文件不存在",
      });
    }

    fs.unlinkSync(filePath);

    res.json({
      message: "文件删除成功",
      filename,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("删除文件错误:", error);
    res.status(500).json({
      error: "删除文件失败",
    });
  }
});

// 重新初始化系统
app.post("/api/reinitialize", async (req, res) => {
  try {
    console.log("重新初始化 RAG 系统...");
    ragSystem.clearDatabase();
    await ragSystem.initializeDatabase("./data");
    isSystemReady = true;

    res.json({
      message: "系统重新初始化成功",
      status: ragSystem.getStatus(),
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("重新初始化错误:", error);
    isSystemReady = false;
    res.status(500).json({
      error: "重新初始化失败",
      details: error instanceof Error ? error.message : "未知错误",
    });
  }
});

// 提供主页
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../public/index.html"));
});

// 错误处理中间件
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error("服务器错误:", error);
  
  if (error instanceof multer.MulterError) {
    if (error.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({
        error: "文件太大，最大支持 5MB",
      });
    }
  }

  res.status(500).json({
    error: "服务器内部错误",
    details: error.message,
  });
});

// 启动服务器
async function startServer() {
  // 创建必要的目录
  const dirs = ["./data", "./uploads", "./public"];
  dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  });

  // 🎯 可观测性 API 端点

  // 获取所有 Traces
  app.get("/api/traces", async (req, res) => {
    try {
      const observabilityData = ragSystem.getObservabilityData();
      res.json({
        success: true,
        traces: observabilityData.traces,
        stats: observabilityData.stats
      });
    } catch (error) {
      console.error("获取 Traces 错误:", error);
      res.status(500).json({
        error: "获取 Traces 失败",
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // 获取特定 Trace
  app.get("/api/traces/:traceId", async (req, res) => {
    try {
      const { traceId } = req.params;
      const trace = ragSystem.getTrace(traceId);
      
      if (!trace) {
        return res.status(404).json({
          error: "Trace 不存在"
        });
      }
      
      res.json({
        success: true,
        trace
      });
    } catch (error) {
      console.error("获取 Trace 错误:", error);
      res.status(500).json({
        error: "获取 Trace 失败",
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // 添加用户反馈
  app.post("/api/traces/:traceId/feedback", async (req, res) => {
    try {
      const { traceId } = req.params;
      const { score, comment } = req.body;
      
      if (score === undefined) {
        return res.status(400).json({
          error: "请提供评分"
        });
      }
      
      const scoreId = ragSystem.addUserFeedback(traceId, score, comment);
      
      res.json({
        success: true,
        scoreId,
        message: "反馈已记录"
      });
    } catch (error) {
      console.error("添加反馈错误:", error);
      res.status(500).json({
        error: "添加反馈失败",
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // 清除可观测性数据
  app.delete("/api/traces", async (req, res) => {
    try {
      ragSystem.clearObservabilityData();
      res.json({
        success: true,
        message: "可观测性数据已清除"
      });
    } catch (error) {
      console.error("清除数据错误:", error);
      res.status(500).json({
        error: "清除数据失败",
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // 初始化 RAG 系统
  await initializeSystem();

  // 启动服务器
  server.listen(PORT, () => {
    console.log(`🚀 服务器运行在 http://localhost:${PORT}`);
    console.log(`📚 RAG 系统状态: ${isSystemReady ? '就绪' : '未就绪'}`);
    console.log(`📁 数据目录: ./data`);
    console.log(`🌐 Web 界面: http://localhost:${PORT}`);
    console.log(`🔌 WebSocket 连接已启用`);
  });
}

// 优雅关闭
process.on("SIGINT", () => {
  console.log("\n正在关闭服务器...");
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("\n正在关闭服务器...");
  process.exit(0);
});

// 启动应用
if (require.main === module) {
  startServer().catch(console.error);
}

export { app, ragSystem };