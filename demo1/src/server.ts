import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import fs from "fs";
import { LocalRAGSystem } from "./rag-system";

const app = express();
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
  },
});

// 初始化 RAG 系统
const ragSystem = new LocalRAGSystem({
  ollamaBaseUrl: "http://localhost:11434",
  llmModel: "llama3.1",
  embeddingModel: "nomic-embed-text",
});

// 启动时初始化数据库
let isSystemReady = false;

async function initializeSystem() {
  try {
    console.log("正在初始化 RAG 系统...");
    await ragSystem.initializeDatabase("./data");
    isSystemReady = true;
    console.log("RAG 系统初始化完成！");
  } catch (error) {
    console.error("RAG 系统初始化失败:", error);
  }
}

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

// 问答接口
app.post("/api/ask", async (req, res) => {
  try {
    if (!isSystemReady) {
      return res.status(503).json({
        error: "RAG 系统尚未就绪，请稍后再试",
      });
    }

    const { question } = req.body;

    if (!question || typeof question !== "string") {
      return res.status(400).json({
        error: "请提供有效的问题",
      });
    }

    const answer = await ragSystem.ask(question.trim());

    res.json({
      question,
      answer,
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

// 文件上传接口
app.post("/api/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "请选择要上传的文件",
      });
    }

    // 读取上传的文件内容
    const fileContent = fs.readFileSync(req.file.path, "utf8");
    const originalName = req.file.originalname;

    // 保存到数据目录并添加到 RAG 系统
    await ragSystem.saveUploadedFile(originalName, fileContent, "./data");

    // 清理临时文件
    fs.unlinkSync(req.file.path);

    res.json({
      message: "文件上传成功",
      filename: originalName,
      size: req.file.size,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("文件上传错误:", error);
    
    // 清理临时文件
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
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

  // 初始化 RAG 系统
  await initializeSystem();

  // 启动服务器
  app.listen(PORT, () => {
    console.log(`🚀 服务器运行在 http://localhost:${PORT}`);
    console.log(`📚 RAG 系统状态: ${isSystemReady ? '就绪' : '未就绪'}`);
    console.log(`📁 数据目录: ./data`);
    console.log(`🌐 Web 界面: http://localhost:${PORT}`);
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