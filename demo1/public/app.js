// 全局状态
let isAsking = false;
let selectedFile = null;

// DOM 元素
const elements = {
    // 聊天相关
    chatMessages: document.getElementById('chat-messages'),
    questionForm: document.getElementById('question-form'),
    questionInput: document.getElementById('question-input'),
    askBtn: document.getElementById('ask-btn'),
    
    // 文件上传相关
    fileUploadArea: document.getElementById('file-upload-area'),
    fileInput: document.getElementById('file-input'),
    uploadBtn: document.getElementById('upload-btn'),
    
    // 文件列表
    filesList: document.getElementById('files-list'),
    refreshFilesBtn: document.getElementById('refresh-files-btn'),
    
    // 系统状态
    statusIndicator: document.getElementById('status-indicator'),
    docCount: document.getElementById('doc-count'),
    systemStatus: document.getElementById('system-status'),
    lastUpdate: document.getElementById('last-update'),
    refreshBtn: document.getElementById('refresh-btn'),
    reinitializeBtn: document.getElementById('reinitialize-btn'),
    
    // Toast 容器
    toastContainer: document.getElementById('toast-container')
};

// 初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkSystemHealth();
    loadFilesList();
});

// 初始化应用
function initializeApp() {
    console.log('初始化 RAG 系统 Web 界面');
    
    // 清空聊天区域的初始消息
    setTimeout(() => {
        if (elements.chatMessages.children.length === 1) {
            // 如果只有欢迎消息，保持它
        }
    }, 1000);
}

// 设置事件监听器
function setupEventListeners() {
    // 问答表单提交
    elements.questionForm.addEventListener('submit', handleQuestionSubmit);
    
    // 文件上传相关
    elements.fileUploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileUploadArea.addEventListener('dragover', handleDragOver);
    elements.fileUploadArea.addEventListener('dragleave', handleDragLeave);
    elements.fileUploadArea.addEventListener('drop', handleFileDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.uploadBtn.addEventListener('click', handleFileUpload);
    
    // 刷新按钮
    elements.refreshBtn.addEventListener('click', checkSystemHealth);
    elements.refreshFilesBtn.addEventListener('click', loadFilesList);
    elements.reinitializeBtn.addEventListener('click', handleReinitialize);
    
    // 回车键发送消息
    elements.questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isAsking) {
                elements.questionForm.dispatchEvent(new Event('submit'));
            }
        }
    });
}

// 处理问题提交
async function handleQuestionSubmit(e) {
    e.preventDefault();
    
    if (isAsking) return;
    
    const question = elements.questionInput.value.trim();
    if (!question) return;
    
    // 添加用户消息到聊天
    addMessage(question, 'user');
    
    // 清空输入框
    elements.questionInput.value = '';
    
    // 设置加载状态
    setAskingState(true);
    
    // 添加加载消息
    const loadingId = addMessage('正在思考中...', 'assistant', true);
    
    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question }),
        });
        
        const data = await response.json();
        
        // 移除加载消息
        removeMessage(loadingId);
        
        if (response.ok) {
            // 添加AI回答
            addMessage(data.answer, 'assistant');
        } else {
            // 添加错误消息
            addMessage(`错误: ${data.error}`, 'error');
            showToast(data.error, 'error');
        }
    } catch (error) {
        console.error('问答请求失败:', error);
        removeMessage(loadingId);
        addMessage('网络错误，请检查服务器连接', 'error');
        showToast('网络错误，请检查服务器连接', 'error');
    } finally {
        setAskingState(false);
    }
}

// 添加消息到聊天区域
function addMessage(content, type, isLoading = false) {
    const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `chat-message flex ${type === 'user' ? 'justify-end' : 'justify-start'}`;
    
    let iconClass, bgClass, textClass;
    
    switch (type) {
        case 'user':
            iconClass = 'fas fa-user';
            bgClass = 'bg-blue-600 text-white';
            textClass = 'text-white';
            break;
        case 'assistant':
            iconClass = 'fas fa-robot';
            bgClass = 'bg-gray-100 text-gray-900';
            textClass = 'text-gray-900';
            break;
        case 'error':
            iconClass = 'fas fa-exclamation-triangle';
            bgClass = 'bg-red-100 text-red-900';
            textClass = 'text-red-900';
            break;
    }
    
    messageDiv.innerHTML = `
        <div class="flex max-w-xs lg:max-w-md ${type === 'user' ? 'flex-row-reverse' : 'flex-row'}">
            <div class="flex-shrink-0 ${type === 'user' ? 'ml-3' : 'mr-3'}">
                <div class="w-8 h-8 rounded-full ${bgClass} flex items-center justify-center">
                    <i class="${iconClass} text-sm"></i>
                </div>
            </div>
            <div class="flex-1">
                <div class="${bgClass} rounded-lg px-4 py-2 ${type === 'user' ? 'rounded-br-none' : 'rounded-bl-none'}">
                    <p class="${textClass} text-sm whitespace-pre-wrap">${isLoading ? `<span class="typing-indicator"></span> ${content}` : content}</p>
                </div>
                <div class="text-xs text-gray-500 mt-1 ${type === 'user' ? 'text-right' : 'text-left'}">
                    ${new Date().toLocaleTimeString()}
                </div>
            </div>
        </div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    return messageId;
}

// 移除消息
function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

// 设置问答状态
function setAskingState(asking) {
    isAsking = asking;
    elements.askBtn.disabled = asking;
    elements.questionInput.disabled = asking;
    
    if (asking) {
        elements.askBtn.innerHTML = '<div class="typing-indicator"></div> 思考中...';
    } else {
        elements.askBtn.innerHTML = '<i class="fas fa-paper-plane mr-2"></i> 发送';
    }
}

// 文件拖拽处理
function handleDragOver(e) {
    e.preventDefault();
    elements.fileUploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.fileUploadArea.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    elements.fileUploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files } });
    }
}

// 文件选择处理
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    // 验证文件类型
    if (!file.name.endsWith('.txt')) {
        showToast('只支持 .txt 文本文件', 'error');
        return;
    }
    
    // 验证文件大小 (5MB)
    if (file.size > 5 * 1024 * 1024) {
        showToast('文件太大，最大支持 5MB', 'error');
        return;
    }
    
    selectedFile = file;
    elements.uploadBtn.disabled = false;
    elements.uploadBtn.innerHTML = `<i class="fas fa-upload mr-2"></i> 上传 "${file.name}"`;
    
    showToast(`已选择文件: ${file.name}`, 'success');
}

// 文件上传处理
async function handleFileUpload() {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // 设置上传状态
    elements.uploadBtn.disabled = true;
    elements.uploadBtn.innerHTML = '<div class="typing-indicator"></div> 上传中...';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast(`文件 "${data.filename}" 上传成功`, 'success');
            
            // 重置上传状态
            selectedFile = null;
            elements.fileInput.value = '';
            elements.uploadBtn.disabled = true;
            elements.uploadBtn.innerHTML = '<i class="fas fa-upload mr-2"></i> 上传文件';
            
            // 刷新文件列表和系统状态
            loadFilesList();
            checkSystemHealth();
        } else {
            showToast(data.error, 'error');
        }
    } catch (error) {
        console.error('文件上传失败:', error);
        showToast('文件上传失败，请检查网络连接', 'error');
    } finally {
        elements.uploadBtn.disabled = selectedFile === null;
        if (selectedFile) {
            elements.uploadBtn.innerHTML = `<i class="fas fa-upload mr-2"></i> 上传 "${selectedFile.name}"`;
        } else {
            elements.uploadBtn.innerHTML = '<i class="fas fa-upload mr-2"></i> 上传文件';
        }
    }
}

// 加载文件列表
async function loadFilesList() {
    try {
        const response = await fetch('/api/files');
        const data = await response.json();
        
        if (response.ok) {
            renderFilesList(data.files);
        } else {
            elements.filesList.innerHTML = '<p class="text-red-500 text-sm">加载文件列表失败</p>';
        }
    } catch (error) {
        console.error('加载文件列表失败:', error);
        elements.filesList.innerHTML = '<p class="text-red-500 text-sm">网络错误</p>';
    }
}

// 渲染文件列表
function renderFilesList(files) {
    if (files.length === 0) {
        elements.filesList.innerHTML = `
            <div class="text-center text-gray-500 text-sm">
                <i class="fas fa-folder-open text-2xl mb-2"></i>
                <p>暂无文件</p>
            </div>
        `;
        return;
    }
    
    const filesHtml = files.map(file => `
        <div class="flex items-center justify-between p-3 border rounded-lg mb-2">
            <div class="flex-1">
                <div class="flex items-center">
                    <i class="fas fa-file-alt text-blue-500 mr-2"></i>
                    <span class="text-sm font-medium text-gray-900 truncate">${file.name}</span>
                </div>
                <div class="text-xs text-gray-500 mt-1">
                    ${formatFileSize(file.size)} • ${formatDate(file.modified)}
                </div>
            </div>
            <button 
                onclick="deleteFile('${file.name}')" 
                class="ml-2 p-1 text-red-400 hover:text-red-600"
                title="删除文件"
            >
                <i class="fas fa-trash text-sm"></i>
            </button>
        </div>
    `).join('');
    
    elements.filesList.innerHTML = filesHtml;
}

// 删除文件
async function deleteFile(filename) {
    if (!confirm(`确定要删除文件 "${filename}" 吗？`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/files/${encodeURIComponent(filename)}`, {
            method: 'DELETE',
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast(`文件 "${filename}" 删除成功`, 'success');
            loadFilesList();
            checkSystemHealth();
        } else {
            showToast(data.error, 'error');
        }
    } catch (error) {
        console.error('删除文件失败:', error);
        showToast('删除文件失败，请检查网络连接', 'error');
    }
}

// 检查系统健康状态
async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (response.ok) {
            updateSystemStatus(data);
        } else {
            updateSystemStatus(null);
        }
    } catch (error) {
        console.error('检查系统状态失败:', error);
        updateSystemStatus(null);
    }
}

// 更新系统状态显示
function updateSystemStatus(data) {
    if (data && data.ragSystem.ready) {
        // 系统就绪
        elements.statusIndicator.innerHTML = `
            <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
            <span class="text-sm text-green-600">系统就绪</span>
        `;
        elements.docCount.textContent = data.ragSystem.documentCount;
        elements.systemStatus.textContent = '正常运行';
        elements.lastUpdate.textContent = formatDate(data.timestamp);
    } else {
        // 系统未就绪
        elements.statusIndicator.innerHTML = `
            <div class="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
            <span class="text-sm text-red-600">系统未就绪</span>
        `;
        elements.docCount.textContent = '-';
        elements.systemStatus.textContent = '未就绪';
        elements.lastUpdate.textContent = '-';
    }
}

// 重新初始化系统
async function handleReinitialize() {
    if (!confirm('确定要重新初始化系统吗？这将重新加载所有文档。')) {
        return;
    }
    
    elements.reinitializeBtn.disabled = true;
    elements.reinitializeBtn.innerHTML = '<div class="typing-indicator"></div> 初始化中...';
    
    try {
        const response = await fetch('/api/reinitialize', {
            method: 'POST',
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast('系统重新初始化成功', 'success');
            checkSystemHealth();
        } else {
            showToast(data.error, 'error');
        }
    } catch (error) {
        console.error('重新初始化失败:', error);
        showToast('重新初始化失败，请检查网络连接', 'error');
    } finally {
        elements.reinitializeBtn.disabled = false;
        elements.reinitializeBtn.innerHTML = '<i class="fas fa-redo mr-2"></i> 重新初始化';
    }
}

// 显示 Toast 通知
function showToast(message, type = 'info') {
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `transform transition-all duration-300 translate-x-full`;
    
    let bgClass, iconClass;
    switch (type) {
        case 'success':
            bgClass = 'bg-green-500';
            iconClass = 'fas fa-check-circle';
            break;
        case 'error':
            bgClass = 'bg-red-500';
            iconClass = 'fas fa-exclamation-circle';
            break;
        case 'warning':
            bgClass = 'bg-yellow-500';
            iconClass = 'fas fa-exclamation-triangle';
            break;
        default:
            bgClass = 'bg-blue-500';
            iconClass = 'fas fa-info-circle';
    }
    
    toast.innerHTML = `
        <div class="${bgClass} text-white px-4 py-3 rounded-lg shadow-lg flex items-center max-w-sm">
            <i class="${iconClass} mr-3"></i>
            <span class="text-sm">${message}</span>
            <button onclick="removeToast('${toastId}')" class="ml-3 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    // 动画显示
    setTimeout(() => {
        toast.classList.remove('translate-x-full');
    }, 100);
    
    // 自动移除
    setTimeout(() => {
        removeToast(toastId);
    }, 5000);
}

// 移除 Toast
function removeToast(toastId) {
    const toast = document.getElementById(toastId);
    if (toast) {
        toast.classList.add('translate-x-full');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }
}

// 工具函数
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// 定期检查系统状态
setInterval(checkSystemHealth, 30000); // 每30秒检查一次