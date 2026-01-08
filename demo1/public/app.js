// 全局状态
let isAsking = false;
let selectedFiles = [];
let socket = null;
let currentTopK = 3;
let currentThreshold = 0.0;
let radarChart = null;
let currentRetrievalData = null;

// DOM 元素
const elements = {
    // 聊天相关
    chatMessages: document.getElementById('chat-messages'),
    questionForm: document.getElementById('question-form'),
    questionInput: document.getElementById('question-input'),
    askBtn: document.getElementById('ask-btn'),
    
    // 用户问题分析
    userQueryAnalysis: document.getElementById('user-query-analysis'),
    tokenizationTime: document.getElementById('tokenization-time'),
    tokenizationResult: document.getElementById('tokenization-result'),
    embeddingTime: document.getElementById('embedding-time'),
    embeddingResult: document.getElementById('embedding-result'),
    
    // 检索链路可视化
    radarChartContainer: document.getElementById('radar-chart-container'),
    pipelineDetails: document.getElementById('pipeline-details'),
    togglePipelineView: document.getElementById('toggle-pipeline-view'),
    
    // 参数控制
    toggleParams: document.getElementById('toggle-params'),
    paramsPanel: document.getElementById('params-panel'),
    topkSlider: document.getElementById('topk-slider'),
    topkValue: document.getElementById('topk-value'),
    thresholdSlider: document.getElementById('threshold-slider'),
    thresholdValue: document.getElementById('threshold-value'),
    
    // 文件上传相关
    fileUploadArea: document.getElementById('file-upload-area'),
    fileInput: document.getElementById('file-input'),
    uploadBtn: document.getElementById('upload-btn'),
    
    // 文件列表
    filesList: document.getElementById('files-list'),
    refreshFilesBtn: document.getElementById('refresh-files-btn'),
    
    // 实时监控
    vectorizationPanel: document.getElementById('vectorization-panel'),
    vectorizationProgress: document.getElementById('vectorization-progress'),
    vectorizationStatus: document.getElementById('vectorization-status'),
    queryProcessingPanel: document.getElementById('query-processing-panel'),
    queryProcessingSpinner: document.getElementById('query-processing-spinner'),
    queryProcessingStatus: document.getElementById('query-processing-status'),
    processingSteps: document.getElementById('processing-steps'),
    tokenizationStep: document.getElementById('tokenization-step'),
    tokenizationStatus: document.getElementById('tokenization-status'),
    tokenizationDetails: document.getElementById('tokenization-details'),
    preprocessingStep: document.getElementById('preprocessing-step'),
    preprocessingStatus: document.getElementById('preprocessing-status'),
    preprocessingDetails: document.getElementById('preprocessing-details'),
    embeddingStep: document.getElementById('embedding-step'),
    embeddingStatus: document.getElementById('embedding-status'),
    embeddingDetails: document.getElementById('embedding-details'),
    processingSummary: document.getElementById('processing-summary'),
    retrievalPanel: document.getElementById('retrieval-panel'),
    retrievalDetails: document.getElementById('retrieval-details'),
    vectorMatchingPanel: document.getElementById('vector-matching-panel'),
    vectorMatchingDetails: document.getElementById('vector-matching-details'),
    
    // 系统状态
    statusIndicator: document.getElementById('status-indicator'),
    docCount: document.getElementById('doc-count'),
    embeddingDim: document.getElementById('embedding-dim'),
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
    initializeWebSocket();
    checkSystemHealth();
    loadFilesList();
});

// 初始化应用
function initializeApp() {
    console.log('初始化 RAG 系统 Web 界面');
    
    // 初始化参数控制
    updateSliderValues();
    
    // 清空聊天区域的初始消息
    setTimeout(() => {
        if (elements.chatMessages.children.length === 1) {
            // 如果只有欢迎消息，保持它
        }
    }, 1000);
}

// 初始化 WebSocket 连接
function initializeWebSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('WebSocket 连接成功');
        showToast('实时监控连接成功', 'success');
    });
    
    socket.on('disconnect', () => {
        console.log('WebSocket 连接断开');
        showToast('实时监控连接断开', 'warning');
    });
    
    // 监听系统状态更新
    socket.on('system-status', (data) => {
        console.log('系统状态更新:', data);
        if (data.ragStatus) {
            updateSystemStatus({ ragSystem: data.ragStatus, timestamp: new Date().toISOString() });
        }
    });
    
    // 监听向量化进度
    socket.on('vectorization-progress', (progress) => {
        console.log('向量化进度:', progress);
        updateVectorizationProgress(progress);
    });
    
    // 监听检索详情
    socket.on('retrieval-details', (details) => {
        console.log('检索详情:', details);
        updateRetrievalDetails(details);
    });
    
    // 监听查询向量化进度
    socket.on('query-vectorization-progress', (progress) => {
        console.log('查询向量化进度:', progress);
        updateQueryVectorizationProgress(progress);
    });
}

// 设置事件监听器
function setupEventListeners() {
    // 问答表单提交
    elements.questionForm.addEventListener('submit', handleQuestionSubmit);
    
    // 参数控制
    elements.toggleParams.addEventListener('click', toggleParamsPanel);
    elements.topkSlider.addEventListener('input', updateSliderValues);
    elements.thresholdSlider.addEventListener('input', updateSliderValues);
    
    // 检索链路可视化
    elements.togglePipelineView.addEventListener('click', togglePipelineView);
    
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
            body: JSON.stringify({ 
                question,
                topK: currentTopK,
                similarityThreshold: currentThreshold
            }),
        });
        
        const data = await response.json();
        
        // 移除加载消息
        removeMessage(loadingId);
        
        if (response.ok) {
            // 添加AI回答（带检索详情）
            const retrievalDetailsWithQuery = {
                ...data.retrievalDetails,
                query: question // 确保查询被正确传递
            };
            addMessage(data.answer, 'assistant', false, retrievalDetailsWithQuery);
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
function addMessage(content, type, isLoading = false, retrievalDetails = null) {
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
    
    let retrievalDetailsHtml = '';
    if (retrievalDetails && type === 'assistant') {
        const detailsId = 'details-' + messageId;
        retrievalDetailsHtml = `
            <div class="mt-2 p-2 bg-gray-50 rounded text-xs">
                <div class="flex items-center justify-between cursor-pointer" onclick="toggleRetrievalDetails('${detailsId}')">
                    <div class="font-medium text-gray-700">
                        <i class="fas fa-search mr-1"></i>
                        检索详情 (${retrievalDetails.searchResults.length}/${retrievalDetails.totalDocuments} 文档, 查询向量化: ${retrievalDetails.queryVectorizationTime || 0}ms, 检索: ${retrievalDetails.searchTime}ms)
                    </div>
                    <i id="${detailsId}-icon" class="fas fa-chevron-down text-gray-500 transition-transform duration-200"></i>
                </div>
                <div id="${detailsId}" class="mt-2 space-y-2 hidden">
                    ${retrievalDetails.searchResults.map((result, index) => `
                        <div class="p-2 bg-white rounded border-l-2 border-blue-300">
                            <div class="flex justify-between items-center mb-1">
                                <span class="font-medium">${result.document.metadata?.source || `文档${index + 1}`}</span>
                                <div class="flex items-center space-x-2">
                                    <div class="w-16 bg-gray-200 rounded-full h-1">
                                        <div class="bg-gradient-to-r from-green-400 to-blue-500 h-1 rounded-full" 
                                             style="width: ${(result.similarity * 100).toFixed(1)}%"></div>
                                    </div>
                                    <span class="text-blue-600 font-mono">${result.similarity.toFixed(4)}</span>
                                </div>
                            </div>
                            <div class="text-gray-600">
                                <div class="max-h-20 overflow-y-auto">
                                    ${highlightMatchingText(result.document.content, retrievalDetails.query)}
                                </div>
                                <button onclick="showFullContent('${result.document.metadata?.source || `文档${index + 1}`}', \`${result.document.content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`, \`${retrievalDetails.query}\`)" 
                                        class="text-blue-500 hover:text-blue-700 mt-1 text-xs">
                                    <i class="fas fa-expand-alt mr-1"></i>查看完整内容
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
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
                ${retrievalDetailsHtml}
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
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    selectedFiles = [];
    const invalidFiles = [];
    
    // 验证每个文件
    files.forEach(file => {
        // 验证文件类型
        if (!file.name.endsWith('.txt')) {
            invalidFiles.push(`${file.name}: 只支持 .txt 文本文件`);
            return;
        }
        
        // 验证文件大小 (5MB)
        if (file.size > 5 * 1024 * 1024) {
            invalidFiles.push(`${file.name}: 文件太大，最大支持 5MB`);
            return;
        }
        
        selectedFiles.push(file);
    });
    
    // 显示验证结果
    if (invalidFiles.length > 0) {
        showToast(`无效文件: ${invalidFiles.join(', ')}`, 'error');
    }
    
    if (selectedFiles.length > 0) {
        elements.uploadBtn.disabled = false;
        if (selectedFiles.length === 1) {
            elements.uploadBtn.innerHTML = `<i class="fas fa-upload mr-2"></i> 上传 "${selectedFiles[0].name}"`;
        } else {
            elements.uploadBtn.innerHTML = `<i class="fas fa-upload mr-2"></i> 上传 ${selectedFiles.length} 个文件`;
        }
        
        showToast(`已选择 ${selectedFiles.length} 个文件`, 'success');
    } else {
        elements.uploadBtn.disabled = true;
        elements.uploadBtn.innerHTML = '<i class="fas fa-upload mr-2"></i> 上传文件';
    }
}

// 文件上传处理
async function handleFileUpload() {
    if (selectedFiles.length === 0) return;
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
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
            // 显示上传结果
            if (data.results.length > 0) {
                showToast(data.message, 'success');
            }
            
            // 显示错误（如果有）
            if (data.errors && data.errors.length > 0) {
                data.errors.forEach(error => {
                    showToast(`${error.filename}: ${error.error}`, 'error');
                });
            }
            
            // 重置上传状态
            selectedFiles = [];
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
        elements.uploadBtn.disabled = selectedFiles.length === 0;
        if (selectedFiles.length > 0) {
            if (selectedFiles.length === 1) {
                elements.uploadBtn.innerHTML = `<i class="fas fa-upload mr-2"></i> 上传 "${selectedFiles[0].name}"`;
            } else {
                elements.uploadBtn.innerHTML = `<i class="fas fa-upload mr-2"></i> 上传 ${selectedFiles.length} 个文件`;
            }
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

// 参数控制相关函数
function toggleParamsPanel() {
    const isHidden = elements.paramsPanel.classList.contains('hidden');
    if (isHidden) {
        elements.paramsPanel.classList.remove('hidden');
        elements.toggleParams.innerHTML = '<i class="fas fa-chevron-up"></i> 收起';
    } else {
        elements.paramsPanel.classList.add('hidden');
        elements.toggleParams.innerHTML = '<i class="fas fa-chevron-down"></i> 展开';
    }
}

function updateSliderValues() {
    currentTopK = parseInt(elements.topkSlider.value);
    currentThreshold = parseFloat(elements.thresholdSlider.value);
    
    elements.topkValue.textContent = currentTopK;
    elements.thresholdValue.textContent = currentThreshold.toFixed(1);
}

// 实时监控相关函数
function updateVectorizationProgress(progress) {
    elements.vectorizationPanel.classList.remove('hidden');
    
    const percentage = (progress.current / progress.total) * 100;
    elements.vectorizationProgress.style.width = `${percentage}%`;
    
    elements.vectorizationStatus.textContent = 
        `${progress.current}/${progress.total} - ${progress.document}`;
    
    if (progress.current === progress.total) {
        setTimeout(() => {
            elements.vectorizationPanel.classList.add('hidden');
        }, 3000);
    }
}

function updateQueryVectorizationProgress(progress) {
    elements.queryProcessingPanel.classList.remove('hidden');
    elements.processingSteps.classList.remove('hidden');
    
    // 重置所有步骤状态
    const resetStepStatus = (stepElement, statusElement) => {
        stepElement.classList.remove('border-blue-400', 'border-green-400', 'border-yellow-400');
        stepElement.classList.add('border-gray-300');
        const indicator = statusElement.querySelector('.w-2');
        const text = statusElement.querySelector('span');
        indicator.className = 'w-2 h-2 rounded-full bg-gray-300';
        text.textContent = '等待中';
        text.className = 'text-xs text-gray-500';
    };
    
    switch (progress.status) {
        case 'starting':
            elements.queryProcessingSpinner.classList.remove('hidden');
            elements.queryProcessingStatus.textContent = `开始处理查询: "${progress.query}"`;
            elements.processingSummary.classList.add('hidden');
            
            // 重置所有步骤
            resetStepStatus(elements.tokenizationStep, elements.tokenizationStatus);
            resetStepStatus(elements.preprocessingStep, elements.preprocessingStatus);
            resetStepStatus(elements.embeddingStep, elements.embeddingStatus);
            break;
            
        case 'tokenizing':
            // 更新词元化步骤状态
            elements.tokenizationStep.classList.remove('border-gray-300');
            elements.tokenizationStep.classList.add('border-yellow-400');
            const tokenIndicator = elements.tokenizationStatus.querySelector('.w-2');
            const tokenText = elements.tokenizationStatus.querySelector('span');
            tokenIndicator.className = 'w-2 h-2 rounded-full bg-yellow-400';
            tokenText.textContent = '处理中...';
            tokenText.className = 'text-xs text-yellow-600';
            break;
            
        case 'preprocessing':
            // 完成词元化
            elements.tokenizationStep.classList.remove('border-yellow-400');
            elements.tokenizationStep.classList.add('border-green-400');
            const tokenDoneIndicator = elements.tokenizationStatus.querySelector('.w-2');
            const tokenDoneText = elements.tokenizationStatus.querySelector('span');
            tokenDoneIndicator.className = 'w-2 h-2 rounded-full bg-green-400';
            tokenDoneText.textContent = `完成 (${progress.tokenization.processingTime}ms)`;
            tokenDoneText.className = 'text-xs text-green-600';
            
            // 显示词元化详情
            elements.tokenizationDetails.classList.remove('hidden');
            elements.tokenizationDetails.innerHTML = `
                <div class="space-y-1">
                    <div>词元数量: ${progress.tokenization.tokenCount}</div>
                    <div>词元: [${progress.tokenization.tokens.slice(0, 10).join(', ')}${progress.tokenization.tokens.length > 10 ? '...' : ''}]</div>
                    <div>处理后文本: "${progress.tokenization.processedText.substring(0, 50)}${progress.tokenization.processedText.length > 50 ? '...' : ''}"</div>
                </div>
            `;
            
            // 开始预处理
            elements.preprocessingStep.classList.remove('border-gray-300');
            elements.preprocessingStep.classList.add('border-yellow-400');
            const prepIndicator = elements.preprocessingStatus.querySelector('.w-2');
            const prepText = elements.preprocessingStatus.querySelector('span');
            prepIndicator.className = 'w-2 h-2 rounded-full bg-yellow-400';
            prepText.textContent = '处理中...';
            prepText.className = 'text-xs text-yellow-600';
            break;
            
        case 'embedding':
            // 完成预处理
            elements.preprocessingStep.classList.remove('border-yellow-400');
            elements.preprocessingStep.classList.add('border-green-400');
            const prepDoneIndicator = elements.preprocessingStatus.querySelector('.w-2');
            const prepDoneText = elements.preprocessingStatus.querySelector('span');
            prepDoneIndicator.className = 'w-2 h-2 rounded-full bg-green-400';
            prepDoneText.textContent = '完成';
            prepDoneText.className = 'text-xs text-green-600';
            
            // 显示预处理详情
            elements.preprocessingDetails.classList.remove('hidden');
            elements.preprocessingDetails.innerHTML = `
                <div>文本标准化和清理完成</div>
            `;
            
            // 开始向量化
            elements.embeddingStep.classList.remove('border-gray-300');
            elements.embeddingStep.classList.add('border-yellow-400');
            const embIndicator = elements.embeddingStatus.querySelector('.w-2');
            const embText = elements.embeddingStatus.querySelector('span');
            embIndicator.className = 'w-2 h-2 rounded-full bg-yellow-400';
            embText.textContent = '向量化中...';
            embText.className = 'text-xs text-yellow-600';
            break;
            
        case 'completed':
            elements.queryProcessingSpinner.classList.add('hidden');
            elements.queryProcessingStatus.textContent = `查询处理完成 (总耗时: ${progress.totalTime}ms)`;
            
            // 完成向量化
            elements.embeddingStep.classList.remove('border-yellow-400');
            elements.embeddingStep.classList.add('border-green-400');
            const embDoneIndicator = elements.embeddingStatus.querySelector('.w-2');
            const embDoneText = elements.embeddingStatus.querySelector('span');
            embDoneIndicator.className = 'w-2 h-2 rounded-full bg-green-400';
            embDoneText.textContent = `完成 (${progress.embedding.processingTime}ms)`;
            embDoneText.className = 'text-xs text-green-600';
            
            // 显示向量化详情
            elements.embeddingDetails.classList.remove('hidden');
            elements.embeddingDetails.innerHTML = `
                <div class="space-y-1">
                    <div>模型: ${progress.embedding.modelInfo.name}</div>
                    <div>向量维度: ${progress.embedding.embeddingDimension}</div>
                    <div>前10维: [${progress.embedding.embedding.slice(0, 10).map(v => v.toFixed(3)).join(', ')}...]</div>
                </div>
            `;
            
            // 显示总结
            elements.processingSummary.classList.remove('hidden');
            elements.processingSummary.innerHTML = `
                <div class="flex justify-between">
                    <span>总处理时间: ${progress.totalTime}ms</span>
                    <span>词元化: ${progress.tokenization.processingTime}ms | 向量化: ${progress.embedding.processingTime}ms</span>
                </div>
            `;
            
            // 显示用户问题分析结果
            showUserQueryAnalysis(progress);
            
            // 5秒后隐藏面板
            setTimeout(() => {
                elements.queryProcessingPanel.classList.add('hidden');
            }, 5000);
            break;
    }
}

function updateRetrievalDetails(details) {
    elements.retrievalPanel.classList.remove('hidden');
    elements.vectorMatchingPanel.classList.remove('hidden');
    
    // 更新检索详情
    elements.retrievalDetails.innerHTML = `
        <div class="bg-blue-50 p-2 rounded mb-2">
            <div class="font-medium">查询: "${details.query}"</div>
            <div class="text-gray-600">
                参数: Top-${details.topK}, 阈值≥${details.threshold}, 
                耗时: ${details.searchTime}ms
            </div>
        </div>
        <div class="space-y-1">
            ${details.searchResults.map((result, index) => `
                <div class="bg-white p-2 rounded border-l-2 border-green-400">
                    <div class="flex justify-between">
                        <span class="font-medium">${result.document.metadata?.source || `文档${index + 1}`}</span>
                        <span class="text-green-600 font-mono text-xs">${result.similarity.toFixed(4)}</span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    // 更新向量匹配详情
    elements.vectorMatchingDetails.innerHTML = `
        <div class="mb-2">
            <div class="font-medium">查询向量维度: ${details.queryEmbedding.length}</div>
            <div class="text-gray-600">前10维: [${details.queryEmbedding.slice(0, 10).map(v => v.toFixed(3)).join(', ')}...]</div>
        </div>
        <div class="space-y-1">
            <div class="font-medium">相似度排序:</div>
            ${details.searchResults.map((result, index) => {
                const barWidth = (result.similarity * 100).toFixed(1);
                return `
                    <div class="flex items-center space-x-2">
                        <span class="text-xs w-12">#${index + 1}</span>
                        <div class="flex-1 bg-gray-200 rounded-full h-2">
                            <div class="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full" 
                                 style="width: ${barWidth}%"></div>
                        </div>
                        <span class="text-xs font-mono w-12">${result.similarity.toFixed(3)}</span>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

// 更新系统状态显示
function updateSystemStatus(data) {
    if (data && data.ragSystem.initialized) {
        // 系统就绪
        elements.statusIndicator.innerHTML = `
            <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
            <span class="text-sm text-green-600">系统就绪</span>
        `;
        elements.docCount.textContent = data.ragSystem.documentCount;
        elements.embeddingDim.textContent = data.ragSystem.embeddingDimension || '-';
        elements.systemStatus.textContent = '正常运行';
        elements.lastUpdate.textContent = formatDate(data.timestamp);
    } else {
        // 系统未就绪
        elements.statusIndicator.innerHTML = `
            <div class="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
            <span class="text-sm text-red-600">系统未就绪</span>
        `;
        elements.docCount.textContent = '-';
        elements.embeddingDim.textContent = '-';
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

// 显示用户问题分析结果
function showUserQueryAnalysis(progress) {
    elements.userQueryAnalysis.classList.remove('hidden');
    
    // 显示词元化时间
    elements.tokenizationTime.textContent = `${progress.tokenization.processingTime}ms`;
    
    // 显示词元化结果
    const tokenizationHtml = `
        <div class="mb-2">
            <span class="text-xs text-gray-500">原始文本:</span>
            <div class="bg-gray-50 rounded px-2 py-1 text-sm font-mono">"${progress.tokenization.originalText}"</div>
        </div>
        <div class="mb-2">
            <span class="text-xs text-gray-500">Token 分解 (${progress.tokenization.tokenCount} 个词元):</span>
            <div class="flex flex-wrap gap-1 mt-1">
                ${progress.tokenization.tokenInfos.map(token => {
                    let colorClass = 'bg-gray-100 text-gray-700';
                    switch(token.type) {
                        case 'chinese': colorClass = 'bg-red-100 text-red-700'; break;
                        case 'english': colorClass = 'bg-blue-100 text-blue-700'; break;
                        case 'number': colorClass = 'bg-green-100 text-green-700'; break;
                        case 'punctuation': colorClass = 'bg-yellow-100 text-yellow-700'; break;
                    }
                    return `
                        <span class="${colorClass} px-2 py-1 rounded text-xs font-mono" title="Token ID: ${token.tokenId}">
                            ${token.token}
                            <sub class="text-xs opacity-60">${token.tokenId}</sub>
                        </span>
                    `;
                }).join('')}
            </div>
        </div>
        <div class="text-xs text-gray-500">
            <div class="flex space-x-4">
                <span><span class="inline-block w-3 h-3 bg-red-100 rounded mr-1"></span>中文</span>
                <span><span class="inline-block w-3 h-3 bg-blue-100 rounded mr-1"></span>英文</span>
                <span><span class="inline-block w-3 h-3 bg-green-100 rounded mr-1"></span>数字</span>
                <span><span class="inline-block w-3 h-3 bg-yellow-100 rounded mr-1"></span>标点</span>
            </div>
        </div>
    `;
    elements.tokenizationResult.innerHTML = tokenizationHtml;
    
    // 显示向量化时间
    elements.embeddingTime.textContent = `${progress.embedding.processingTime}ms`;
    
    // 显示向量化结果
    const semanticAnalysis = progress.embedding.semanticAnalysis;
    const vectorFeatures = semanticAnalysis.vectorFeatures || {};
    
    const embeddingHtml = `
        <div class="mb-2">
            <span class="text-xs text-gray-500">语义分析:</span>
            <div class="bg-gradient-to-r from-purple-50 to-blue-50 rounded p-2 mt-1">
                <div class="text-sm font-medium text-purple-800">${semanticAnalysis.context}</div>
                <div class="text-xs text-purple-600 mt-1">
                    分类: ${semanticAnalysis.semanticCategory} 
                    (置信度: ${(semanticAnalysis.confidence * 100).toFixed(1)}%)
                </div>
                <div class="text-xs text-purple-600 mt-1">
                    相关概念: ${semanticAnalysis.nearestConcepts.join(', ')}
                </div>
            </div>
        </div>
        <div class="mb-2">
            <span class="text-xs text-gray-500">向量特征分析:</span>
            <div class="bg-gray-50 rounded p-2 mt-1 space-y-1">
                <div class="flex justify-between text-xs">
                    <span>技术特征:</span>
                    <span class="font-mono">${(vectorFeatures.techScore || 0).toFixed(3)}</span>
                </div>
                <div class="flex justify-between text-xs">
                    <span>商业特征:</span>
                    <span class="font-mono">${(vectorFeatures.businessScore || 0).toFixed(3)}</span>
                </div>
                <div class="flex justify-between text-xs">
                    <span>日常特征:</span>
                    <span class="font-mono">${(vectorFeatures.dailyScore || 0).toFixed(3)}</span>
                </div>
                <div class="flex justify-between text-xs">
                    <span>情感倾向:</span>
                    <span class="font-mono">${(vectorFeatures.emotionScore || 0).toFixed(3)}</span>
                </div>
                <div class="flex justify-between text-xs">
                    <span>向量模长:</span>
                    <span class="font-mono">${(vectorFeatures.vectorMagnitude || 0).toFixed(3)}</span>
                </div>
            </div>
        </div>
        <div class="mb-2">
            <span class="text-xs text-gray-500">向量表示 (${progress.embedding.embeddingDimension} 维):</span>
            <div class="bg-gray-50 rounded p-2 mt-1">
                <div class="text-xs font-mono text-gray-600">
                    [${progress.embedding.embedding.slice(0, 8).map(v => v.toFixed(3)).join(', ')}, ...]
                </div>
                <div class="text-xs text-gray-500 mt-1">
                    显示前8维，完整向量包含 ${progress.embedding.embeddingDimension} 个浮点数
                </div>
            </div>
        </div>
        <div class="text-xs text-gray-500">
            模型: ${progress.embedding.modelInfo.name} | 词汇表大小: ${progress.embedding.modelInfo.vocabularySize || 'N/A'}
        </div>
    `;
    elements.embeddingResult.innerHTML = embeddingHtml;
    
    // 初始化雷达图
    initializeRadarChart(progress);
    
    // 存储当前数据用于后续检索分析
    currentRetrievalData = progress;
}

// 文本高亮和内容展示辅助函数
function highlightMatchingText(content, query) {
    if (!query || query.length < 2) return content.substring(0, 200) + '...';
    
    // 简单的关键词高亮
    const keywords = query.toLowerCase().split(/\s+/).filter(word => word.length > 1);
    let highlightedContent = content;
    
    keywords.forEach(keyword => {
        const regex = new RegExp(`(${keyword})`, 'gi');
        highlightedContent = highlightedContent.replace(regex, '<mark class="bg-yellow-200">$1</mark>');
    });
    
    // 截取前200个字符
    if (highlightedContent.length > 200) {
        highlightedContent = highlightedContent.substring(0, 200) + '...';
    }
    
    return highlightedContent;
}

function showFullContent(filename, content, query) {
    // 创建模态框显示完整内容
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4';
    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    };
    
    const highlightedContent = highlightMatchingText(content, query);
    
    modal.innerHTML = `
        <div class="bg-white rounded-lg max-w-4xl max-h-full overflow-hidden flex flex-col">
            <div class="flex justify-between items-center p-4 border-b">
                <h3 class="text-lg font-medium">${filename}</h3>
                <button onclick="this.closest('.fixed').remove()" class="text-gray-400 hover:text-gray-600">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            <div class="p-4 overflow-y-auto flex-1">
                <div class="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                    ${highlightedContent}
                </div>
            </div>
            <div class="p-4 border-t bg-gray-50 text-xs text-gray-500">
                <i class="fas fa-info-circle mr-1"></i>
                高亮显示与查询 "${query || '当前查询'}" 相关的关键词
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

// 初始化雷达图
function initializeRadarChart(progress) {
    if (!elements.radarChartContainer) return;
    
    // 销毁现有图表
    if (radarChart) {
        radarChart.dispose();
    }
    
    // 创建新图表
    radarChart = echarts.init(elements.radarChartContainer);
    
    const vectorFeatures = progress.embedding.semanticAnalysis.vectorFeatures || {};
    
    // 雷达图配置
    const option = {
        title: {
            text: '向量特征分析',
            left: 'center',
            textStyle: {
                // 距离下方30px
                fontSize: 12,
                color: '#374151'
            }
        },
        tooltip: {
            trigger: 'item',
            // formatter: function(params) {
            //     console.log("item", params.value)
            //     return `${params.name}: ${params.value.toFixed(3)}`;
            // }
        },
        radar: {
            indicator: [
                { name: '技术特征', max: 1 },
                { name: '商业特征', max: 1 },
                { name: '日常特征', max: 1 },
                { name: '情感倾向', max: 1, min: -1 },
                { name: '向量强度', max: Math.max(1, vectorFeatures.vectorMagnitude || 1) }
            ],
            radius: '60%',
            axisName: {
                fontSize: 10,
                color: '#6B7280'
            },
            splitLine: {
                lineStyle: {
                    color: '#E5E7EB'
                }
            },
            axisLine: {
                lineStyle: {
                    color: '#D1D5DB'
                }
            }
        },
        series: [{
            name: '向量特征',
            type: 'radar',
            data: [{
                value: [
                    vectorFeatures.techScore || 0,
                    vectorFeatures.businessScore || 0,
                    vectorFeatures.dailyScore || 0,
                    vectorFeatures.emotionScore || 0,
                    (vectorFeatures.vectorMagnitude || 0) / Math.max(1, vectorFeatures.vectorMagnitude || 1)
                ],
                name: '当前查询',
                itemStyle: {
                    color: '#3B82F6'
                },
                areaStyle: {
                    color: 'rgba(59, 130, 246, 0.2)'
                }
            }]
        }]
    };
    
    radarChart.setOption(option);
    
    // 更新检索链路详情
    updatePipelineDetails(progress);
}

// 更新检索链路详情
function updatePipelineDetails(progress) {
    const vectorFeatures = progress.embedding.semanticAnalysis.vectorFeatures || {};
    const semanticAnalysis = progress.embedding.semanticAnalysis;
    
    const pipelineHtml = `
        <div class="space-y-2">
            <div class="text-xs font-medium text-gray-700">检索链路分析:</div>
            
            <div class="bg-blue-50 rounded p-2">
                <div class="text-xs font-medium text-blue-800 mb-1">1. 查询理解</div>
                <div class="text-xs text-blue-600">
                    • 词元数量: ${progress.tokenization.tokenCount}<br>
                    • 语义分类: ${semanticAnalysis.semanticCategory}<br>
                    • 置信度: ${(semanticAnalysis.confidence * 100).toFixed(1)}%
                </div>
            </div>
            
            <div class="bg-green-50 rounded p-2">
                <div class="text-xs font-medium text-green-800 mb-1">2. 向量编码</div>
                <div class="text-xs text-green-600">
                    • 向量维度: ${progress.embedding.embeddingDimension}<br>
                    • 向量模长: ${vectorFeatures.vectorMagnitude?.toFixed(3) || 'N/A'}<br>
                    • 主要特征: ${getMainFeature(vectorFeatures)}
                </div>
            </div>
            
            <div class="bg-purple-50 rounded p-2">
                <div class="text-xs font-medium text-purple-800 mb-1">3. 相似度计算</div>
                <div class="text-xs text-purple-600">
                    • 算法: 余弦相似度<br>
                    • 搜索空间: ${progress.embedding.embeddingDimension} 维向量空间<br>
                    • 匹配策略: Top-K + 阈值过滤
                </div>
            </div>
            
            <div class="bg-orange-50 rounded p-2">
                <div class="text-xs font-medium text-orange-800 mb-1">4. 结果排序</div>
                <div class="text-xs text-orange-600">
                    • 排序依据: 相似度分数<br>
                    • 过滤条件: 阈值 ≥ ${currentThreshold}<br>
                    • 返回数量: Top-${currentTopK}
                </div>
            </div>
        </div>
    `;
    
    elements.pipelineDetails.innerHTML = pipelineHtml;
}

// 获取主要特征
function getMainFeature(vectorFeatures) {
    if (!vectorFeatures) return '未知';
    
    const features = [
        { name: '技术', score: vectorFeatures.techScore || 0 },
        { name: '商业', score: vectorFeatures.businessScore || 0 },
        { name: '日常', score: vectorFeatures.dailyScore || 0 }
    ];
    
    features.sort((a, b) => b.score - a.score);
    return features[0].name + '导向';
}

// 切换检索链路视图
function togglePipelineView() {
    if (!radarChart) return;
    
    // 这里可以添加不同的视图切换逻辑
    // 例如：雷达图 <-> 柱状图 <-> 折线图
    console.log('切换检索链路视图');
    
    // 简单的重新渲染
    if (currentRetrievalData) {
        initializeRadarChart(currentRetrievalData);
    }
}

// 折叠/展开检索详情
function toggleRetrievalDetails(detailsId) {
    const details = document.getElementById(detailsId);
    const icon = document.getElementById(detailsId + '-icon');
    
    if (details.classList.contains('hidden')) {
        details.classList.remove('hidden');
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
        icon.style.transform = 'rotate(180deg)';
    } else {
        details.classList.add('hidden');
        icon.classList.remove('fa-chevron-up');
        icon.classList.add('fa-chevron-down');
        icon.style.transform = 'rotate(0deg)';
    }
}

// 定期检查系统状态
setInterval(checkSystemHealth, 30000); // 每30秒检查一次