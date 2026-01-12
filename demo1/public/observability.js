// 🔍 RAG 可观测性仪表盘 - Langfuse 风格

// 全局状态
let traces = [];
let selectedTrace = null;
let performanceChart = null;
let socket = null;

// DOM 元素
const elements = {
    // 统计卡片
    totalTraces: document.getElementById('total-traces'),
    successRate: document.getElementById('success-rate'),
    avgDuration: document.getElementById('avg-duration'),
    totalTokens: document.getElementById('total-tokens'),
    
    // 列表和详情
    tracesList: document.getElementById('traces-list'),
    traceDetails: document.getElementById('trace-details'),
    
    // 按钮
    refreshBtn: document.getElementById('refresh-btn'),
    clearBtn: document.getElementById('clear-btn'),
    
    // 模态框
    traceModal: document.getElementById('trace-modal'),
    modalTraceContent: document.getElementById('modal-trace-content'),
    closeModal: document.getElementById('close-modal'),
    
    // 图表
    performanceChart: document.getElementById('performance-chart')
};

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🔍 可观测性仪表盘初始化...');
    
    // 初始化 WebSocket
    initializeWebSocket();
    
    // 初始化性能图表
    initializePerformanceChart();
    
    // 绑定事件监听器
    bindEventListeners();
    
    // 加载初始数据
    await loadTraces();
    
    console.log('✅ 可观测性仪表盘初始化完成');
});

// 初始化 WebSocket
function initializeWebSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('🔌 WebSocket 连接成功');
    });
    
    socket.on('disconnect', () => {
        console.log('🔌 WebSocket 连接断开');
    });
    
    // 监听 Trace 更新
    socket.on('trace-update', (trace) => {
        console.log('🔍 收到 Trace 更新:', trace.name);
        updateTraceInList(trace);
        updateStatistics();
        updatePerformanceChart();
    });
}

// 绑定事件监听器
function bindEventListeners() {
    elements.refreshBtn.addEventListener('click', loadTraces);
    elements.clearBtn.addEventListener('click', clearTraces);
    elements.closeModal.addEventListener('click', closeModal);
    
    // 点击模态框外部关闭
    elements.traceModal.addEventListener('click', (e) => {
        if (e.target === elements.traceModal) {
            closeModal();
        }
    });
}

// 加载 Traces
async function loadTraces() {
    try {
        console.log('📡 加载 Traces...');
        elements.refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>加载中';
        
        const response = await fetch('/api/traces');
        const data = await response.json();
        
        if (data.success) {
            traces = data.traces;
            updateStatistics(data.stats);
            renderTracesList();
            updatePerformanceChart();
            console.log(`✅ 加载了 ${traces.length} 个 Traces`);
        } else {
            console.error('❌ 加载 Traces 失败:', data.error);
        }
    } catch (error) {
        console.error('❌ 加载 Traces 错误:', error);
    } finally {
        elements.refreshBtn.innerHTML = '<i class="fas fa-sync-alt mr-1"></i>刷新';
    }
}

// 更新统计信息
function updateStatistics(stats) {
    if (!stats) {
        // 从当前 traces 计算统计
        stats = calculateStats(traces);
    }
    
    elements.totalTraces.textContent = stats.totalTraces || 0;
    elements.successRate.textContent = `${(stats.successRate * 100).toFixed(1)}%`;
    elements.avgDuration.textContent = `${Math.round(stats.avgDuration)}ms`;
    elements.totalTokens.textContent = stats.totalTokens || 0;
}

// 计算统计信息
function calculateStats(traces) {
    if (!traces || traces.length === 0) {
        return {
            totalTraces: 0,
            successRate: 0,
            avgDuration: 0,
            totalTokens: 0
        };
    }
    
    const successCount = traces.filter(t => t.status === 'SUCCESS').length;
    const successRate = successCount / traces.length;
    
    const completedTraces = traces.filter(t => t.endTime);
    const totalDuration = completedTraces.reduce((sum, trace) => {
        const duration = new Date(trace.endTime).getTime() - new Date(trace.startTime).getTime();
        return sum + duration;
    }, 0);
    const avgDuration = completedTraces.length > 0 ? totalDuration / completedTraces.length : 0;
    
    const totalTokens = traces.reduce((sum, trace) => {
        return sum + trace.observations
            .filter(obs => obs.type === 'GENERATION')
            .reduce((obsSum, gen) => obsSum + (gen.usage?.totalTokens || 0), 0);
    }, 0);
    
    return {
        totalTraces: traces.length,
        successRate,
        avgDuration,
        totalTokens
    };
}

// 渲染 Traces 列表
function renderTracesList() {
    if (!traces || traces.length === 0) {
        elements.tracesList.innerHTML = `
            <div class="text-center text-gray-500 py-8">
                <i class="fas fa-search text-3xl mb-2"></i>
                <p>暂无 Traces 数据</p>
                <p class="text-sm">开始提问以生成 Traces</p>
            </div>
        `;
        return;
    }
    
    // 按时间倒序排列
    const sortedTraces = [...traces].sort((a, b) => 
        new Date(b.startTime).getTime() - new Date(a.startTime).getTime()
    );
    
    const tracesHtml = sortedTraces.map(trace => renderTraceCard(trace)).join('');
    elements.tracesList.innerHTML = tracesHtml;
}

// 渲染单个 Trace 卡片
function renderTraceCard(trace) {
    const duration = trace.endTime 
        ? new Date(trace.endTime).getTime() - new Date(trace.startTime).getTime()
        : null;
    
    const statusColor = {
        'SUCCESS': 'text-green-600 bg-green-100',
        'ERROR': 'text-red-600 bg-red-100',
        'PENDING': 'text-yellow-600 bg-yellow-100'
    }[trace.status] || 'text-gray-600 bg-gray-100';
    
    const totalTokens = trace.observations
        .filter(obs => obs.type === 'GENERATION')
        .reduce((sum, gen) => sum + (gen.usage?.totalTokens || 0), 0);
    
    return `
        <div class="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition-colors" 
             onclick="selectTrace('${trace.id}')">
            <div class="flex items-start justify-between mb-2">
                <div class="flex-1">
                    <div class="flex items-center space-x-2 mb-1">
                        <h3 class="font-medium text-gray-900">${trace.name}</h3>
                        <span class="px-2 py-1 text-xs rounded-full ${statusColor}">
                            ${trace.status}
                        </span>
                    </div>
                    <p class="text-sm text-gray-600 truncate">
                        ${trace.input?.question || 'No question'}
                    </p>
                </div>
                <div class="text-right text-sm text-gray-500">
                    <div>${new Date(trace.startTime).toLocaleTimeString()}</div>
                    ${duration ? `<div>${duration}ms</div>` : ''}
                </div>
            </div>
            
            <div class="flex items-center justify-between text-xs text-gray-500">
                <div class="flex items-center space-x-4">
                    <span>
                        <i class="fas fa-eye mr-1"></i>
                        ${trace.observations.length} observations
                    </span>
                    <span>
                        <i class="fas fa-coins mr-1"></i>
                        ${totalTokens} tokens
                    </span>
                    ${trace.scores.length > 0 ? `
                        <span>
                            <i class="fas fa-star mr-1"></i>
                            ${trace.scores.length} scores
                        </span>
                    ` : ''}
                </div>
                <div>
                    ${trace.userId ? `User: ${trace.userId}` : ''}
                </div>
            </div>
        </div>
    `;
}

// 选择 Trace
async function selectTrace(traceId) {
    try {
        const response = await fetch(`/api/traces/${traceId}`);
        const data = await response.json();
        
        if (data.success) {
            selectedTrace = data.trace;
            renderTraceDetails(selectedTrace);
            openModal();
        } else {
            console.error('❌ 获取 Trace 详情失败:', data.error);
        }
    } catch (error) {
        console.error('❌ 获取 Trace 详情错误:', error);
    }
}

// 渲染 Trace 详情
function renderTraceDetails(trace) {
    const duration = trace.endTime 
        ? new Date(trace.endTime).getTime() - new Date(trace.startTime).getTime()
        : null;
    
    const detailsHtml = `
        <div class="space-y-6">
            <!-- Trace 基本信息 -->
            <div class="bg-gray-50 rounded-lg p-4">
                <h4 class="font-semibold text-gray-900 mb-3">基本信息</h4>
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <span class="text-gray-500">Trace ID:</span>
                        <div class="font-mono text-xs bg-white px-2 py-1 rounded mt-1">${trace.id}</div>
                    </div>
                    <div>
                        <span class="text-gray-500">状态:</span>
                        <div class="mt-1">
                            <span class="px-2 py-1 text-xs rounded-full ${getStatusColor(trace.status)}">
                                ${trace.status}
                            </span>
                        </div>
                    </div>
                    <div>
                        <span class="text-gray-500">开始时间:</span>
                        <div class="mt-1">${new Date(trace.startTime).toLocaleString()}</div>
                    </div>
                    <div>
                        <span class="text-gray-500">耗时:</span>
                        <div class="mt-1">${duration ? `${duration}ms` : '进行中'}</div>
                    </div>
                </div>
            </div>
            
            <!-- 输入输出 -->
            <div class="bg-blue-50 rounded-lg p-4">
                <h4 class="font-semibold text-gray-900 mb-3">输入输出</h4>
                <div class="space-y-3">
                    <div>
                        <span class="text-gray-500 text-sm">输入:</span>
                        <div class="bg-white rounded p-3 mt-1">
                            <pre class="text-sm text-gray-800 whitespace-pre-wrap">${JSON.stringify(trace.input, null, 2)}</pre>
                        </div>
                    </div>
                    ${trace.output ? `
                        <div>
                            <span class="text-gray-500 text-sm">输出:</span>
                            <div class="bg-white rounded p-3 mt-1">
                                <pre class="text-sm text-gray-800 whitespace-pre-wrap">${JSON.stringify(trace.output, null, 2)}</pre>
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
            
            <!-- Observations 树 -->
            <div class="bg-green-50 rounded-lg p-4">
                <h4 class="font-semibold text-gray-900 mb-3">
                    <i class="fas fa-sitemap mr-2"></i>
                    Observations 树 (${trace.observations.length})
                </h4>
                <div class="trace-tree">
                    ${renderObservationsTree(trace.observations)}
                </div>
            </div>
            
            <!-- 评分 -->
            ${trace.scores.length > 0 ? `
                <div class="bg-yellow-50 rounded-lg p-4">
                    <h4 class="font-semibold text-gray-900 mb-3">
                        <i class="fas fa-star mr-2"></i>
                        评分 (${trace.scores.length})
                    </h4>
                    <div class="space-y-2">
                        ${trace.scores.map(score => `
                            <div class="bg-white rounded p-3 text-sm">
                                <div class="flex items-center justify-between">
                                    <span class="font-medium">${score.name}</span>
                                    <span class="text-gray-500">${score.source}</span>
                                </div>
                                <div class="mt-1">
                                    <span class="text-lg font-bold text-blue-600">${score.value}</span>
                                    ${score.comment ? `<p class="text-gray-600 mt-1">${score.comment}</p>` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <!-- 反馈按钮 -->
            <div class="flex space-x-2">
                <button onclick="addFeedback('${trace.id}', true)" 
                        class="bg-green-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-green-700">
                    <i class="fas fa-thumbs-up mr-1"></i>有用
                </button>
                <button onclick="addFeedback('${trace.id}', false)" 
                        class="bg-red-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-red-700">
                    <i class="fas fa-thumbs-down mr-1"></i>无用
                </button>
            </div>
        </div>
    `;
    
    elements.modalTraceContent.innerHTML = detailsHtml;
}

// 渲染 Observations 树
function renderObservationsTree(observations) {
    // 构建层级结构
    const rootObservations = observations.filter(obs => !obs.parentObservationId);
    
    return rootObservations.map(obs => renderObservationNode(obs, observations, 0)).join('');
}

// 渲染单个 Observation 节点
function renderObservationNode(observation, allObservations, depth) {
    const children = allObservations.filter(obs => obs.parentObservationId === observation.id);
    const duration = observation.endTime 
        ? new Date(observation.endTime).getTime() - new Date(observation.startTime).getTime()
        : null;
    
    const typeIcon = {
        'GENERATION': 'fas fa-robot text-blue-600',
        'SPAN': 'fas fa-clock text-green-600',
        'EVENT': 'fas fa-bolt text-yellow-600'
    }[observation.type] || 'fas fa-circle';
    
    const nodeClass = `observation-node ${observation.type.toLowerCase()}-node`;
    
    let html = `
        <div class="${nodeClass}" style="margin-left: ${depth * 1.5}rem;">
            <div class="bg-white rounded p-3 mb-2 border-l-4">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                        <i class="${typeIcon}"></i>
                        <span class="font-medium">${observation.name}</span>
                        <span class="text-xs px-2 py-1 bg-gray-100 rounded">${observation.type}</span>
                    </div>
                    <div class="text-sm text-gray-500">
                        ${duration ? `${duration}ms` : '进行中'}
                    </div>
                </div>
                
                ${observation.input ? `
                    <div class="text-xs text-gray-600 mb-1">
                        <strong>输入:</strong> ${JSON.stringify(observation.input).substring(0, 100)}...
                    </div>
                ` : ''}
                
                ${observation.output ? `
                    <div class="text-xs text-gray-600 mb-1">
                        <strong>输出:</strong> ${JSON.stringify(observation.output).substring(0, 100)}...
                    </div>
                ` : ''}
                
                ${observation.usage ? `
                    <div class="text-xs text-gray-500">
                        <i class="fas fa-coins mr-1"></i>
                        Tokens: ${observation.usage.totalTokens} 
                        (${observation.usage.promptTokens}+${observation.usage.completionTokens})
                    </div>
                ` : ''}
            </div>
        </div>
    `;
    
    // 递归渲染子节点
    if (children.length > 0) {
        html += children.map(child => renderObservationNode(child, allObservations, depth + 1)).join('');
    }
    
    return html;
}

// 获取状态颜色
function getStatusColor(status) {
    const colors = {
        'SUCCESS': 'text-green-600 bg-green-100',
        'ERROR': 'text-red-600 bg-red-100',
        'PENDING': 'text-yellow-600 bg-yellow-100'
    };
    return colors[status] || 'text-gray-600 bg-gray-100';
}

// 添加反馈
async function addFeedback(traceId, isPositive) {
    try {
        const response = await fetch(`/api/traces/${traceId}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                score: isPositive,
                comment: isPositive ? '用户认为有用' : '用户认为无用'
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log('✅ 反馈已添加');
            // 重新加载 Trace 详情
            await selectTrace(traceId);
        } else {
            console.error('❌ 添加反馈失败:', data.error);
        }
    } catch (error) {
        console.error('❌ 添加反馈错误:', error);
    }
}

// 更新 Trace 列表中的单个 Trace
function updateTraceInList(updatedTrace) {
    const index = traces.findIndex(t => t.id === updatedTrace.id);
    if (index >= 0) {
        traces[index] = updatedTrace;
    } else {
        traces.unshift(updatedTrace);
    }
    renderTracesList();
}

// 清除所有 Traces
async function clearTraces() {
    if (!confirm('确定要清除所有 Traces 数据吗？此操作不可撤销。')) {
        return;
    }
    
    try {
        elements.clearBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>清除中';
        
        const response = await fetch('/api/traces', {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            traces = [];
            renderTracesList();
            updateStatistics();
            updatePerformanceChart();
            console.log('✅ Traces 数据已清除');
        } else {
            console.error('❌ 清除 Traces 失败:', data.error);
        }
    } catch (error) {
        console.error('❌ 清除 Traces 错误:', error);
    } finally {
        elements.clearBtn.innerHTML = '<i class="fas fa-trash mr-1"></i>清除数据';
    }
}

// 初始化性能图表
function initializePerformanceChart() {
    performanceChart = echarts.init(elements.performanceChart);
    
    const option = {
        title: {
            text: '响应时间趋势',
            textStyle: {
                fontSize: 14,
                color: '#374151'
            }
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                return `${params[0].name}<br/>响应时间: ${params[0].value}ms`;
            }
        },
        xAxis: {
            type: 'category',
            data: [],
            axisLabel: {
                fontSize: 10
            }
        },
        yAxis: {
            type: 'value',
            name: '时间 (ms)',
            axisLabel: {
                fontSize: 10
            }
        },
        series: [{
            name: '响应时间',
            type: 'line',
            data: [],
            smooth: true,
            itemStyle: {
                color: '#3B82F6'
            },
            areaStyle: {
                color: 'rgba(59, 130, 246, 0.1)'
            }
        }]
    };
    
    performanceChart.setOption(option);
}

// 更新性能图表
function updatePerformanceChart() {
    if (!performanceChart || !traces || traces.length === 0) return;
    
    const completedTraces = traces
        .filter(t => t.endTime)
        .sort((a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime())
        .slice(-10); // 只显示最近10个
    
    const times = completedTraces.map(t => new Date(t.startTime).toLocaleTimeString());
    const durations = completedTraces.map(t => 
        new Date(t.endTime).getTime() - new Date(t.startTime).getTime()
    );
    
    performanceChart.setOption({
        xAxis: {
            data: times
        },
        series: [{
            data: durations
        }]
    });
}

// 打开模态框
function openModal() {
    elements.traceModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

// 关闭模态框
function closeModal() {
    elements.traceModal.classList.add('hidden');
    document.body.style.overflow = 'auto';
}

// 窗口大小改变时重新调整图表
window.addEventListener('resize', () => {
    if (performanceChart) {
        performanceChart.resize();
    }
});