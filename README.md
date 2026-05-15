# 企业知识库助手 - RAG + Agent + LangGraph

基于 LangChain 和 LangGraph 构建的企业知识库问答系统，集成 RAG 检索、Agent 工具调用、工作流编排等功能。

## 项目架构

```
rag-pro/
├── main.py              # LangGraph 工作流主程序
├── api.py               # FastAPI 服务接口（流式输出）
├── config.py            # 配置管理（支持动态更新）
├── tools.py             # Agent 工具定义
├── retriever.py         # 检索模块（向量 + BM25 + Rerank）
├── llm_utils.py         # LLM 工具（多问句生成、上下文压缩、回答生成）
├── embedding_utils.py   # 向量生成工具
├── chroma_store.py      # ChromaDB 向量数据库
├── pdf_utils.py         # PDF 解析工具
├── pdf_import_script.py # PDF 导入脚本
├── session_store.py     # Redis 会话存储
├── log_utils.py         # 日志工具
└── requirements.txt     # 依赖包
```

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| LLM 框架 | LangChain + LangGraph | 工作流编排、Agent 构建 |
| 向量数据库 | ChromaDB | 本地向量存储 |
| 向量模型 | 阿里云 DashScope | text-embedding |
| LLM | 通义千问 qwen-turbo | 多问句生成、回答生成 |
| Rerank | gte-rerank-v2 | 精排序优化 |
| 关键词检索 | BM25 (rank_bm25) | 混合检索 |
| Web 框架 | FastAPI | 流式接口、CORS |
| 会话存储 | Redis | 历史对话缓存 |
| 前端 | Vue 3 + Element Plus | 知识库助手界面 |

## 核心功能

### 1. RAG 检索流程

```
用户问题 → 多问句生成 → 向量检索 + BM25检索 → 合并去重 → Rerank精排序 → 上下文压缩 → LLM回答
```

### 2. LangGraph 工作流

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(WorkflowState)

# 添加节点
graph.add_node("init_database", node_init_database)
graph.add_node("multi_query", node_multi_query)
graph.add_node("vector_search", node_vector_search)
graph.add_node("rerank", node_rerank)
graph.add_node("generate_answer", node_generate_answer)

# 添加边
graph.add_edge(START, "init_database")
graph.add_edge("init_database", "multi_query")
graph.add_edge("multi_query", "vector_search")

# 条件边
graph.add_conditional_edges(
    "permission_check",
    lambda state: END if state.get("end") else "agent_tool_call"
)

workflow = graph.compile()
```

### 3. 混合检索

```python
from retriever import hybrid_search, rerank

# 向量 + BM25 混合检索
candidates = hybrid_search(
    db, bm25, query,
    top_k=3,
    vector_weight=0.6,
    bm25_weight=0.4,
    vector_threshold=0.5,
    bm25_threshold=0.5
)

# Rerank 精排序
reranked = rerank(question, candidates)
```

### 4. Agent 工具调用

```python
from langchain.tools import tool

@tool
def get_collection_count(collection_name: str) -> str:
    """查询表数量。参数 collection_name 只能是 'handbook' 或 'regulation'。"""
    db = ChromaDB(collection_name=collection_name)
    db.load()
    count = db.collection.count()
    return f"表 {collection_name} 数据量：{count}"

@tool
def clear_collection(collection_name: str) -> str:
    """清空数据表（高危操作，需管理员权限）。"""
    if not ENABLE_DANGER_OP:
        return "❌ 高危操作已禁用"
    db = ChromaDB(collection_name=collection_name)
    db.clear()
    return f"✅ 已清空表 {collection_name}"
```

### 5. 流式输出

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        # 发送思考日志
        yield f"data: {json.dumps({'type': 'log', 'data': log})}\n\n"
        
        # 发送回答内容
        yield f"data: {json.dumps({'type': 'answer', 'data': answer})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 6. 动态配置

```python
# config.py
class Config:
    TOP_K = 3
    VECTOR_WEIGHT = 0.6
    RERANK_THRESHOLD = 0.3
    
    def to_dict(self):
        return {...}
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

config = Config()

# 其他模块动态获取
from config import config
top_k = config.TOP_K  # 运行时可更新
```

### 7. 历史对话摘要

```python
def summarize_history(messages):
    """使用LLM对历史对话进行摘要"""
    if len(messages) <= 4:
        return messages
    
    old_messages = messages[:-4]
    dialog_text = "\n".join([f"用户：{q}\n助手：{a}" for q, a in old_messages])
    
    summary = llm.invoke(f"请精简总结以下对话：{dialog_text}")
    
    return [("对话历史摘要", summary)] + messages[-4:]
```

## 安装部署

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
API_KEY=your_dashscope_api_key
KEY=your_openai_api_key
```

### 3. 启动 Redis（可选）

```bash
redis-server
```

### 4. 启动后端服务

```bash
python api.py
```

服务地址：`http://localhost:8000`

### 5. 启动前端

```bash
cd knowledge-assistant
npm install
npm run dev
```

## API 接口

### 问答接口（流式）

```
POST /api/chat/stream
Content-Type: application/json

{
  "message": "年假有多少天？",
  "session_id": "user_123"
}
```

响应（SSE）：
```
data: {"type": "log", "data": {"step": "向量检索", "status": "done"}}
data: {"type": "answer", "data": "根据员工手册，年假为5-15天..."}
```

### 配置接口

```
GET /api/config          # 获取配置
POST /api/config         # 更新配置
```

### 统计接口

```
GET /api/stats           # 获取知识库统计
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TOP_K | 3 | 检索返回数量 |
| VECTOR_WEIGHT | 0.6 | 向量检索权重 |
| BM25_WEIGHT | 0.4 | BM25检索权重 |
| RERANK_TOP_K | 5 | Rerank保留数量 |
| RERANK_THRESHOLD | 0 | Rerank阈值 |
| VECTOR_SEARCH_THRESHOLD | 0.5 | 向量检索阈值 |
| BM25_SEARCH_THRESHOLD | 0.5 | BM25检索阈值 |
| ENABLE_HISTORY_SUMMARY | True | 启用历史摘要 |
| HISTORY_SUMMARY_THRESHOLD | 6 | 摘要触发阈值 |

## 使用示例

### 命令行交互

```bash
python main.py

💬 请输入问题：年假有多少天？

📋 开始处理: 年假有多少天？
────────────────────────────────────
✅ 检查加载数据库表: handbook(100条) + regulation(50条)
✅ 生成多检索问句: 3个
✅ 向量检索: 6个候选
✅ Rerank精排序: 保留前3个
✅ 模型生成回答: 根据员工手册，年假为5-15天...
```

### 工具调用

```bash
💬 请输入问题：查询表数量

✅ 调用工具: get_collection_count
📊 表 handbook 数据量：100
📊 表 regulation 数据量：50
```

### 权限控制

```bash
/role admin    # 切换为管理员
/role user     # 切换为普通用户
```

## 项目特色

1. **混合检索**：向量检索 + BM25 关键词检索，提升召回率
2. **Rerank 精排序**：使用 gte-rerank-v2 对候选结果重排序
3. **多问句生成**：LLM 将用户问题扩展为多个检索问句
4. **上下文压缩**：控制上下文长度，减少 Token 消耗
5. **流式输出**：实时展示思考过程和回答内容
6. **动态配置**：前端可实时调整检索参数
7. **历史摘要**：LLM 自动压缩历史对话，节省上下文空间
8. **权限控制**：管理员/普通用户角色区分
9. **工作流编排**：LangGraph 实现灵活的节点编排

## License

MIT
