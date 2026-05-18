# 企业知识库助手 - RAG + Agent + LangGraph

基于 LangChain 和 LangGraph 构建的企业知识库问答系统，集成 RAG 检索、Agent 工具调用、工作流编排等功能。
使用的模型是通义千问 qwen-turbo，支持多问句生成、上下文压缩、生成回答。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件，配置 API Key：

```
API_KEY=your_dashscope_api_key
```

> 获取 API Key：访问 [阿里云 DashScope](https://dashscope.console.aliyun.com/) 创建 API Key

### 3. 导入知识库文档

#### 3.1 创建 docs 目录

```bash
# Windows
mkdir docs

# Linux/Mac
mkdir -p docs
```

#### 3.2 放入 PDF 文件

将您的 PDF 文档放入 `docs` 目录：

```bash
# 复制单个文件
cp 员工手册.pdf docs/
cp 规章制度.pdf docs/

# 或复制多个文件
cp *.pdf docs/
```

**支持的文件格式：**
- `.pdf` / `.PDF` 文件

**文件命名建议：**
- 使用有意义的文件名，如 `员工手册.pdf`、`规章制度.pdf`
- 文件名会自动作为知识库表名（转小写、替换特殊字符）
- 例如：`新员工手册.pdf` → 表名 `xin_yuangong_shouce`

#### 3.3 执行导入脚本

```bash
python pdf_import_script.py
```

**导入过程：**
```
📝 扫描PDF文件 → 发现 2 个 PDF 文件
📖 读取PDF文件: ./docs/员工手册.pdf
✂️ 文本分块 → 分块数量：150
🔢 生成向量 → 生成150个向量
📥 导入ChromaDB → 表名：yuan_gong_shou_ce
✅ 成功导入表【yuan_gong_shou_ce】

📖 读取PDF文件: ./docs/规章制度.pdf
✂️ 文本分块 → 分块数量：80
🔢 生成向量 → 生成80个向量
📥 导入ChromaDB → 表名：gui_zhang_zhi_du
✅ 成功导入表【gui_zhang_zhi_du】

✅ 所有 PDF 导入完成
```

#### 3.4 验证导入结果

```bash
# 启动命令行交互
python main.py

# 输入：查询表数量
💬 请输入问题：查询表数量
# 输出示例：
# 表 yuan_gong_shou_ce 数据量：150
# 表 gui_zhang_zhi_du 数据量：80
```

### 4. 启动服务

#### 方式一：API 服务（推荐）

```bash
python api.py
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/

#### 方式二：命令行交互

```bash
python main.py
```

### 5. 开始问答

#### 命令行问答

```bash
💬 请输入问题：年假有多少天？

📋 开始处理: 年假有多少天？
────────────────────────────────────────
✅ 检查加载数据库表: yuan_gong_shou_ce(150条) + gui_zhang_zhi_du(80条)
✅ 获取历史消息: 已获取0条历史消息
✅ 生成多检索问句: 生成3个检索问句
✅ 向量检索: 检索到12个候选
✅ Rerank精排序: 排序完成，保留前5个结果
✅ 模型生成回答: 回答长度:156字符

🤖 回答：
根据《员工手册》规定，员工年假天数如下：
- 工龄满1年不满10年：5天
- 工龄满10年不满20年：10天
- 工龄满20年及以上：15天
```

#### API 问答

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "年假有多少天？", "session_id": "user_001"}'
```

### 6. 常用操作

| 操作 | 命令/输入 | 说明 |
|------|-----------|------|
| 查询表数量 | `查询表数量` | 显示所有知识库表的数据量 |
| 清空指定表 | `清空表名` | 清空指定的知识库表（需管理员权限） |
| 清空所有表 | `清空all` | 清空所有知识库表（需管理员权限） |
| 重新导入 | `导入PDF` | 重新扫描并导入 docs 目录下的 PDF |
| 切换角色 | `/role admin` | 切换为管理员角色 |

---

## 项目架构

```
rag-pro/
├── docs/                # 文档目录，用于存储导入的 PDF 文件
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
| PDF解析 | PyPDF2 + 智能分块 | 文档解析、语义分块 |
| Web 框架 | FastAPI | 流式接口、CORS |
| 会话存储 | Redis | 历史对话缓存、支持TTL过期、分布式 |
| 前端 | Vue 3 + Element Plus | 知识库助手界面 |

---

## 核心功能详解

### RAG 检索流程

```
PDF导入 → 智能分块 → 向量化存储
                                    ↓
用户问题 → Redis获取历史 → 多问句生成 → 向量检索 + BM25检索 → 合并去重 → Rerank精排序 → 上下文压缩 → LLM回答 → Redis保存历史
```

下面详细分析每个步骤的技术选型和实现代码。

---

## 步骤1：多问句生成（Query Expansion）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **单问句检索** | 简单直接，延迟低 | 召回率低，语义覆盖不足 | 简单问答 |
| **LLM多问句生成** | 语义扩展，召回率高 | 增加LLM调用成本 | 复杂查询、多角度问题 |
| **同义词扩展** | 无LLM成本，速度快 | 需要维护同义词库 | 特定领域 |
| **Query2Query** | 基于历史优化问句 | 依赖历史数据 | 多轮对话 |

**本项目选择：LLM多问句生成**

选择理由：
1. 用户问题可能表述不精确，需要语义扩展
2. 多角度检索提高召回率
3. 通义千问成本低，延迟可接受

### 代码实现

```python
# llm_utils.py
def generate_multi_query(question: str) -> list:
    """使用LLM生成多个检索问句"""
    prompt = f"""你是一个AI助手，帮助生成多个相关的检索问句。
用户问题：{question}

请生成3个不同角度的检索问句，用于知识库检索，提高召回率。
要求：
1. 保持原问题的核心语义
2. 从不同角度表述
3. 简洁明了

返回格式：JSON数组，如：["问句1", "问句2", "问句3"]
"""
    
    response = requests.post(LLM_URL, json={
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]}
    })
    
    queries = json.loads(response.json()["output"]["text"])
    return [question] + queries  # 原问题 + 扩展问句
```

**示例：**
```
输入：年假有多少天？
输出：["年假有多少天？", "年假天数规定", "员工年假政策"]
```

---

## 步骤2：向量检索（Vector Search）

### 技术选型分析

| 向量数据库 | 优点 | 缺点 | 适用场景 |
|------------|------|------|----------|
| **ChromaDB** | 轻量级、分布式、易部署、Python原生、支持Chroma Cloud | 自托管需配置 | 中小规模、本地/云端部署 |
| **Milvus** | 高性能、分布式、云原生 | 部署复杂 | 大规模生产环境 |
| **Pinecone** | 全托管、零运维 | 收费、数据出境 | 快速上线 |
| **Weaviate** | 语义搜索强、GraphQL | 学习成本高 | 复杂语义场景 |
| **FAISS** | Meta开源、极高性能 | 仅向量存储 | 纯向量检索 |
| **Qdrant** | Rust实现、高性能 | 社区较小 | 高性能需求 |

**本项目选择：ChromaDB**

选择理由：
1. 轻量级，支持本地运行、自托管或 Chroma Cloud 云端部署、易拓展分布式
2. Python原生，与LangChain无缝集成
3. 支持本地持久化，适合中小规模知识库
4. 开发调试方便，降低运维成本

### 代码实现

```python
# chroma_store.py
import chromadb

class ChromaDB:
    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
    
    def search(self, query_vector, top_k=5):
        """向量检索"""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        candidates = []
        for i, doc in enumerate(results["documents"][0]):
            candidates.append({
                "chunk": doc,
                "score": 1 - results["distances"][0][i],  # 距离转相似度
                "metadata": results["metadatas"][0][i]
            })
        return candidates
```

### 向量模型选型

| 模型 | 维度 | 提供商 | 特点 |
|------|------|--------|------|
| **text-embedding-v2** | 1536 | 阿里云 | 中文效果好、成本低 |
| **text-embedding-3-small** | 1536 | OpenAI | 多语言、高性能 |
| **bge-large-zh** | 1024 | BGE | 开源、中文优化 |
| **m3e-base** | 768 | M3E | 轻量、本地部署 |

**本项目选择：阿里云 text-embedding-v2**

选择理由：
1. 中文语义效果好
2. 与通义千问同生态
3. API调用简单，成本低

---

## 步骤3：BM25 关键词检索（Lexical Search）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **BM25** | 经典算法、精确匹配强 | 无语义理解 | 关键词精确匹配 |
| **TF-IDF** | 简单、快速 | 效果不如BM25 | 基础检索 |
| **Elasticsearch** | 分布式、功能丰富 | 部署复杂 | 企业级搜索 |
| **Whoosh** | 纯Python、轻量 | 性能一般 | 小规模 |

**本项目选择：rank_bm25**

选择理由：
1. 经典算法，关键词匹配效果好
2. 纯Python实现，无需额外服务
3. 与向量检索互补，提升召回

### 代码实现

```python
# retriever.py
from rank_bm25 import BM25Okapi
import jieba

class BM25Retriever:
    def __init__(self, chunks, doc_name="未知文档"):
        self.chunks = chunks
        self.doc_name = doc_name
        # 中文分词
        tokenized = [list(jieba.cut(c)) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query, top_k=5):
        """BM25检索"""
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        
        # 获取top_k索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                "chunk": self.chunks[idx],
                "score": scores[idx],
                "source": "关键词检索(BM25)",
                "doc_name": self.doc_name
            })
        return candidates
```

---

## 步骤4：混合检索（Hybrid Search）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **加权融合** | 简单可控、可调参 | 需要调权重 | 通用场景 |
| **RRF (Reciprocal Rank Fusion)** | 无需调参、效果好 | 忽略分数绝对值 | 排序融合 |
| **LLM重排** | 效果最好 | 成本高、延迟大 | 高精度需求 |
| **学习到排序(LTR)** | 自动学习权重 | 需要训练数据 | 大规模 |

**本项目选择：加权融合 + 分数归一化**

选择理由：
1. 简单可控，权重可动态调整
2. 分数归一化消除量纲差异
3. 支持阈值过滤，控制质量

### 代码实现

```python
# retriever.py
def hybrid_search(db, bm25, question, top_k=5, vector_weight=0.6, bm25_weight=0.4):
    """混合检索：向量 + BM25"""
    
    # 1. 向量检索
    q_vec = get_embedding(question)
    vector_candidates = db.search(q_vec, top_k=top_k)
    
    # 2. BM25检索
    bm25_candidates = bm25.search(question, top_k=top_k)
    
    # 3. 分数归一化（消除量纲差异）
    def normalize_scores(scores):
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    norm_vector = normalize_scores([c["score"] for c in vector_candidates])
    norm_bm25 = normalize_scores([c["score"] for c in bm25_candidates])
    
    for i, c in enumerate(vector_candidates):
        c["normalized_score"] = norm_vector[i]
        c["weighted_score"] = norm_vector[i] * vector_weight
    
    for i, c in enumerate(bm25_candidates):
        c["normalized_score"] = norm_bm25[i]
        c["weighted_score"] = norm_bm25[i] * bm25_weight
    
    # 4. 合并 + 加权排序
    all_candidates = vector_candidates + bm25_candidates
    all_candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
    
    # 5. 去重
    seen = set()
    unique = []
    for c in all_candidates:
        if c["chunk"] not in seen:
            seen.add(c["chunk"])
            unique.append(c)
    
    return unique[:top_k]
```

---

## 步骤5：Rerank 精排序（Re-ranking）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Cross-Encoder Rerank** | 精度高、语义理解强 | 计算成本高 | 高精度需求 |
| **ColBERT** | 细粒度交互、效果好 | 部署复杂 | 学术研究 |
| **LLM Rerank** | 效果最好、可解释 | 成本最高 | 关键场景 |
| **学习排序(LTR)** | 自动学习 | 需要训练数据 | 大规模 |
| **无Rerank** | 速度快 | 精度受限 | 低延迟需求 |

**本项目选择：gte-rerank-v2 (Cross-Encoder)**

选择理由：
1. 阿里云DashScope提供，与向量模型同生态
2. Cross-Encoder架构，Query和Document联合编码
3. 中文效果好，延迟可接受（~100ms）
4. API调用简单，无需本地部署

### Cross-Encoder vs Bi-Encoder

```
Bi-Encoder（向量检索）:
Query → Encoder → Vector_Q
Doc  → Encoder → Vector_D
Score = cosine(Vector_Q, Vector_D)

Cross-Encoder（Rerank）:
Score = Encoder(Query, Doc)  # 联合编码，捕获交互
```

Cross-Encoder精度更高，因为Query和Document可以充分交互。

### 代码实现

```python
# retriever.py
def rerank(question, candidates):
    """Rerank精排序"""
    if not candidates:
        return []
    
    # 准备文档列表
    documents = [c["chunk"] for c in candidates]
    
    # 调用Rerank API
    response = requests.post(RERANK_URL, json={
        "model": "gte-rerank-v2",
        "input": {
            "query": question,
            "documents": documents
        },
        "parameters": {"return_documents": True}
    })
    
    results = response.json()["output"]["results"]
    
    # 重构结果
    reranked = []
    for item in results:
        doc = item["document"]
        score = item["relevance_score"]
        
        # 找到原始候选的元数据
        meta = next((c for c in candidates if c["chunk"] == doc), {})
        
        reranked.append({
            "chunk": doc,
            "score": score,
            "source": meta.get("source"),
            "doc_name": meta.get("doc_name")
        })
    
    # 按分数排序 + 阈值过滤
    reranked.sort(key=lambda x: x["score"], reverse=True)
    filtered = [r for r in reranked if r["score"] >= config.RERANK_THRESHOLD]
    
    return filtered[:config.RERANK_TOP_K]
```

---

## 步骤6：上下文压缩（Context Compression）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **截断** | 简单快速 | 可能丢失信息 | 简单场景 |
| **LLM压缩** | 保留关键信息 | 增加LLM调用 | 高质量需求 |
| **滑动窗口** | 保留完整段落 | 可能超长 | 长文档 |
| **摘要提取** | 信息密度高 | 可能丢失细节 | 摘要场景 |

**本项目选择：智能拼接 + 长度控制**

选择理由：
1. 简单有效，无额外LLM成本
2. 按Rerank分数排序，优先高分内容
3. 动态可配置最大长度

### 代码实现

```python
# llm_utils.py
def compress_context(candidates, max_len=2000):
    """压缩上下文，控制长度"""
    context_parts = []
    total_len = 0
    
    for c in candidates:
        chunk = c["chunk"]
        doc_name = c.get("doc_name", "未知文档")
        score = c.get("score", 0)
        
        # 格式化片段
        part = f"【{doc_name}】(相关度:{score:.2f})\n{chunk}\n"
        
        # 检查长度
        if total_len + len(part) > max_len:
            break
        
        context_parts.append(part)
        total_len += len(part)
    
    return "\n---\n".join(context_parts)
```

---

## 步骤7：LLM 回答生成

### 技术选型分析

| 模型 | 提供商 | 优点 | 缺点 |
|------|--------|------|------|
| **通义千问** | 阿里云 | 中文好、成本低 | 英文一般 |
| **GPT-4** | OpenAI | 效果最好 | 成本高 |
| **Claude** | Anthropic | 长上下文 | 国内访问难 |
| **GLM-4** | 智谱 | 国产、效果好 | 生态较小 |
| **Llama 3** | Meta | 开源、可本地 | 需要GPU |

**本项目选择：通义千问 qwen-turbo**

选择理由：
1. 中文问答效果好
2. 成本低（~0.002元/千token）
3. 与向量模型、Rerank同生态
4. 国内访问稳定

### 代码实现

```python
# llm_utils.py
def llm_answer(question, context, history=""):
    """LLM生成回答"""
    
    prompt = f"""你是一个企业知识库助手，请根据以下知识库内容回答用户问题。

【知识库内容】
{context}

【历史对话】
{history}

【用户问题】
{question}

【回答要求】
1. 基于知识库内容回答，不要编造
2. 如果知识库没有相关信息，请诚实说明
3. 回答简洁明了，条理清晰
4. 可以引用具体条款或规定

请回答："""
    
    response = requests.post(LLM_URL, json={
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.1}  # 低温度，减少幻觉
    })
    
    return response.json()["output"]["text"]
```

---

## 步骤8：历史对话摘要（History Summarization）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **全量保留** | 信息完整 | 上下文超长 | 短对话 |
| **滑动窗口** | 简单 | 丢失早期信息 | 中等对话 |
| **LLM摘要** | 保留关键信息 | 增加LLM调用 | 长对话 |
| **向量摘要** | 无LLM成本 | 效果一般 | 成本敏感 |

**本项目选择：LLM摘要 + 保留最新N轮**

选择理由：
1. 长对话场景，历史可能很长
2. LLM摘要保留关键信息
3. 最新N轮完整保留，保证上下文连贯

### 代码实现

```python
# session_store.py
def summarize_history(messages):
    """使用LLM对历史对话进行摘要"""
    if len(messages) <= 4:
        return messages
    
    # 分离旧消息和最新消息
    old_messages = messages[:-4]  # 保留最新2轮（4条消息）
    recent_messages = messages[-4:]
    
    # 构建摘要提示
    dialog_text = ""
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            dialog_text += f"用户：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            dialog_text += f"助手：{msg.content}\n"
    
    prompt = f"""请精简总结以下对话的关键内容，只用一两句话概括，保留重要信息：

{dialog_text}"""
    
    # 调用LLM生成摘要
    summary = llm.invoke(prompt)
    
    # 重构历史：摘要 + 最新消息
    summary_message = HumanMessage(content=f"[历史对话摘要] {summary}")
    return [summary_message] + recent_messages
```

---

## 步骤9：Redis 会话存储（Session Storage）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Redis** | 高性能、支持过期、分布式 | 需要额外部署 | 生产环境 |
| **内存存储** | 简单、无依赖 | 重启丢失、不支持分布式 | 开发测试 |
| **数据库存储** | 持久化、可靠 | 性能较低 | 需要持久化 |
| **文件存储** | 简单、无依赖 | 性能差、并发问题 | 单机小规模 |

**本项目选择：Redis（支持降级到内存存储）**

选择理由：
1. 高性能，支持高并发读写
2. 支持 TTL 过期时间，自动清理旧会话
3. 支持分布式部署，多实例共享会话
4. 项目支持自动降级：Redis 不可用时使用内存存储

### 代码实现

```python
# session_store.py
import json
import redis
from langchain_core.messages import HumanMessage, AIMessage
from config import config, LLM_URL, NO_PROXY
import os
import requests

API_KEY = os.getenv("API_KEY")

# 连接本地/服务器 Redis
r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    decode_responses=True
)

# 把消息对象转可序列化列表
def serialize_messages(messages):
    res = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            res.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            res.append({"type": "ai", "content": msg.content})
    return res

# 反序列换回 LangChain 消息对象
def deserialize_messages(data):
    messages = []
    for item in data:
        if item["type"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages

# 获取会话历史
def get_session_history(session_id: str):
    key = f"chat:session:{session_id}"
    raw = r.get(key)
    if not raw:
        return []
    return deserialize_messages(json.loads(raw))

# 使用LLM对历史对话进行摘要
def summarize_history(messages):
    if len(messages) <= config.HISTORY_KEEP_ROUNDS * 2:
        return messages
    
    old_messages = messages[:-config.HISTORY_KEEP_ROUNDS * 2]
    dialog_text = ""
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            dialog_text += f"用户：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            dialog_text += f"助手：{msg.content}\n"
    
    if not dialog_text.strip():
        return messages
    
    prompt = f"""请精简总结以下对话的关键内容，只用一两句话概括，保留重要信息：
{dialog_text}"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.3}
    }
    
    try:
        resp = requests.post(LLM_URL, headers=headers, json=data, proxies=NO_PROXY, timeout=10)
        result = resp.json()
        summary = result.get("output", {}).get("text", "").strip()
        if not summary:
            return messages
        
        summary_message = HumanMessage(content=f"[历史对话摘要] {summary}")
        new_messages = [summary_message] + messages[-config.HISTORY_KEEP_ROUNDS * 2:]
        print(f"📝 历史对话已摘要: {len(messages)}条 → {len(new_messages)}条")
        return new_messages
    except Exception as e:
        print(f"⚠️ 摘要生成失败: {e}")
        return messages

# 保存会话历史
def save_session_history(session_id: str, messages, expire_seconds=86400*7):
    key = f"chat:session:{session_id}"
    
    # 历史摘要处理
    if config.ENABLE_HISTORY_SUMMARY and len(messages) > config.HISTORY_SUMMARY_THRESHOLD:
        messages = summarize_history(messages)
    
    data = json.dumps(serialize_messages(messages))
    r.set(key, data)
    r.expire(key, expire_seconds)
```

### 自动降级机制

```python
# main.py
try:
    from session_store import get_session_history, save_session_history
    REDIS_AVAILABLE = True
except ImportError:
    print("⚠️ session_store 未找到，使用内存存储")
    REDIS_AVAILABLE = False
    _memory_store = {}
    
    def get_session_history(session_id):
        return _memory_store.get(session_id, [])
    
    def save_session_history(session_id, messages):
        _memory_store[session_id] = messages
```

### Redis 数据结构

```
Key: chat:session:{session_id}
Value: [{"type": "human", "content": "..."}, {"type": "ai", "content": "..."}]
TTL: 7天（可配置）
```

---

## 步骤10：PDF 文档分块（Document Chunking）

### 技术选型分析

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **固定长度分块** | 简单、速度快 | 可能切断语义 | 通用场景 |
| **句子分块** | 语义完整 | 块大小不均匀 | 短文本 |
| **段落分块** | 语义完整、结构清晰 | 块可能过大 | 结构化文档 |
| **语义分块** | 语义连贯、效果好 | 计算成本高 | 高质量需求 |
| **递归字符分块** | 灵活、可配置 | 需要调参 | LangChain推荐 |
| **智能语义分块** | 识别标题层级、结构化 | 实现复杂 | **企业文档** |

**本项目选择：智能语义分块（Smart Semantic Split）**

选择理由：
1. 企业文档（员工手册、规章制度）有明确的章节结构
2. 识别章节标题（第X章、第X条），按层级分块
3. 保留语义完整性，避免切断关键信息
4. 支持动态参数：max_size、min_size 可配置

### 分块策略详解

```
文档结构识别：
├── 第X章（一级标题）→ 强制分块
├── 一、二、三、（二级标题）→ 达到min_size时分块
├── (1)(2)(3)（三级标题）→ 达到min_size时分块
└── 普通段落 → 超过max_size时分块
```

### 代码实现

```python
# pdf_utils.py
import re
from PyPDF2 import PdfReader
from config import config
from log_utils import log_step

def extract_pdf_text(pdf_path):
    """提取PDF文本"""
    full_text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        # 清理多余空白
        full_text = re.sub(r'[ \t]+', ' ', full_text)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = full_text.strip()
    except Exception as e:
        log_step("PDF读取", "error", f"读取失败: {e}")
    return full_text

def smart_semantic_split(text, max_size=config.MAX_SIZE, min_size=config.MIN_SIZE):
    """智能语义分块"""
    # 定义标题正则模式
    major_title_pat = re.compile(r"^(第[一二三四五六七八九十百]+章|第[一二三四五六七八九十百]+条)")
    minor_title_pat = re.compile(r"^([一二三四五六七八九十百]+、|\(\d+\)|\([一二三四五六七八九十]+\)|（\d+）|（[一二三四五六七八九十]+）|\d+\.|\d+、|[①②③④⑤⑥⑦⑧⑨⑩])")
    
    # 修复标题前没有换行的情况
    title_insert_pat = re.compile(r'([^\n\s])(第[一二三四五六七八九十百]+章|第[一二三四五六七八九十百]+条|[一二三四五六七八九十百]+、|\(\d+\)|\([一二三四五六七八九十]+\)|（\d+）|（[一二三四五六七八九十]+）|\d+\.|\d+、|[①②③④⑤⑥⑦⑧⑨⑩])')
    text = title_insert_pat.sub(r'\1\n\2', text)

    # 按行分割
    sentences = re.split(r'\n', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 按句号分句（保留标点）
    sentences2 = []
    for s in sentences:
        parts = re.split(r'([。；！？])', s)
        buf = ""
        for p in parts:
            buf += p
            if p in ('。', '；', '！', '？'):
                sentences2.append(buf)
                buf = ""
        if buf.strip():
            sentences2.append(buf)
    sentences = [s.strip() for s in sentences2 if s.strip()]

    # 智能分块
    chunks = []
    current = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if current:
            text_block = "".join(current)
            if len(text_block) >= min_size:
                chunks.append(text_block)
            elif chunks:
                chunks[-1] += text_block  # 过短的块合并到上一块
            else:
                chunks.append(text_block)
        current = []
        current_len = 0

    for sent in sentences:
        is_major = bool(major_title_pat.match(sent))  # 一级标题
        is_minor = bool(minor_title_pat.match(sent))  # 二级标题

        if is_major and current:
            flush_current()  # 一级标题强制分块
        elif is_minor and current_len >= min_size:
            flush_current()  # 二级标题达到最小长度分块
        elif not is_major and not is_minor and current_len + len(sent) > max_size:
            flush_current()  # 超过最大长度分块

        current.append(sent)
        current_len += len(sent)

    if current:
        flush_current()

    return [c for c in chunks if c.strip()]

def save_chunks_to_txt(chunks, save_path="chunks_查看全部.txt"):
    """保存分块结果到文件（调试用）"""
    with open(save_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            f.write("="*60 + "\n")
            f.write(f"【第 {idx+1} 块 | 字符数：{len(chunk)}】\n")
            f.write("="*60 + "\n")
            f.write(chunk)
            f.write("\n\n")
    log_step("保存分块", "info", f"已保存到：{save_path}，共 {len(chunks)} 块")
```

### 分块效果示例

**输入文档：**
```
第一章 总则
第一条 为规范公司管理，制定本手册。
第二条 本手册适用于全体员工。

第二章 考勤制度
第一条 上班时间为上午9:00。
第二条 迟到超过30分钟按旷工处理。
```

**输出分块：**
```
块1: 第一章 总则\n第一条 为规范公司管理，制定本手册。\n第二条 本手册适用于全体员工。
块2: 第二章 考勤制度\n第一条 上班时间为上午9:00。\n第二条 迟到超过30分钟按旷工处理。
```

---

## LangGraph 工作流编排

### 为什么选择 LangGraph？

| 框架 | 优点 | 缺点 |
|------|------|------|
| **LangGraph** | 状态管理、条件分支、可视化 | 学习成本 |
| **LangChain Chain** | 简单 | 不支持复杂流程 |
| **自研Pipeline** | 灵活 | 无生态支持 |

**选择理由：**
1. 支持复杂的有向图工作流
2. 内置状态管理（TypedDict）
3. 支持条件分支（conditional_edges）
4. 可视化调试

### 工作流图

```
START
  │
  ▼
init_database ──▶ get_history ──▶ permission_check
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼ (无权限)                              ▼ (有权限)
                   END                          agent_tool_call
                                                   │
                            ┌──────────────────────┴──────────────────────┐
                            │                                             │
                            ▼ (工具调用)                                  ▼ (知识检索)
                           END                                    multi_query
                                                                    │
                                                                    ▼
                                                            vector_search
                                                                    │
                                                                    ▼
                                                            bm25_search
                                                                    │
                                                                    ▼
                                                           merge_results
                                                                    │
                                                                    ▼
                                                           deduplicate
                                                                    │
                                                                    ▼
                                                              rerank
                                                                    │
                                                                    ▼
                                                        compress_context
                                                                    │
                                                                    ▼
                                                        generate_answer
                                                                    │
                                                                    ▼
                                                           save_history
                                                                    │
                                                                    ▼
                                                                   END
```

### 代码实现

```python
# main.py
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    messages: list
    session_id: str
    user_input: str
    db_handbook: object
    db_regulation: object
    queries: list
    vector_candidates: list
    bm25_candidates: list
    all_candidates: list
    unique_candidates: list
    candidates: list
    context: str
    answer: str
    end: bool

def build_workflow():
    graph = StateGraph(WorkflowState)
    
    # 添加节点
    graph.add_node("init_database", node_init_database)
    graph.add_node("get_history", node_get_history)
    graph.add_node("permission_check", node_permission_check)
    graph.add_node("agent_tool_call", node_agent_tool_call)
    graph.add_node("multi_query", node_multi_query)
    graph.add_node("vector_search", node_vector_search)
    graph.add_node("bm25_search", node_bm25_search)
    graph.add_node("merge_results", node_merge_results)
    graph.add_node("deduplicate", node_deduplicate)
    graph.add_node("rerank", node_rerank)
    graph.add_node("compress_context", node_compress_context)
    graph.add_node("generate_answer", node_generate_answer)
    graph.add_node("save_history", node_save_history)
    
    # 添加边
    graph.add_edge(START, "init_database")
    graph.add_edge("init_database", "get_history")
    graph.add_edge("get_history", "permission_check")
    
    # 条件分支
    graph.add_conditional_edges(
        "permission_check",
        lambda state: END if state.get("end") else "agent_tool_call"
    )
    
    graph.add_conditional_edges(
        "agent_tool_call",
        lambda state: END if state.get("end") else "multi_query"
    )
    
    # RAG流程
    graph.add_edge("multi_query", "vector_search")
    graph.add_edge("vector_search", "bm25_search")
    graph.add_edge("bm25_search", "merge_results")
    graph.add_edge("merge_results", "deduplicate")
    graph.add_edge("deduplicate", "rerank")
    graph.add_edge("rerank", "compress_context")
    graph.add_edge("compress_context", "generate_answer")
    graph.add_edge("generate_answer", "save_history")
    graph.add_edge("save_history", END)
    
    return graph.compile()
```

---

## 流式输出实现

### 技术选型

| 方案 | 优点 | 缺点 |
|------|------|------|
| **SSE (Server-Sent Events)** | 简单、HTTP原生 | 单向 |
| **WebSocket** | 双向、实时 | 复杂 |
| **长轮询** | 兼容性好 | 效率低 |

**本项目选择：SSE**

选择理由：
1. 单向推送足够（服务端→客户端）
2. FastAPI原生支持
3. 前端EventSource API简单

### 后端实现

```python
# api.py
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        # 设置日志回调
        def log_callback(log_data):
            yield f"data: {json.dumps({'type': 'log', 'data': log_data})}\n\n"
        
        set_log_callback(log_callback)
        
        # 执行工作流
        result = workflow.invoke(initial_state)
        
        # 发送最终答案
        yield f"data: {json.dumps({'type': 'answer', 'data': result['answer']})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 前端实现

```javascript
// KnowledgeAssistant.vue
const response = await fetch('/api/chat/stream', {
  method: 'POST',
  body: JSON.stringify({ message, session_id })
})

const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  
  const text = decoder.decode(value)
  const lines = text.split('\n')
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6))
      
      if (event.type === 'log') {
        // 实时显示思考过程
        logs.value.push(event.data)
      } else if (event.type === 'answer') {
        // 显示最终答案
        answer.value = event.data
      }
    }
  }
}
```

---

## Agent 工具调用

### 工具定义

```python
# tools.py
from langchain.tools import tool

@tool
def get_collection_count(collection_name: str) -> str:
    """查询表数量。collection_name 只能是 'handbook' 或 'regulation'。"""
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

@tool
def import_pdf_files() -> str:
    """导入PDF文件到知识库。"""
    batch_pdf_import()
    return "✅ PDF导入完成"
```

### Agent 构建

```python
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

tools = [get_collection_count, clear_collection, import_pdf_files]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是企业知识库助手，可以执行管理操作..."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm_qwen, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

---

## 动态配置系统

### 设计思路

传统配置方式（`from config import TOP_K`）导入的是值副本，运行时修改不生效。

本项目使用**配置类实例**，所有模块通过实例访问，实现动态更新。

### 实现

```python
# config.py
class Config:
    TOP_K = 3
    VECTOR_WEIGHT = 0.6
    RERANK_THRESHOLD = 0.3
    
    def to_dict(self):
        return {
            "TOP_K": self.TOP_K,
            "VECTOR_WEIGHT": self.VECTOR_WEIGHT,
            ...
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

config = Config()  # 全局单例

# 其他模块
from config import config
top_k = config.TOP_K  # 动态获取，修改立即生效
```

### API

```python
@app.get("/config")
async def get_config():
    return config.to_dict()

@app.post("/config")
async def update_config(config_data: ConfigUpdate):
    for key, value in config_data.dict(exclude_none=True).items():
        setattr(config, key, value)
    return {"message": "配置已更新"}
```

---

## 安装部署

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
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

---

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

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MAX_SIZE | 600 | 最大chunk长度 |
| MIN_SIZE | 50 | 最小chunk长度 |
| MAX_CONTEXT_LEN | 2000 | 最大上下文长度 |
| TOP_K | 3 | 检索返回数量 |
| VECTOR_WEIGHT | 0.6 | 向量检索权重 |
| BM25_WEIGHT | 0.4 | BM25检索权重 |
| RERANK_TOP_K | 5 | Rerank保留数量 |
| RERANK_THRESHOLD | 0 | Rerank阈值 |
| VECTOR_SEARCH_THRESHOLD | 0.5 | 向量检索阈值 |
| BM25_SEARCH_THRESHOLD | 0.5 | BM25检索阈值 |
| ENABLE_HISTORY_SUMMARY | True | 启用历史摘要 |
| HISTORY_SUMMARY_THRESHOLD | 6 | 摘要触发阈值 |
| HISTORY_KEEP_ROUNDS | 2 | 历史摘要保留轮数 |
| ENABLE_DANGER_OP | True | 启用危险操作 |

---

## 知识库管理

### PDF 文件导入

```python
from langchain_classic.utils import batch_pdf_import

batch_pdf_import()
```

### 表管理

```python
db = ChromaDB(collection_name="handbook")
db.load()
db.clear()
```

### 数据统计

```python
from langchain_classic.utils import get_collection_count

count = get_collection_count(collection_name="handbook")
```

---

## Redis 缓存

### 设计思路

---

## 项目特色

1. **混合检索**：向量检索 + BM25 关键词检索，互补提升召回率
2. **Rerank 精排序**：Cross-Encoder 架构，语义交互更充分
3. **多问句生成**：LLM 扩展查询语义，多角度检索
4. **上下文压缩**：智能拼接，控制 Token 消耗
5. **流式输出**：SSE 实时展示思考过程
6. **动态配置**：前端实时调整检索参数
7. **历史摘要**：LLM 自动压缩，节省上下文
8. **权限控制**：管理员/普通用户角色区分
9. **工作流编排**：LangGraph 实现复杂流程
10. **知识库管理**：PDF 导入、表管理、数据统计
11. **Redis 缓存**：提升检索速度，降低 API 负载
---

## License

MIT
