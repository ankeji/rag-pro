import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["LANGCHAIN_PROJECT"] = "rag-pro-unified"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import json
import queue

from config import config
from chroma_store import ChromaDB
from retriever import BM25Retriever, hybrid_search, rerank
from llm_utils import generate_multi_query, compress_context, llm_answer
from log_utils import log_step, clear_logs, get_logs, set_log_callback
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_community.chat_models import ChatTongyi
import time

KEY = os.getenv("KEY")
API_KEY = os.getenv("API_KEY")

try:
    from session_store import get_session_history, save_session_history
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    _memory_store = {}
    def get_session_history(session_id):
        return _memory_store.get(session_id, [])
    def save_session_history(session_id, messages):
        _memory_store[session_id] = messages

from tools import (
    get_collection_count, clear_collection, import_pdf_files,
    check_tool_permission, set_user_role
)

tools = [get_collection_count, clear_collection, import_pdf_files]

# llm_qwen = ChatOpenAI(
#     model="qwen-turbo", 
#     api_key=API_KEY, 
#     base_url=LLM_URL,
#     temperature=0.1,
#     max_tokens=500
# )

llm_qwen = ChatTongyi(model="qwen-turbo", api_key=API_KEY)

def _get_doc_name(db, fallback="未知文档"):
    if db.metadatas and len(db.metadatas) > 0:
        return (db.metadatas[0] or {}).get("doc_name", fallback)
    return fallback

class WorkflowState(TypedDict):
    messages: list
    session_id: str
    user_input: str
    db_handbook: object
    db_regulation: object
    bm25_handbook: object
    bm25_regulation: object
    queries: list
    vector_candidates: list
    bm25_candidates: list
    all_candidates: list
    unique_candidates: list
    candidates: list
    context: str
    answer: str
    rag_available: bool
    end: bool
    error: str

def node_init_database(state):
    log_step("检查加载数据库表")
    try:
        db_handbook = ChromaDB(collection_name="handbook")
        db_regulation = ChromaDB(collection_name="regulation")
        
        handbook_loaded = db_handbook.load()
        regulation_loaded = db_regulation.load()
        
        handbook_count = db_handbook.collection.count() if handbook_loaded else 0
        regulation_count = db_regulation.collection.count() if regulation_loaded else 0
        
        log_step("检查加载数据库表", "done", f"handbook({handbook_count}条) + regulation({regulation_count}条)")
        
        bm25_handbook = None
        bm25_regulation = None
        
        if handbook_loaded and handbook_count > 0:
            handbook_doc_name = _get_doc_name(db_handbook, "新员工手册")
            bm25_handbook = BM25Retriever(db_handbook.chunks, doc_name=handbook_doc_name)
        
        if regulation_loaded and regulation_count > 0:
            regulation_doc_name = _get_doc_name(db_regulation, "规章制度")
            bm25_regulation = BM25Retriever(db_regulation.chunks, doc_name=regulation_doc_name)
        
        rag_available = handbook_count > 0 or regulation_count > 0
        
        return {
            **state,
            "db_handbook": db_handbook,
            "db_regulation": db_regulation,
            "bm25_handbook": bm25_handbook,
            "bm25_regulation": bm25_regulation,
            "rag_available": rag_available
        }
    except Exception as e:
        log_step("检查加载数据库表", "error", str(e))
        return {**state, "answer": f"❌ 数据库初始化失败：{str(e)}", "error": str(e), "end": True}

def node_get_history(state):
    log_step("开始获取历史消息")
    try:
        session_id = state["session_id"]
        messages = get_session_history(session_id)
        history_count = len(messages)
        log_step("获取历史消息", "done", f"已获取{history_count}条历史消息")
        return {**state, "messages": messages}
    except Exception as e:
        log_step("获取历史消息", "error", str(e))
        return {**state, "messages": []}

def node_permission_check(state):
    user_input = state["user_input"]
    
    tool_to_call = None
    if "清空" in user_input:
        tool_to_call = "clear_collection"
    elif "导入" in user_input:
        tool_to_call = "import_pdf_files"
    
    if tool_to_call:
        log_step(f"权限校验", "info", f"检测到高危操作: {tool_to_call}")
        has_permission = check_tool_permission(tool_name=tool_to_call)
        
        from tools import current_user_role
        if not has_permission:
            log_step("权限校验", "error", f"用户[{current_user_role}]无权限执行[{tool_to_call}]")
            return {
                **state,
                "answer": f"❌ 权限不足！您无权限执行【{tool_to_call}】操作，请联系管理员。",
                "end": True
            }
        log_step("权限校验", "done", f"用户[{current_user_role}]有权限执行[{tool_to_call}]")
    
    return state

def node_multi_query(state):
    user_input = state["user_input"]
    log_step("生成多检索问句")
    try:
        queries = generate_multi_query(user_input)
        log_step("生成多检索问句", "done", f"生成{len(queries)}个检索问句")
        log_step("检索问句", "info", str(queries))
        return {**state, "queries": queries}
    except Exception as e:
        log_step("生成多检索问句", "error", str(e))
        return {**state, "queries": [user_input]}

def node_vector_search(state):
    log_step("开始向量检索")
    try:
        db_handbook = state["db_handbook"]
        db_regulation = state["db_regulation"]
        bm25_handbook = state.get("bm25_handbook")
        bm25_regulation = state.get("bm25_regulation")
        queries = state.get("queries", [state["user_input"]])
        
        all_candidates = []
        for i, query in enumerate(queries):
            log_step(f"向量检索问句{i+1}", "info", query[:30] + "...")
            if bm25_handbook:
                candidates_handbook = hybrid_search(
                    db_handbook, bm25_handbook, query,
                    top_k=config.TOP_K, vector_weight=config.VECTOR_WEIGHT, bm25_weight=config.BM25_WEIGHT
                )
                all_candidates.extend(candidates_handbook)
            if bm25_regulation:
                candidates_regulation = hybrid_search(
                    db_regulation, bm25_regulation, query,
                    top_k=config.TOP_K, vector_weight=config.VECTOR_WEIGHT, bm25_weight=config.BM25_WEIGHT
                )
                all_candidates.extend(candidates_regulation)
        
        log_step("向量检索", "done", f"检索到{len(all_candidates)}个候选")
        return {**state, "vector_candidates": all_candidates}
    except Exception as e:
        log_step("向量检索", "error", str(e))
        return {**state, "vector_candidates": []}

def node_bm25_search(state):
    log_step("开始BM25关键词检索")
    try:
        db_handbook = state["db_handbook"]
        db_regulation = state["db_regulation"]
        bm25_handbook = state.get("bm25_handbook")
        bm25_regulation = state.get("bm25_regulation")
        queries = state.get("queries", [state["user_input"]])
        
        all_candidates = []
        for i, query in enumerate(queries):
            log_step(f"BM25检索问句{i+1}", "info", query[:30] + "...")
            if bm25_handbook:
                candidates_handbook = hybrid_search(
                    db_handbook, bm25_handbook, query,
                    top_k=config.TOP_K, vector_weight=config.VECTOR_WEIGHT, bm25_weight=config.BM25_WEIGHT
                )
                all_candidates.extend(candidates_handbook)
            if bm25_regulation:
                candidates_regulation = hybrid_search(
                    db_regulation, bm25_regulation, query,
                    top_k=config.TOP_K, vector_weight=config.VECTOR_WEIGHT, bm25_weight=config.BM25_WEIGHT
                )
                all_candidates.extend(candidates_regulation)
        
        log_step("BM25关键词检索", "done", f"检索到{len(all_candidates)}个候选")
        return {**state, "bm25_candidates": all_candidates}
    except Exception as e:
        log_step("BM25关键词检索", "error", str(e))
        return {**state, "bm25_candidates": []}

def node_merge_results(state):
    log_step("开始合并向量检索结果和BM25关键词检索结果")
    try:
        vector_candidates = state.get("vector_candidates", [])
        bm25_candidates = state.get("bm25_candidates", [])
        
        all_candidates = vector_candidates + bm25_candidates
        log_step("合并检索结果", "done", f"合并后共{len(all_candidates)}个候选")
        return {**state, "all_candidates": all_candidates}
    except Exception as e:
        log_step("合并检索结果", "error", str(e))
        return {**state, "all_candidates": []}

def node_deduplicate(state):
    log_step("开始对结果进行去重")
    try:
        all_candidates = state.get("all_candidates", [])
        
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            chunk = c.get("chunk", "")
            if chunk not in seen:
                seen.add(chunk)
                unique_candidates.append(c)
        
        log_step("结果去重", "done", f"去重后剩余{len(unique_candidates)}个候选")
        return {**state, "unique_candidates": unique_candidates}
    except Exception as e:
        log_step("结果去重", "error", str(e))
        return {**state, "unique_candidates": []}

def node_rerank(state):
    log_step("开始使用Rerank精排序")
    try:
        user_input = state["user_input"]
        unique_candidates = state.get("unique_candidates", [])
        
        if not unique_candidates:
            log_step("Rerank精排序", "info", "无候选结果，跳过")
            return {**state, "candidates": []}
        
        candidates = rerank(user_input, unique_candidates)
        log_step("Rerank精排序", "done", f"排序完成，保留前{len(candidates)}个结果")
        
        for i, c in enumerate(candidates[:3]):
            doc_name = c.get("doc_name", "未知文档")
            score = c.get("score", 0)
            log_step(f"Top{i+1}结果", "info", f"文档:{doc_name} 得分:{score:.4f}")
        
        return {**state, "candidates": candidates}
    except Exception as e:
        log_step("Rerank精排序", "error", str(e))
        return {**state, "candidates": state.get("unique_candidates", [])}

def node_compress_context(state):
    log_step("开始压缩上下文")
    try:
        candidates = state.get("candidates", [])
        
        if not candidates:
            log_step("压缩上下文", "info", "无候选结果，上下文为空")
            return {**state, "context": ""}
        
        ctx = compress_context(candidates, max_len=2000)
        log_step("压缩上下文", "done", f"上下文长度:{len(ctx)}字符")
        return {**state, "context": ctx}
    except Exception as e:
        log_step("压缩上下文", "error", str(e))
        return {**state, "context": ""}

def node_generate_answer(state):
    log_step("开始交给模型生成回答")
    try:
        user_input = state["user_input"]
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        history_ctx = ""
        if messages:
            for msg in messages[-4:]:
                if isinstance(msg, HumanMessage):
                    history_ctx += f"用户：{msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_ctx += f"助手：{msg.content}\n"
        
        answer = llm_answer(user_input, context, history_ctx)
        log_step("模型生成回答", "done", f"回答长度:{len(answer)}字符")
        return {**state, "answer": answer}
    except Exception as e:
        log_step("模型生成回答", "error", str(e))
        return {**state, "answer": f"生成回答失败：{str(e)}"}

def node_save_history(state):
    log_step("开始Redis缓存历史")
    try:
        session_id = state["session_id"]
        user_input = state["user_input"]
        answer = state.get("answer", "")
        messages = state.get("messages", [])
        
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=answer))
        save_session_history(session_id, messages)
        
        storage_type = "Redis" if REDIS_AVAILABLE else "内存"
        log_step("Redis缓存历史", "done", f"已保存到{storage_type}，共{len(messages)}条消息")
        return {**state, "messages": messages}
    except Exception as e:
        log_step("Redis缓存历史", "error", str(e))
        return state

def node_agent_tool_call(state):
    user_input = state["user_input"]
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识库助手，可以执行以下管理操作：

【工具调用规则】
1. get_collection_count(collection_name)：查询数据表的数量
   - 当用户询问"表数量"、"数据量"、"有多少条"、"表里有多少数据"时调用
   - collection_name 只能是 'handbook' 或 'regulation'
   - 如果用户没有指定表名，默认查询 handbook

2. clear_collection(collection_name)：清空数据表（高危操作，需管理员权限）
   - 当用户说"清空xxx表"时调用
   - collection_name 只能是 'handbook' 或 'regulation'

3. import_pdf_files()：导入PDF文件到知识库
   - 当用户说"导入PDF"、"导入文件"时调用
   - 无需参数

【重要】
- 对于知识库内容查询（如"年假有多少天"、"报销流程"等），不要调用任何工具
- 只有管理操作才调用工具"""),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm_qwen, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    
    try:
        res = agent_executor.invoke({"input": user_input})
        answer = res.get("output", "无输出")
        
        tool_calls = res.get("intermediate_steps", [])
        if tool_calls:
            log_step("检测到工具调用意图", "info", f"调用了 {len(tool_calls)} 个工具")
            log_step("Agent工具调用", "done", "调用成功")
            return {**state, "answer": answer, "end": True}
        
        tool_keywords = ["表数量", "数据量", "有多少条", "清空", "导入", "表数据"]
        is_tool_intent = any(kw in user_input for kw in tool_keywords)
        
        if is_tool_intent:
            log_step("Agent工具调用", "done", "关键词匹配，返回结果")
            return {**state, "answer": answer, "end": True}
        
        return {**state, "end": False}
    except Exception as e:
        log_step("Agent工具调用", "error", str(e))
        return {**state, "answer": f"工具调用失败：{str(e)}", "end": True}

def build_workflow():
    graph = StateGraph(WorkflowState)
    
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
    
    graph.add_edge(START, "init_database")
    graph.add_edge("init_database", "get_history")
    graph.add_edge("get_history", "permission_check")
    
    graph.add_conditional_edges(
        "permission_check",
        lambda state: END if state.get("end") else "agent_tool_call"
    )
    
    graph.add_conditional_edges(
        "agent_tool_call",
        lambda state: END if state.get("end") else "multi_query"
    )
    
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

workflow = build_workflow()

app = FastAPI(title="企业知识库助手 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    logs: List[dict] = []
    error: Optional[str] = None

class RoleRequest(BaseModel):
    role: str

class HistoryResponse(BaseModel):
    messages: List[dict]
    session_id: str

@app.get("/")
async def root():
    return {"message": "企业知识库助手 API", "status": "running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        clear_logs()
        SESSION_ID = request.session_id
        user_input = request.message
        initial_state = {
            "messages": [],
            "session_id": SESSION_ID,
            "user_input": user_input,
            "db_handbook": None,
            "db_regulation": None,
            "bm25_handbook": None,
            "bm25_regulation": None,
            "queries": [],
            "vector_candidates": [],
            "bm25_candidates": [],
            "all_candidates": [],
            "unique_candidates": [],
            "candidates": [],
            "context": "",
            "answer": "",
            "rag_available": False,
            "end": False,
            "error": ""
        }
        
        final_state = workflow.invoke(initial_state)
        logs = get_logs()
        
        return ChatResponse(
            answer=final_state.get("answer", "无回答"),
            session_id=request.session_id,
            logs=logs,
            error=final_state.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    log_queue = queue.Queue()
    
    def log_callback(log_entry):
        log_queue.put(log_entry)
    
    def generate():
        try:
            clear_logs()
            set_log_callback(log_callback)
            
            SESSION_ID = request.session_id
            user_input = request.message
            initial_state = {
                "messages": [],
                "session_id": SESSION_ID,
                "user_input": user_input,
                "db_handbook": None,
                "db_regulation": None,
                "bm25_handbook": None,
                "bm25_regulation": None,
                "queries": [],
                "vector_candidates": [],
                "bm25_candidates": [],
                "all_candidates": [],
                "unique_candidates": [],
                "candidates": [],
                "context": "",
                "answer": "",
                "rag_available": False,
                "end": False,
                "error": ""
            }
            
            def run_workflow():
                return workflow.invoke(initial_state)
            
            import threading
            result = [None]
            error = [None]
            
            def workflow_thread():
                try:
                    result[0] = workflow.invoke(initial_state)
                except Exception as e:
                    error[0] = str(e)
            
            thread = threading.Thread(target=workflow_thread)
            thread.start()
            
            while thread.is_alive():
                try:
                    log_entry = log_queue.get(timeout=0.1)
                    yield f"data: {json.dumps({'type': 'log', 'data': log_entry})}\n\n"
                except queue.Empty:
                    continue
            
            while not log_queue.empty():
                log_entry = log_queue.get()
                yield f"data: {json.dumps({'type': 'log', 'data': log_entry})}\n\n"
            
            thread.join()
            
            if error[0]:
                yield f"data: {json.dumps({'type': 'error', 'data': error[0]})}\n\n"
            else:
                final_state = result[0]
                answer = final_state.get("answer", "无回答")
                yield f"data: {json.dumps({'type': 'answer', 'data': answer})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        finally:
            set_log_callback(None)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/role")
async def set_role(request: RoleRequest):
    success = set_user_role(request.role)
    if success:
        return {"message": f"已切换为 {request.role} 角色", "role": request.role}
    raise HTTPException(status_code=400, detail=f"无效的角色: {request.role}")

@app.get("/role")
async def get_role():
    from tools import current_user_role
    return {"role": current_user_role}

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    messages = get_session_history(session_id)
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
    return HistoryResponse(messages=formatted_messages, session_id=session_id)

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    save_session_history(session_id, [])
    return {"message": f"已清空会话 {session_id} 的历史记录"}

@app.get("/stats")
async def get_stats():
    try:
        db_handbook = ChromaDB(collection_name="handbook")
        db_regulation = ChromaDB(collection_name="regulation")
        
        handbook_loaded = db_handbook.load()
        regulation_loaded = db_regulation.load()
        
        handbook_count = db_handbook.collection.count() if handbook_loaded else 0
        regulation_count = db_regulation.collection.count() if regulation_loaded else 0
        
        return {
            "handbook_count": handbook_count,
            "regulation_count": regulation_count,
            "total_count": handbook_count + regulation_count,
            "rag_available": handbook_count > 0 or regulation_count > 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    from config import config
    return config.to_dict()

class ConfigUpdate(BaseModel):
    MAX_SIZE: Optional[int] = None
    MIN_SIZE: Optional[int] = None
    MAX_CONTEXT_LEN: Optional[int] = None
    TOP_K: Optional[int] = None
    VECTOR_WEIGHT: Optional[float] = None
    BM25_WEIGHT: Optional[float] = None
    RERANK_TOP_K: Optional[int] = None
    RERANK_THRESHOLD: Optional[float] = None
    VECTOR_SEARCH_THRESHOLD: Optional[float] = None
    BM25_SEARCH_THRESHOLD: Optional[float] = None
    ENABLE_HISTORY_SUMMARY: Optional[bool] = None
    HISTORY_SUMMARY_THRESHOLD: Optional[int] = None
    HISTORY_KEEP_ROUNDS: Optional[int] = None
    ENABLE_DANGER_OP: Optional[bool] = None

@app.post("/config")
async def update_config(config_data: ConfigUpdate):
    from config import config
    
    updates = []
    for key, value in config_data.dict(exclude_none=True).items():
        if value is not None:
            setattr(config, key, value)
            updates.append(f"{key}={value}")
    
    log_step("更新配置", "done", ", ".join(updates))
    return {"message": "配置已更新", "updates": updates}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
