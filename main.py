import os
import traceback

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["LANGCHAIN_PROJECT"] = "rag-pro-unified"

from dotenv import load_dotenv
load_dotenv()

from config import config
from chroma_store import ChromaDB
from retriever import BM25Retriever, hybrid_search, rerank
from llm_utils import generate_multi_query, compress_context, llm_answer
from log_utils import log_step
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_community.chat_models import ChatTongyi

KEY = os.getenv("KEY")
API_KEY = os.getenv("API_KEY")

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

from tools import (
    get_collection_count, clear_collection, import_pdf_files,
    check_tool_permission, set_user_role
)

# ================= 公共组件 =================
tools = [get_collection_count, clear_collection, import_pdf_files]

# llm_glm = ChatOpenAI(
#     model="glm-5.1", 
#     api_key=KEY, 
#     base_url=BASE_URL.replace("/chat/completions", ""),
#     temperature=0.1
# )

llm_qwen = ChatTongyi(model="qwen-turbo", api_key=API_KEY)

def _get_doc_name(db, fallback="未知文档"):
    if db.metadatas and len(db.metadatas) > 0:
        return (db.metadatas[0] or {}).get("doc_name", fallback)
    return fallback

# ================= LangGraph 工作流节点 =================
class WorkflowState(TypedDict):
    messages: list
    session_id: str
    user_input: str
    collections: dict
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
    """节点1：初始化数据库（动态加载所有 collection）"""
    log_step("检查加载数据库表")
    try:
        from chroma_store import get_all_collections
        
        collection_names = get_all_collections()
        if not collection_names:
            log_step("检查加载数据库表", "warning", "未找到任何 collection，请先运行 pdf_import_script.py 导入文档")
            return {
                **state,
                "collections": {},
                "rag_available": False
            }
        
        collections = {}
        total_count = 0
        collection_info = []
        
        for collection_name in collection_names:
            db = ChromaDB(collection_name=collection_name)
            if db.load():
                count = db.collection.count()
                if count > 0:
                    doc_name = _get_doc_name(db, collection_name)
                    bm25 = BM25Retriever(db.chunks, doc_name=doc_name)
                    collections[collection_name] = {
                        "db": db,
                        "bm25": bm25,
                        "doc_name": doc_name,
                        "count": count
                    }
                    total_count += count
                    collection_info.append(f"{collection_name}({count}条)")
        
        if collection_info:
            log_step("检查加载数据库表", "done", " + ".join(collection_info))
        else:
            log_step("检查加载数据库表", "warning", "所有 collection 都为空")
        
        rag_available = total_count > 0
        
        return {
            **state,
            "collections": collections,
            "rag_available": rag_available
        }
    except Exception as e:
        log_step("检查加载数据库表", "error", str(e))
        return {**state, "answer": f"❌ 数据库初始化失败：{str(e)}", "error": str(e), "end": True}

def node_get_history(state):
    """节点2：获取历史消息"""
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
    """节点3：权限校验"""
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
    """节点4：生成多检索问句"""
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
    """节点5：向量检索"""
    log_step("开始向量检索")
    try:
        collections = state.get("collections", {})
        queries = state.get("queries", [state["user_input"]])
        
        all_candidates = []
        for i, query in enumerate(queries):
            log_step(f"向量检索问句{i+1}", "info", query[:30] + "...")
            for collection_name, col_data in collections.items():
                db = col_data["db"]
                bm25 = col_data.get("bm25")
                if bm25:
                    candidates = hybrid_search(
                        db, bm25, query,
                        top_k=config.TOP_K, vector_weight=config.VECTOR_WEIGHT, bm25_weight=config.BM25_WEIGHT
                    )
                    all_candidates.extend(candidates)
        
        log_step("向量检索", "done", f"检索到{len(all_candidates)}个候选")
        return {**state, "vector_candidates": all_candidates}
    except Exception as e:
        log_step("向量检索", "error", str(e))
        return {**state, "vector_candidates": []}

def node_bm25_search(state):
    """节点6：BM25关键词检索"""
    log_step("开始BM25关键词检索")
    try:
        collections = state.get("collections", {})
        queries = state.get("queries", [state["user_input"]])
        
        all_candidates = []
        for i, query in enumerate(queries):
            log_step(f"BM25检索问句{i+1}", "info", query[:30] + "...")
            for collection_name, col_data in collections.items():
                db = col_data["db"]
                bm25 = col_data.get("bm25")
                if bm25:
                    candidates = hybrid_search(
                        db, bm25, query,
                        top_k=config.TOP_K, vector_weight=config.VECTOR_WEIGHT, bm25_weight=config.BM25_WEIGHT
                    )
                    all_candidates.extend(candidates)
        
        log_step("BM25关键词检索", "done", f"检索到{len(all_candidates)}个候选")
        return {**state, "bm25_candidates": all_candidates}
    except Exception as e:
        log_step("BM25关键词检索", "error", str(e))
        return {**state, "bm25_candidates": []}

def node_merge_results(state):
    """节点7：合并检索结果"""
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
    """节点8：结果去重"""
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
    """节点9：Rerank精排序"""
    log_step("开始使用Rerank精排序")
    try:
        user_input = state["user_input"]
        unique_candidates = state.get("unique_candidates", [])
        
        if not unique_candidates:
            log_step("Rerank精排序", "info", "无候选结果，跳过")
            return {**state, "candidates": []}
        
        candidates = rerank(user_input, unique_candidates)[:config.RERANK_TOP_K]
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
    """节点10：压缩上下文"""
    log_step("开始压缩上下文")
    try:
        candidates = state.get("candidates", [])
        
        if not candidates:
            log_step("压缩上下文", "info", "无候选结果，上下文为空")
            return {**state, "context": ""}
        
        ctx = compress_context(candidates, max_len=config.MAX_CONTEXT_LEN)
        log_step("压缩上下文", "done", f"上下文长度:{len(ctx)}字符")
        return {**state, "context": ctx}
    except Exception as e:
        log_step("压缩上下文", "error", str(e))
        return {**state, "context": ""}

def node_generate_answer(state):
    """节点11：模型生成回答"""
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
    """节点12：Redis缓存历史"""
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
    """节点：Agent工具调用"""
    user_input = state["user_input"]
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识库助手，可以执行以下管理操作：

【工具调用规则】
1. get_collection_count(collection_name)：查询数据表的数量
   - 当用户询问"表数量"、"数据量"、"有多少条"、"表里有多少数据"时调用
   - 如果不传参数，返回所有表的数量
   - 如果传入 collection_name，返回指定表的数量

2. clear_collection(collection_name)：清空数据表（高危操作，需管理员权限）
   - 当用户说"清空xxx表"时调用
   - 如果用户说清空所有表，请传入 'all'

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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    try:
        res = agent_executor.invoke({"input": user_input})
        answer = res.get("output", "无输出")
        
        print(f"[DEBUG] res keys: {res.keys()}")
        print(f"[DEBUG] intermediate_steps: {res.get('intermediate_steps', 'NOT FOUND')}")
        
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

# ================= 构建工作流 =================
def build_workflow():
    """构建完整的LangGraph工作流"""
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

# ================= 主函数 =================
def main():
    print("\n" + "=" * 60)
    print("🚀 企业知识库助手 - RAG + Agent + 工作流 完整版")
    print("=" * 60)
    print("功能特性：")
    print("  ✅ RAG 检索（向量 + BM25 混合检索 + 去重 + Ranker重排 + 上下文压缩 + 回答生成 + Redis缓存历史会话 + 权限校验）")
    print("  ✅ Agent 工具调用（查表、清空、导入）")
    print("  ✅ LangGraph 工作流编排")
    print("  ✅ Redis 会话存储")
    print("  ✅ 权限校验")
    print("  ✅ 详细执行日志")
    print("=" * 60)
    
    workflow = build_workflow()
    SESSION_ID = "user_default"
    
    print("\n💡 使用说明：")
    print("  - 输入问题进行知识库检索")
    print("  - 输入 '清空表名' 执行工具调用（如：清空handbook）")
    print("  - 输入 '/role admin/user' 切换角色")
    print("  - 输入 'q' 退出")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n💬 请输入问题：").strip()
            
            if user_input.lower() == 'q':
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            if user_input.startswith("/role "):
                new_role = user_input.split()[1]
                set_user_role(new_role)
                print(f"✅ 已切换为 {new_role} 角色")
                continue
            
            print("\n" + "─" * 60)
            print(f"📋 开始处理: {user_input}")
            print("─" * 60)
            
            initial_state = {
                "messages": [],
                "session_id": SESSION_ID,
                "user_input": user_input,
                "collections": {},
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
            
            print("\n" + "=" * 60)
            print(f"✅ 最终回答：")
            print("=" * 60)
            print(final_state.get("answer", "无回答"))
            print("=" * 60)
            
            if final_state.get("error"):
                print(f"⚠️ 错误信息：{final_state['error']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 系统错误：{type(e).__name__}: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()