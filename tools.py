from chroma_store import ChromaDB, get_all_collections
from langchain.tools import tool
from pdf_import_script import batch_pdf_import
from config import config, ENABLE_DANGER_OP
from embedding_utils import get_embedding
from retriever import BM25Retriever, hybrid_search, rerank
from llm_utils import compress_context, llm_answer
import traceback


# ================= 权限配置 =================
user_roles = {
    "admin": ["get_collection_count", "clear_collection", "import_pdf_files", "query_knowledge_base"],
    "user": ["get_collection_count", "query_knowledge_base"]
}

current_user_role = "user"

def check_tool_permission(tool_name: str, user_role: str = None) -> bool:
    """判断当前用户角色是否有权限调用指定工具"""
    role = user_role or current_user_role
    allowed_tools = user_roles.get(role, [])
    return tool_name in allowed_tools

def set_user_role(role: str):
    """设置当前用户角色"""
    global current_user_role
    if role in user_roles:
        current_user_role = role
        return True
    return False

# ================= 工具1：查询表数量 =================
@tool
def get_collection_count(collection_name: str = None) -> str:
    """查询表数量。
    1. 如果不传参数，返回所有表的数量
    2. 如果传入 collection_name，返回指定表的数量
    """
    try:
        if collection_name:
            db = ChromaDB(collection_name=collection_name)
            count = db.collection.count()
            return f"表 {collection_name} 数据量：{count}"
        else:
            collection_names = get_all_collections()
            if not collection_names:
                return "❌ 未找到任何数据表，请先运行 pdf_import_script.py 导入文档"
            
            results = []
            for name in collection_names:
                db = ChromaDB(collection_name=name)
                count = db.collection.count()
                if count > 0:
                    results.append(f"表 {name} 数据量：{count}")
            
            if not results:
                return "❌ 所有数据表都为空，请先运行 pdf_import_script.py 导入文档"
            
            return "\n".join(results)
    except Exception as e:
        return f"查询失败：{str(e)}"

# ================= 工具2：清空表 =================
@tool
def clear_collection(collection_name: str) -> str:
    """清空指定的数据表。参数 collection_name 为要清空的表名。如果用户说清空所有表，请传入 'all'。"""
    if not ENABLE_DANGER_OP:
        return "❌ 系统已禁用高危清空操作"

    try:
        if collection_name == "all":
            collection_names = get_all_collections()
            results = []
            for name in collection_names:
                db = ChromaDB(collection_name=name)
                count = db.collection.count()
                if count > 0:
                    db.clear()
                    results.append(f"✅ 表 {name} 已清空")
            if not results:
                return "❌ 所有数据表都为空，无需清空"
            return "\n".join(results)
        else:
            db = ChromaDB(collection_name=collection_name)
            count = db.collection.count()
            if count == 0:
                return f"⚠️ 表 {collection_name} 已经为空"
            db.clear()
            return f"✅ 表 {collection_name} 已清空"
    except Exception as e:
        return f"❌ 清空失败：{str(e)}"

# ================= 工具3：导入PDF =================
@tool
def import_pdf_files() -> str:
    """把docs目录下的所有pdf文件导入到ChromaDB中"""
    try:
        batch_pdf_import()
        return "✅ PDF 文件已导入到 Chroma 数据库"
    except Exception as e:
        return f"导入失败：{str(e)}"

# ================= 工具4：查询知识库 =================
@tool
def query_knowledge_base(question: str) -> str:
    """
    从知识库中查询相关信息并回答问题。
    参数：
        question: 用户的问题
    注意：如果知识库为空，会返回错误提示。
    """
    try:
        print(f"🔍 开始查询知识库：{question}")
        
        collection_names = get_all_collections()
        if not collection_names:
            return "❌ 知识库为空，无法进行查询。请先使用「导入PDF」功能导入数据。"
        
        def _get_doc_name(db, fallback="未知文档"):
            if db.metadatas and len(db.metadatas) > 0:
                return (db.metadatas[0] or {}).get("doc_name", fallback)
            return fallback
        
        all_candidates = []
        
        for collection_name in collection_names:
            db = ChromaDB(collection_name=collection_name)
            if db.load():
                count = db.collection.count()
                if count > 0:
                    doc_name = _get_doc_name(db, collection_name)
                    bm25 = BM25Retriever(db.chunks, doc_name=doc_name)
                    candidates = hybrid_search(
                        db, bm25, question,
                        top_k=config.TOP_K,
                        vector_weight=config.VECTOR_WEIGHT,
                        bm25_weight=config.BM25_WEIGHT
                    )
                    all_candidates.extend(candidates)
                    print(f"📚 {collection_name} 检索到 {len(candidates)} 条结果")
        
        # 去重
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c["chunk"] not in seen:
                seen.add(c["chunk"])
                unique_candidates.append(c)
        
        print(f"📄 检索到 {len(unique_candidates)} 个相关片段")
        
        if not unique_candidates:
            return "❌ 没有找到相关信息"
        
        # 重排
        candidates = rerank(question, unique_candidates)
        
        # 压缩上下文
        ctx = compress_context(candidates, max_len=2000)
        
        print(f"📝 生成回答...")
        
        # 调用 LLM 生成答案
        ans = llm_answer(question, ctx, "")
        
        # 添加引用信息
        doc_refs = []
        for i, cand in enumerate(candidates[:3]):  # 只引用前3个
            doc_name = cand.get("doc_name", "未知文档")
            doc_refs.append(f"{i+1}（{doc_name}）")
        
        if doc_refs:
            ans += f"\n\n参考来源：{'、'.join(doc_refs)}"
        
        return ans
        
    except Exception as e:
        print(f"❌ 知识库查询失败：{str(e)}")
        print(traceback.format_exc())
        return f"知识库查询失败：{str(e)}"