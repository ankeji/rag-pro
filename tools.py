'''
Author: ankeji ankeji1995@163.com
Date: 2026-05-11 10:31:32
LastEditors: ankeji ankeji1995@163.com
LastEditTime: 2026-05-15 15:58:25
FilePath: \AI_Projects\my_agent\rag-pro\tools.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from chroma_store import ChromaDB
from langchain.tools import tool
from pdf_import_script import batch_pdf_import
from config import config, ALLOW_COLLECTIONS, ENABLE_DANGER_OP
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
def get_collection_count(collection_name: str) -> str:
    """查询表数量。参数 collection_name 只能是 'handbook' 或 'regulation'。
    1. 如果用户说查询表数量，请分别调用两次。
        例如：查询 handbook 表数量，调用两次 get_collection_count(collection_name="handbook")
        例如：查询 regulation 表数量，调用两次 get_collection_count(collection_name="regulation")
    2. 如果表数量都为空，则分别返回"表 handbook 数据量：0"和"表 regulation 数据量：0"。
    """
    try:
        db = ChromaDB(collection_name=collection_name)
        return f"表 {collection_name} 数据量：{db.collection.count()}"
    except:
        return "查询失败"

# ================= 工具2：清空表 =================
@tool
def clear_collection(collection_name: str) -> str:
    """清空指定的数据表。参数 collection_name 只能是 'handbook' 或 'regulation'。如果用户说清空所有表，请分别调用两次。"""
    if not ENABLE_DANGER_OP:
        return "❌ 系统已禁用高危清空操作"

    if collection_name not in ALLOW_COLLECTIONS:
        return f"❌ 非法表名，仅允许操作：{ALLOW_COLLECTIONS}"

    try:
        db = ChromaDB(collection_name=collection_name)
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
        
        db_handbook = ChromaDB(collection_name="handbook")
        db_regulation = ChromaDB(collection_name="regulation")
        
        handbook_loaded = db_handbook.load()
        regulation_loaded = db_regulation.load()
        
        handbook_count = db_handbook.collection.count() if handbook_loaded else 0
        regulation_count = db_regulation.collection.count() if regulation_loaded else 0
        
        if handbook_count == 0 and regulation_count == 0:
            return "❌ 知识库为空，无法进行查询。请先使用「导入PDF」功能导入数据。"
        
        def _get_doc_name(db, fallback="未知文档"):
            if db.metadatas and len(db.metadatas) > 0:
                return (db.metadatas[0] or {}).get("doc_name", fallback)
            return fallback
        
        all_candidates = []
        
        if handbook_count > 0:
            handbook_doc_name = _get_doc_name(db_handbook, "新员工手册")
            bm25_handbook = BM25Retriever(db_handbook.chunks, doc_name=handbook_doc_name)
            candidates_handbook = hybrid_search(
                db_handbook, bm25_handbook, question,
                top_k=config.TOP_K,
                vector_weight=config.VECTOR_WEIGHT,
                bm25_weight=config.BM25_WEIGHT
            )
            all_candidates.extend(candidates_handbook)
            print(f"📚 handbook 检索到 {len(candidates_handbook)} 条结果")
        
        if regulation_count > 0:
            regulation_doc_name = _get_doc_name(db_regulation, "规章制度")
            bm25_regulation = BM25Retriever(db_regulation.chunks, doc_name=regulation_doc_name)
            candidates_regulation = hybrid_search(
                db_regulation, bm25_regulation, question,
                top_k=config.TOP_K,
                vector_weight=config.VECTOR_WEIGHT,
                bm25_weight=config.BM25_WEIGHT
            )
            all_candidates.extend(candidates_regulation)
            print(f"📚 regulation 检索到 {len(candidates_regulation)} 条结果")
        
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