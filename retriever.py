from re import L
import jieba
import os
from dotenv import load_dotenv
import requests
from rank_bm25 import BM25Okapi
from config import config, NO_PROXY, RERANK_URL
from embedding_utils import get_embedding
from log_utils import log_step


load_dotenv()
API_KEY = os.getenv("API_KEY")
# BM25 检索
class BM25Retriever:
    def __init__(self, chunks, doc_name="未知文档"):
        self.chunks = chunks
        self.doc_name = doc_name
        self.tokenized_chunks = [self.cut(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def cut(self, text):
        return list(jieba.cut(text))

    def search(self, query, top_k=5):
        tokenized_query = self.cut(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            results.append({"chunk": self.chunks[i], "score": float(scores[i])})
        return results

# 分数归一化
def normalize_scores(scores):
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

# 混合检索
def hybrid_search(db, bm25, question, top_k=5, vector_weight=0.6, bm25_weight=0.4, vector_threshold=None, bm25_threshold=None):
    if vector_threshold is None:
        vector_threshold = config.VECTOR_SEARCH_THRESHOLD
    if bm25_threshold is None:
        bm25_threshold = config.BM25_SEARCH_THRESHOLD

    q_vec = get_embedding(question)
    vector_candidates = db.search(q_vec, top_k=top_k)
    for item in vector_candidates:
        item["source"] = "向量检索"
        item["vector_score"] = item["score"]

    bm25_candidates = bm25.search(question, top_k=top_k)
    for item in bm25_candidates:
        item["source"] = "关键词检索(BM25)"
        item["bm25_score"] = item["score"]
        item["doc_name"] = bm25.doc_name

    normalized_vector_scores = normalize_scores([x["vector_score"] for x in vector_candidates])
    normalized_bm25_scores = normalize_scores([x["bm25_score"] for x in bm25_candidates])

    for i, item in enumerate(vector_candidates):
        item["normalized_vector_score"] = normalized_vector_scores[i]
    for i, item in enumerate(bm25_candidates):
        item["normalized_bm25_score"] = normalized_bm25_scores[i]

    filtered_vector = [c for c in vector_candidates if c["normalized_vector_score"] >= vector_threshold]
    filtered_bm25 = [c for c in bm25_candidates if c["normalized_bm25_score"] >= bm25_threshold]

    all_candidates = filtered_vector + filtered_bm25
    for item in all_candidates:
        if "normalized_vector_score" in item:
            item["weighted_score"] = item["normalized_vector_score"] * vector_weight
        else:
            item["weighted_score"] = item["normalized_bm25_score"] * bm25_weight

    all_candidates_sorted = sorted(all_candidates, key=lambda x: x["weighted_score"], reverse=True)
    seen = set()
    unique_candidates = []
    for c in all_candidates_sorted:
        if c["chunk"] not in seen:
            seen.add(c["chunk"])
            unique_candidates.append(c)
    
    log_step("Rerank结果", "info", f"🔍 检索问题: {question}，向量{len(vector_candidates)}个(过滤后{len(filtered_vector)}个), BM25{len(bm25_candidates)}个(过滤后{len(filtered_bm25)}个), 最终{len(unique_candidates[:top_k])}个")
    return unique_candidates[:top_k]

# Rerank 重排
def rerank(question, candidates):
    valid_candidates = [c for c in candidates if c.get("chunk", "").strip()]
    if not valid_candidates:
        log_step("Rerank结果", "warning", "⚠️ 所有候选文档都为空，跳过 Rerank")
        return candidates

    chunk_to_meta = {c["chunk"]: c for c in valid_candidates}
    data = {
        "model": "gte-rerank-v2",
        "input": {"query": question, "documents": [c["chunk"] for c in valid_candidates]},
        "parameters": {"return_documents": True}
    }

    headers = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

    try:
        resp = requests.post(RERANK_URL, json=data, headers=headers, timeout=15, proxies=NO_PROXY)
        result = resp.json()
        if "code" in result:
            log_step("Rerank结果", "error", f"❌ Rerank API 错误: {result}")
            return candidates

        reranked = []
        for item in result["output"]["results"]:
            doc = item.get("document", "")
            if isinstance(doc, dict):
                doc = doc.get("text", str(doc))
            meta = chunk_to_meta.get(doc, {})
            entry = {
                "chunk": doc,
                "score": item["relevance_score"],
                "source": meta.get("source", "未知来源"),
                "doc_name": meta.get("doc_name", "未知文档"),
            }
            reranked.append(entry)
        
        sorted_results = sorted(reranked, key=lambda x: x["score"], reverse=True)
        filtered_results = [r for r in sorted_results if r["score"] >= config.RERANK_THRESHOLD]
        final_results = filtered_results[:config.RERANK_TOP_K]
        
        log_step("Rerank结果", "info", f"📊 Rerank结果: 总数{len(sorted_results)}个, 闸值是{config.RERANK_THRESHOLD}, 阈值过滤后{len(filtered_results)}个, 最终保留{len(final_results)}个")
        return final_results
    except Exception as e:
        log_step("Rerank结果", "error", f"⚠️ Rerank 调用失败: {e}")
        return candidates