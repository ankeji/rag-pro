import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import time
from config import CHROMA_DB_DIR

class ChromaDB:
    def __init__(self, collection_name="handbook", db_path=CHROMA_DB_DIR):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.collection_name = collection_name
        self.chunks = []

    def load(self, expected_count=None):
        try:
            existing = self.collection.count()
            if expected_count and existing != expected_count:
                return False
            data = self.collection.get(include=["documents", "metadatas"])
            self.chunks = data["documents"]
            self.metadatas = data["metadatas"]
            return len(self.chunks) > 0
        except:
            return False

    def add(self, chunks, vectors, doc_name="未知文档"):
        print(f"\n📥 入库到表：{self.collection_name}")
        self.chunks = chunks
        BATCH_SIZE = 32
        total = len(chunks)
        for i in range(0, total, BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            batch_vecs = vectors[i:i+BATCH_SIZE]
            batch_ids = [f"{self.collection_name}_{i+j}" for j in range(len(batch_chunks))]
            batch_metas = [{"doc_name": doc_name}] * len(batch_chunks)
            self.collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=batch_vecs,
                metadatas=batch_metas
            )
            progress = min(i + BATCH_SIZE, total)
            print(f"\r入库进度：{progress}/{total}", end="", flush=True)
            time.sleep(0.1)
        print(f"\n✅ 入库完成！当前表数量：{self.collection.count()}")

    def search(self, query_vec, top_k=10):
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        res = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            score = 1.0 / (1.0 + dist)
            meta = (results["metadatas"][0][i] if results["metadatas"] else None) or {}
            res.append({
                "chunk": results["documents"][0][i],
                "score": float(score),
                "doc_name": meta.get("doc_name", "未知文档")
            })
        return res

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        print(f"✅ 表【{self.collection_name}】已清空！")

    # 在你的 chroma_store.py 里加这个方法
    def create_index(self):
        self.collection.create_index()
        print("✅ 向量索引创建完成，检索速度提升！")

    # 清空整个表
    def clearAll(self):
        all_ids = [item["id"] for item in self.collection.get()["ids"]]
        if all_ids:
            self.collection.delete(ids=all_ids)
        print("✅ 表已清空")

        # 加在 ChromaDB 类里
    def delete_by_doc_name(self, doc_name):
        self.collection.delete(
            where={"doc_name": doc_name}
        )
        print(f"✅ 已删除文档：{doc_name} 的所有分块")

    def list_documents(self):
        """列出当前库中所有不重复的文档名"""
        metas = self.collection.get()["metadatas"]
        doc_names = list({m.get("doc_name", "未知文档") for m in metas})
        print(f"📄 {self.collection_name} 表中的文档：{doc_names}")
        return doc_names

def get_all_collections(db_path=CHROMA_DB_DIR):
    """获取 ChromaDB 中所有的 collection 名称"""
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    return [c.name for c in collections]
