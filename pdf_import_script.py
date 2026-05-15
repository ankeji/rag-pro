'''
Author: ankeji ankeji1995@163.com
Date: 2026-05-09 15:20:20
LastEditors: ankeji ankeji1995@163.com
LastEditTime: 2026-05-15 11:03:56
FilePath: \AI_Projects\my_agent\rag-pro\pdf_import_script.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from config import config
from pdf_utils import extract_pdf_text, smart_semantic_split, save_chunks_to_txt
from embedding_utils import batch_embedding
from chroma_store import ChromaDB
from log_utils import log_step

pdf_collection_map = {
    "./docs/新员工手册.pdf": {"collection": "handbook", "doc_name": "新员工手册"},
    "./docs/企业规章制度+培训资料PDF内容（适配RAG多表管理）.pdf": {"collection": "regulation", "doc_name": "企业规章制度+培训资料"},
}

def single_pdf_import(pdf_path, collection_name, doc_name="未知文档"):
    log_step(f"读取PDF文件", "start", pdf_path)
    text = extract_pdf_text(pdf_path)
    if not text:
        log_step(f"读取PDF文件", "error", f"{pdf_path} 读取失败")
        return
    log_step(f"读取PDF文件", "done", f"读取成功")
    
    log_step(f"文本分块", "start")
    chunks = smart_semantic_split(text, max_size=config.MAX_SIZE, min_size=config.MIN_SIZE)
    log_step(f"文本分块", "done", f"分块数量：{len(chunks)}")
    save_chunks_to_txt(chunks, f"{collection_name}_分块.txt")
    
    log_step(f"生成向量", "start")
    vectors = batch_embedding(chunks)
    if not vectors:
        log_step(f"生成向量", "error", "向量生成失败")
        return
    log_step(f"生成向量", "done", f"生成{len(vectors)}个向量")
    
    log_step(f"导入ChromaDB", "start", f"表名：{collection_name}")
    db = ChromaDB(collection_name=collection_name)
    if db.load(expected_count=len(chunks)):
        log_step(f"导入ChromaDB", "done", f"表【{collection_name}】已存在，无需重复导入")
        return
    
    db.add(chunks, vectors, doc_name=doc_name)
    log_step(f"导入ChromaDB", "done", f"成功导入表【{collection_name}】")

def batch_pdf_import():
    log_step("批量导入PDF", "start", "开始导入2个PDF到对应表")
    for pdf_path, info in pdf_collection_map.items():
        single_pdf_import(pdf_path, info["collection"], info["doc_name"])
    log_step("批量导入PDF", "done", "所有PDF导入完成")

if __name__ == "__main__":
    batch_pdf_import()
    