'''
Author: ankeji ankeji1995@163.com
Date: 2026-05-09 15:20:20
LastEditors: ankeji ankeji1995@163.com
LastEditTime: 2026-05-18 11:40:59
FilePath: \AI_Projects\my_agent\rag-pro\pdf_import_script.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import glob
from config import config
from pdf_utils import extract_pdf_text, smart_semantic_split, save_chunks_to_txt
from embedding_utils import batch_embedding
from chroma_store import ChromaDB
from log_utils import log_step

DOCS_DIR = "./docs"

def scan_pdf_files():
    """扫描 docs 目录下的所有 PDF 文件"""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        log_step("扫描PDF文件", "info", f"创建 docs 目录：{DOCS_DIR}")
        return []
    
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    pdf_files.extend(glob.glob(os.path.join(DOCS_DIR, "*.PDF")))
    
    pdf_collection_map = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        doc_name = os.path.splitext(filename)[0]
        collection_name = doc_name.lower().replace(" ", "_").replace("+", "_").replace("（", "_").replace("）", "")
        collection_name = "".join(c for c in collection_name if c.isalnum() or c == "_")
        
        pdf_collection_map.append({
            "pdf_path": pdf_path,
            "collection": collection_name,
            "doc_name": doc_name
        })
    
    return pdf_collection_map

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
    # save_chunks_to_txt(chunks, f"{collection_name}_分块.txt")
    
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
    """批量导入 docs 目录下的所有 PDF 文件"""
    pdf_collection_map = scan_pdf_files()
    
    if not pdf_collection_map:
        log_step("批量导入PDF", "warning", f"docs 目录下没有找到 PDF 文件，请将 PDF 文件放入 {DOCS_DIR} 目录")
        return
    
    log_step("批量导入PDF", "start", f"发现 {len(pdf_collection_map)} 个 PDF 文件")
    for info in pdf_collection_map:
        single_pdf_import(info["pdf_path"], info["collection"], info["doc_name"])
    log_step("批量导入PDF", "done", "所有 PDF 导入完成")

if __name__ == "__main__":
    batch_pdf_import()
    