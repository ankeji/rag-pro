'''
Author: ankeji ankeji1995@163.com
Date: 2026-05-09 14:35:14
LastEditors: ankeji ankeji1995@163.com
LastEditTime: 2026-05-15 15:53:52
FilePath: \AI_Projects\my_agent\rag-pro\pdf_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import re
from PyPDF2 import PdfReader
from config import config
from log_utils import log_step

def extract_pdf_text(pdf_path):
    full_text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        full_text = re.sub(r'[ \t]+', ' ', full_text)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = full_text.strip()
    except Exception as e:
        log_step("PDF读取", "error", f"读取失败: {e}")
    return full_text

def smart_semantic_split(text, max_size=config.MAX_SIZE, min_size=config.MIN_SIZE):
    major_title_pat = re.compile(r"^(第[一二三四五六七八九十百]+章|第[一二三四五六七八九十百]+条)")
    minor_title_pat = re.compile(r"^([一二三四五六七八九十百]+、|\(\d+\)|\([一二三四五六七八九十]+\)|（\d+）|（[一二三四五六七八九十]+）|\d+\.|\d+、|[①②③④⑤⑥⑦⑧⑨⑩])")
    title_insert_pat = re.compile(r'([^\n\s])(第[一二三四五六七八九十百]+章|第[一二三四五六七八九十百]+条|[一二三四五六七八九十百]+、|\(\d+\)|\([一二三四五六七八九十]+\)|（\d+）|（[一二三四五六七八九十]+）|\d+\.|\d+、|[①②③④⑤⑥⑦⑧⑨⑩])')
    text = title_insert_pat.sub(r'\1\n\2', text)

    sentences = re.split(r'\n', text)
    sentences = [s.strip() for s in sentences if s.strip()]
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
                chunks[-1] += text_block
            else:
                chunks.append(text_block)
        current = []
        current_len = 0

    for sent in sentences:
        is_major = bool(major_title_pat.match(sent))
        is_minor = bool(minor_title_pat.match(sent))

        if is_major and current:
            flush_current()
        elif is_minor and current_len >= min_size:
            flush_current()
        elif not is_major and not is_minor and current_len + len(sent) > max_size:
            flush_current()

        current.append(sent)
        current_len += len(sent)

    if current:
        flush_current()

    return [c for c in chunks if c.strip()]

def save_chunks_to_txt(chunks, save_path="chunks_查看全部.txt"):
    with open(save_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            f.write("="*60 + "\n")
            f.write(f"【第 {idx+1} 块 | 字符数：{len(chunk)}】\n")
            f.write("="*60 + "\n")
            f.write(chunk)
            f.write("\n\n")
    log_step("保存分块", "info", f"已保存到：{save_path}，共 {len(chunks)} 块")