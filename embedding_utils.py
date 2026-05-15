'''
Author: ankeji ankeji1995@163.com
Date: 2026-05-09 14:35:23
LastEditors: ankeji ankeji1995@163.com
LastEditTime: 2026-05-15 11:41:46
FilePath: \AI_Projects\my_agent\rag-pro\embedding_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
import os
from dotenv import load_dotenv
load_dotenv()
import requests
from config import EMBEDDING_URL, NO_PROXY
import numpy as np
from log_utils import log_step

API_KEY = os.getenv("API_KEY")

def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-v2",
        "input": {"texts": [text]},
    }
    try:
        resp = requests.post(EMBEDDING_URL, headers=headers, json=data, proxies=NO_PROXY)
        result = resp.json()
        if "code" in result:
            log_step("向量生成", "error", f"API错误: {result}")
            return None
        return result["output"]["embeddings"][0]["embedding"]
    except Exception as e:
        log_step("向量生成", "error", f"请求异常: {e}")
        return None

def batch_embedding(texts):
    vectors = []
    total = len(texts)
    log_step("批量生成向量", "start", f"总块数：{total}")
    for idx, t in enumerate(texts):
        vec = get_embedding(t)
        if vec:
            vectors.append(vec)
        progress = (idx + 1) / total * 100
        print(f"\r✅ 已生成：{idx + 1}/{total} | 进度：{progress:.1f}%", end="", flush=True)
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            log_step("向量生成进度", "info", f"已生成：{idx + 1}/{total} ({progress:.1f}%)")
        time.sleep(0.05)
    print()
    log_step("批量生成向量", "done", f"成功：{len(vectors)}/{total}")
    return vectors