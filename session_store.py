'''
Author: ankeji ankeji1995@163.com
Date: 2026-05-12 11:31:50
LastEditors: ankeji ankeji1995@163.com
LastEditTime: 2026-05-15 15:52:34
FilePath: \AI_Projects\my_agent\rag-pro\session_store.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import redis
from langchain_core.messages import HumanMessage, AIMessage
from config import config, LLM_URL, NO_PROXY
import os
import requests

API_KEY = os.getenv("API_KEY")

# 连接本地/服务器 Redis
r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    decode_responses=True
)

# 把消息对象转可序列化列表
def serialize_messages(messages):
    res = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            res.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            res.append({"type": "ai", "content": msg.content})
    return res

# 反序列换回 LangChain 消息对象
def deserialize_messages(data):
    messages = []
    for item in data:
        if item["type"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages

# 获取会话历史
def get_session_history(session_id: str):
    key = f"chat:session:{session_id}"
    raw = r.get(key)
    if not raw:
        return []
    return deserialize_messages(json.loads(raw))

# 保存会话历史
def summarize_history(messages):
    """使用LLM对历史对话进行摘要"""
    if len(messages) <= config.HISTORY_KEEP_ROUNDS * 2:
        return messages
    
    old_messages = messages[:-config.HISTORY_KEEP_ROUNDS * 2]
    dialog_text = ""
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            dialog_text += f"用户：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            dialog_text += f"助手：{msg.content}\n"
    
    if not dialog_text.strip():
        return messages
    
    prompt = f"""请精简总结以下对话的关键内容，只用一两句话概括，保留重要信息：
{dialog_text}"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.3}
    }
    
    try:
        resp = requests.post(LLM_URL, headers=headers, json=data, proxies=NO_PROXY, timeout=10)
        result = resp.json()
        summary = result.get("output", {}).get("text", "").strip()
        if not summary:
            return messages
        
        summary_message = HumanMessage(content=f"[历史对话摘要] {summary}")
        new_messages = [summary_message] + messages[-config.HISTORY_KEEP_ROUNDS * 2:]
        print(f"📝 历史对话已摘要: {len(messages)}条 → {len(new_messages)}条")
        return new_messages
    except Exception as e:
        print(f"⚠️ 摘要生成失败: {e}")
        return messages

def save_session_history(session_id: str, messages, expire_seconds=86400*7):
    key = f"chat:session:{session_id}"
    
    if config.ENABLE_HISTORY_SUMMARY and len(messages) > config.HISTORY_SUMMARY_THRESHOLD:
        messages = summarize_history(messages)
    
    data = json.dumps(serialize_messages(messages))
    r.set(key, data)
    r.expire(key, expire_seconds)