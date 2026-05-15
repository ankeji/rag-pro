import requests
import os
from dotenv import load_dotenv
load_dotenv()
from config import LLM_URL, NO_PROXY

API_KEY = os.getenv("API_KEY")

# 多轮对话历史
# llm_utils.py 里的 ChatHistory
class ChatHistory:
    def __init__(self, max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add(self, question, answer):
        self.history.append((question, answer))
        # 每加一轮自动裁剪
        self.trim_history()

    # 1. 自动裁剪：只保留最近 max_turns 轮
    def trim_history(self):
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    # 2. 自动摘要：把更早的对话浓缩，只留摘要+最新2轮
    def summarize_history(self, llm):
        # 少于4轮没必要摘要
        if len(self.history) <= 4:
            return

        # 拿出前面较早的对话
        old_dialog = self.history[:-2]
        dialog_text = "\n".join([f"用户：{q}\n助手：{a}" for q, a in old_dialog])

        prompt = f"""
请精简总结以下对话的关键内容，只用一两句话概括：
{dialog_text}
        """

        # 调用LLM生成摘要
        summary = llm.invoke(prompt).content.strip()

        # 重构历史：摘要 + 最新2轮
        self.history = [("对话历史摘要", summary)] + self.history[-2:]

    def get_context(self):
        """拼接成上下文字符串"""
        ctx = ""
        for q, a in self.history:
            ctx += f"用户：{q}\n助手：{a}\n"
        return ctx

# 上下文压缩
def compress_context(candidates, max_len=2000):
    context = []
    total_len = 0
    for idx, item in enumerate(candidates):
        source = item.get("source", "未知来源")
        doc_name = item.get("doc_name", "未知文档")
        chunk_info = f"[{idx+1}]（文档：{doc_name} | 来源：{source}）{item['chunk']}"
        chunk_len = len(chunk_info)
        if total_len + chunk_len > max_len:
            break
        context.append(chunk_info)
        total_len += chunk_len
    return "\n\n".join(context)

# 生成多检索问句
def generate_multi_query(question):
    prompt = f"请针对用户问题，生成3条语义一致、表述不同、适合文档检索的问句。每行一条，不要编号、不要多余内容。用户问题：{question}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role":"user","content":prompt}]},
        "parameters": {"temperature":0.3}
    }
    res = requests.post(LLM_URL, headers=headers, json=data, proxies=NO_PROXY)
    lines = res.json()["output"]["text"].strip().split("\n")
    queries = list(set([line.strip() for line in lines if line.strip()]))
    queries.insert(0, question)
    return queries

# LLM 生成回答
def llm_answer(question, context, chat_history=""):
    history_section = f"以下是历史对话：\n{chat_history}\n" if chat_history else ""
    prompt = f"""{history_section}你是企业内部智能助手，回答必须：
1. 只根据提供的资料回答
2. 语气正式、简洁、专业
3. 回答员工手册、规章制度类问题要严谨
4. 不知道就说“未找到相关信息”

参考内容：
{context}

问题：{question}
"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.1}
    }
    resp = requests.post(LLM_URL, headers=headers, json=data, proxies=NO_PROXY)
    return resp.json()["output"]["text"]