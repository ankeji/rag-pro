import os

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    LLM_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    BASE_URL = "https://openapi.7x24cc.com/v1/chat/completions"
    
    CHROMA_DB_DIR = os.path.join(_PROJECT_DIR, "chroma_db")
    
    MAX_SIZE = 600 # 最大chunk长度
    MIN_SIZE = 50 # 最小chunk长度
    MAX_CONTEXT_LEN = 2000 # 最大上下文长度
    
    TOP_K = 3 # 最多返回多少个chunk
    VECTOR_WEIGHT = 0.6 # 向量权重
    BM25_WEIGHT = 0.4 # BM25权重
    
    RERANK_TOP_K = 5 # Rerank最多返回多少个chunk
    RERANK_THRESHOLD = 0 # Rerank阈值
    
    VECTOR_SEARCH_THRESHOLD = 0.5 # 向量搜索阈值
    BM25_SEARCH_THRESHOLD = 0.5 # BM25搜索阈值
    
    ENABLE_HISTORY_SUMMARY = True # 是否启用历史摘要
    HISTORY_SUMMARY_THRESHOLD = 6 # 历史摘要阈值
    HISTORY_KEEP_ROUNDS = 2 # 历史摘要保留轮数
    
    NO_PROXY = {"http": None, "https": None} 
    ENABLE_DANGER_OP = True # 是否启用危险操作
    
    def to_dict(self):
        return {
            "MAX_SIZE": self.MAX_SIZE,
            "MIN_SIZE": self.MIN_SIZE,
            "MAX_CONTEXT_LEN": self.MAX_CONTEXT_LEN,
            "TOP_K": self.TOP_K,
            "VECTOR_WEIGHT": self.VECTOR_WEIGHT,
            "BM25_WEIGHT": self.BM25_WEIGHT,
            "RERANK_TOP_K": self.RERANK_TOP_K,
            "RERANK_THRESHOLD": self.RERANK_THRESHOLD,
            "VECTOR_SEARCH_THRESHOLD": self.VECTOR_SEARCH_THRESHOLD,
            "BM25_SEARCH_THRESHOLD": self.BM25_SEARCH_THRESHOLD,
            "ENABLE_HISTORY_SUMMARY": self.ENABLE_HISTORY_SUMMARY,
            "HISTORY_SUMMARY_THRESHOLD": self.HISTORY_SUMMARY_THRESHOLD,
            "HISTORY_KEEP_ROUNDS": self.HISTORY_KEEP_ROUNDS,
            "ENABLE_DANGER_OP": self.ENABLE_DANGER_OP
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

config = Config()

EMBEDDING_URL = config.EMBEDDING_URL
LLM_URL = config.LLM_URL
RERANK_URL = config.RERANK_URL
BASE_URL = config.BASE_URL
CHROMA_DB_DIR = config.CHROMA_DB_DIR
NO_PROXY = config.NO_PROXY
ENABLE_DANGER_OP = config.ENABLE_DANGER_OP
