import os

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    LLM_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    BASE_URL = "https://openapi.7x24cc.com/v1/chat/completions"
    
    CHROMA_DB_DIR = os.path.join(_PROJECT_DIR, "chroma_db")
    PDF_PATH = os.path.join(_PROJECT_DIR, "docs", "新员工手册.pdf")
    
    MAX_SIZE = 600
    MIN_SIZE = 50
    MAX_CONTEXT_LEN = 2000
    
    TOP_K = 3
    VECTOR_WEIGHT = 0.6
    BM25_WEIGHT = 0.4
    
    RERANK_TOP_K = 5
    RERANK_THRESHOLD = 0
    
    VECTOR_SEARCH_THRESHOLD = 0.5
    BM25_SEARCH_THRESHOLD = 0.5
    
    ENABLE_HISTORY_SUMMARY = True
    HISTORY_SUMMARY_THRESHOLD = 6
    HISTORY_KEEP_ROUNDS = 2
    
    NO_PROXY = {"http": None, "https": None}
    ALLOW_COLLECTIONS = {"handbook", "regulation"}
    ENABLE_DANGER_OP = True
    
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
PDF_PATH = config.PDF_PATH
NO_PROXY = config.NO_PROXY
ALLOW_COLLECTIONS = config.ALLOW_COLLECTIONS
ENABLE_DANGER_OP = config.ENABLE_DANGER_OP
