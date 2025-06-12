# src/utils.py

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

# ─── Embedding Model Loader ────────────────────────────────────────────────────
# We’ll instantiate the SentenceTransformer once and reuse it in both ingest & search.

_EMBEDDING_MODEL = None
VECTOR_DIM = 384
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"

def get_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL

def get_embedding_st_minilm(text: str) -> np.ndarray:
    """
    Embed a single piece of text with the same MiniLM model used in ingestion.
    Returns a 384-dim float32 numpy array.
    """
    model = get_sentence_transformer()
    emb = model.encode([text], convert_to_numpy=True)[0]  # returns shape (384,)
    return emb.astype(np.float32)

# ─── Redis Client Helper ───────────────────────────────────────────────────────
def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
