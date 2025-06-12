import redis
import numpy as np

import os
import fitz
import re
from sklearn.metrics.pairwise import cosine_similarity
from .utils import get_redis_client, get_sentence_transformer, VECTOR_DIM, INDEX_NAME, DOC_PREFIX



class RedisIngestor:
    def __init__(self, data_dir="../data/"):
        self.data_dir = data_dir
        self.redis_client = get_redis_client()
        self.embedding_model = get_sentence_transformer()
        self.similarities = []
        self.VECTOR_DIM = VECTOR_DIM
        self.INDEX_NAME = INDEX_NAME
        self.DOC_PREFIX = DOC_PREFIX

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_by_page.append((page_num, text))
        return text_by_page

    def split_text_into_chunks(self, text, chunk_size=500, overlap=150):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
        return chunks

    def calculate_similarity(self, embeddings):
        if len(embeddings) < 2:
            return []
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
        return sim_matrix.max(axis=1)  # Get max similarity for each chunk

    def clear_redis_store(self):
        print("Clearing existing Redis store...")
        self.redis_client.flushdb()
        print("Redis store cleared.")

    def create_hnsw_index(self):
        try:
            self.redis_client.execute_command(f"FT.DROPINDEX {self.INDEX_NAME} DD")
        except redis.exceptions.ResponseError:
            pass

        self.redis_client.execute_command(
            f"""
            FT.CREATE {self.INDEX_NAME} ON HASH PREFIX 1 {self.DOC_PREFIX}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {self.VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC COSINE
            """
        )
        print("Index created successfully.")

    def store_embeddings(self, file_name, page_num, chunks):
        if not chunks:
            return

        embeddings = self.embedding_model.encode(chunks)
        chunk_similarities = self.calculate_similarity(embeddings)
        self.similarities.extend(chunk_similarities)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            key = f"{self.DOC_PREFIX}:{file_name}_page_{page_num}_chunk_{i}"
            self.redis_client.hset(
                key,
                mapping={
                    "file": file_name,
                    "page": str(page_num),
                    "chunk": chunk,
                    "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                    "similarity": str(chunk_similarities[i]) if i < len(chunk_similarities) else "0.0"
                }
            )

    def process_pdfs(self):
        self.clear_redis_store()
        self.create_hnsw_index()

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.data_dir, file_name)

                try:
                    text_by_page = self.extract_text_from_pdf(pdf_path)

                    for page_num, text in text_by_page:
                        cleaned_text = self.clean_text(text)
                        chunks = self.split_text_into_chunks(cleaned_text)
                        if chunks:
                            self.store_embeddings(file_name, page_num, chunks)

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")




def ingest_pdfs(data_dir="data/"):
    """
    Entry point for other scripts. Clears Redis store, ingests PDFs in data_dir,
    builds HNSW index, and prints a summary.
    """
    ingestor = RedisIngestor(data_dir=data_dir)
    ingestor.process_pdfs()
