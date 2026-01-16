import logging
import hashlib
import time
import functools
from typing import List, Dict, Any, Optional
import openai
from conclave.core.vector_store import VectorStore

logger = logging.getLogger("Conclave.EmbeddingManager")

class EmbeddingManager:
    def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
        """
        Principal-level Embedding Service.
        Features: LRU Caching, Batching, Exponential Backoff.
        """
        self.vector_store = vector_store
        self.model = config.get("model", "text-embedding-3-large")
        self.api_key = config.get("api_key")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 5)
        
        self.client = openai.OpenAI(api_key=self.api_key)
        # In-memory LRU cache to avoid re-embedding same text in one session
        self._cache: Dict[str, List[float]] = {}

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Retrieves embeddings for a list of strings using caching and batching.
        """
        if not texts:
            return []

        results = [None] * len(texts)
        to_embed_indices = []
        to_embed_strings = []

        # 1. Check Cache
        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_embed_indices.append(i)
                to_embed_strings.append(text)

        # 2. Batch Request for missing items
        if to_embed_strings:
            for i in range(0, len(to_embed_strings), self.batch_size):
                batch = to_embed_strings[i : i + self.batch_size]
                indices = to_embed_indices[i : i + self.batch_size]
                
                embeddings = self._request_with_retry(batch)
                
                for idx_in_batch, emb in enumerate(embeddings):
                    original_idx = indices[idx_in_batch]
                    results[original_idx] = emb
                    # Update Cache
                    self._cache[self._get_cache_key(batch[idx_in_batch])] = emb

        return results

    def _request_with_retry(self, batch: List[str]) -> List[List[float]]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
                return [d.embedding for d in response.data]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                wait = 2 ** attempt
                logger.warning(f"Embedding retry {attempt+1}/{self.max_retries} in {wait}s: {e}")
                time.sleep(wait)
        return []