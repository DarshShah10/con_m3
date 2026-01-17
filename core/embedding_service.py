import logging
import hashlib
import time
import uuid
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAIError

logger = logging.getLogger("Conclave.Embedding")

class EmbeddingService:
    """
    Unified Embedding Service.
    Features: 
    - Deterministic UUIDv5 IDs (Crucial for Qdrant deduplication)
    - In-memory LRU Cache (Saves money/time on repeated text)
    - Automatic Batching & Retry (Resilience)
    """
    def __init__(self, config: Dict[str, Any]):
        self.model = config.get("model", "text-embedding-3-large")
        self.api_key = config.get("api_key")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 5)
        self.base_retry_delay = 2.0
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Simple in-memory cache: {hash(text): embedding}
        # This prevents re-embedding the same dialogue/descriptions multiple times
        self._cache: Dict[str, List[float]] = {}

    def generate_deterministic_id(self, content: str) -> str:
        """
        Creates a valid UUIDv5 based on content.
        Ensures that if we process the same memory twice, we get the same ID.
        """
        unique_string = f"{self.model}:{content}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def _request_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Internal method to hit OpenAI API with backoff."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [data.embedding for data in response.data]
            except OpenAIError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Embedding failed after {self.max_retries} attempts: {e}")
                    raise e
                
                delay = self.base_retry_delay * (2 ** attempt)
                logger.warning(f"Embedding API error. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        return []

    def get_embeddings_batched(self, texts: List[str]) -> List[List[float]]:
        """
        Public API. Handles Caching + Batching transparently.
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

        # 2. Fetch missing items in batches
        if to_embed_strings:
            for i in range(0, len(to_embed_strings), self.batch_size):
                batch_texts = to_embed_strings[i : i + self.batch_size]
                batch_indices = to_embed_indices[i : i + self.batch_size]
                
                # Network Call
                embeddings = self._request_with_retry(batch_texts)
                
                # Fill results and update cache
                for idx_in_batch, emb in enumerate(embeddings):
                    original_idx = batch_indices[idx_in_batch]
                    results[original_idx] = emb
                    
                    # Update Cache
                    cache_key = self._get_cache_key(batch_texts[idx_in_batch])
                    self._cache[cache_key] = emb

        return results