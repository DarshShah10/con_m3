import logging
import hashlib
import time
import uuid  # <-- Added
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAIError

logger = logging.getLogger("Conclave.Embedding")

class EmbeddingService:
    def __init__(self, config: Dict[str, Any]):
        self.model = config.get("model", "text-embedding-3-large")
        self.api_key = config.get("api_key")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 5)
        self.base_retry_delay = 2.0
        
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_deterministic_id(self, content: str) -> str:
        """
        Creates a valid UUIDv5 based on content.
        This fits Qdrant's strict ID requirements while remaining deterministic.
        """
        # Combine model and content to ensure uniqueness per embedding space
        unique_string = f"{self.model}:{content}"
        # Generate UUIDv5 using DNS namespace as a seed
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

    def _get_embeddings_with_backoff(self, texts: List[str]) -> List[List[float]]:
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
                logger.warning(f"Embedding API error: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        return []

    def get_embeddings_batched(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._get_embeddings_with_backoff(batch)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings