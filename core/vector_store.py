import json
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VectorStore:
    def __init__(self, config: Dict[str, Any]):
        self.client = QdrantClient(
            url=config["url"], 
            api_key=config["api_key"],
            timeout=30
        )

    def upsert(self, collection: str, point_id: str, vector: List[float], payload: Dict[str, Any]):
        self.client.upsert(
            collection_name=collection,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    def search(self, collection: str, vector: List[float], filter_kv: Dict[str, Any], limit: int = 5):
        must_filters = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filter_kv.items()
        ]
        return self.client.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=models.Filter(must=must_filters),
            limit=limit
        )