from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, config: Dict[str, Any]):
        self.client = QdrantClient(
            url=config["url"], 
            api_key=config["api_key"],
            timeout=30
        )

    def point_exists(self, collection: str, point_id: str) -> bool:
        """Checks if a deterministic ID already exists."""
        try:
            res = self.client.retrieve(collection_name=collection, ids=[point_id])
            return len(res) > 0
        except:
            return False

    def upsert(self, collection: str, point_id: str, vector: List[float], payload: Dict[str, Any]):
        """
        Singular upsert (Used by IdentityManager and SceneProcessor).
        """
        self.client.upsert(
            collection_name=collection,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    def upsert_batch(self, collection: str, points: List[models.PointStruct]):
        """
        Batch upsert (Used by Engine for Memories).
        """
        if not points:
            return
        self.client.upsert(
            collection_name=collection,
            points=points
        )

    def search(self, collection: str, vector: List[float], filter_kv: Dict[str, Any], limit: int = 5):
        """
        Similarity search with exact match filtering.
        """
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

    def update_point_entity(self, collection: str, old_id: str, new_id: str, video_id: str):
        """
        Retags all vectors belonging to 'old_id' with 'new_id'.
        """
        # 1. Find points
        res, _ = self.client.scroll(
            collection_name=collection,
            scroll_filter=self._build_filter({"entity_id": old_id, "video_id": video_id}),
            limit=1000
        )
        
        if not res: return
        
        points_to_update = [p.id for p in res]
        
        # 2. Update payload
        self.client.set_payload(
            collection_name=collection,
            payload={"entity_id": new_id},
            points=points_to_update
        )

    def _build_filter(self, kv_dict):
        # Helper for filter construction
        from qdrant_client import models
        return models.Filter(
            must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k,v in kv_dict.items()]
        )