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

    def update_point_entity(self, collection: str, point_ids: List[str], new_entity_id: str):
        """
        Updates the entity_id payload field for a list of points.
        Used during Identity Merge.
        """
        if not point_ids: return
        
        # Set payload for specific points
        self.client.set_payload(
            collection_name=collection,
            payload={"entity_id": new_entity_id},
            points=point_ids
        )

    def get_points_by_entity(self, collection: str, entity_id: str, video_id: str) -> List[str]:
        """
        Retrieves all point IDs belonging to an entity (for merging).
        """
        res = self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="entity_id", match=models.MatchValue(value=entity_id)),
                    models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))
                ]
            ),
            limit=1000, # Get a large batch
            with_payload=False
        )
        return [p.id for p in res[0]]