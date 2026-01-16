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
        """Checks if a deterministic ID already exists to skip redundant processing."""
        try:
            res = self.client.retrieve(collection_name=collection, ids=[point_id])
            return len(res) > 0
        except:
            return False

    def upsert_batch(self, collection: str, points: List[models.PointStruct]):
        """Professional batch upsert for performance."""
        if not points:
            return
        self.client.upsert(
            collection_name=collection,
            points=points
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

    def update_point_entity(self, collection: str, point_id: str, new_entity_id: str):
        """Updates the entity_id in a vector's payload."""
        self.client.set_payload(
            collection_name=collection,
            payload={"entity_id": new_entity_id},
            points=[point_id]
        )

    def get_points_by_entity(self, collection: str, entity_id: str, video_id: str):
        """Retrieves all point IDs belonging to an entity."""
        res = self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="entity_id", match=models.MatchValue(value=entity_id)),
                    models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))
                ]
            ),
            limit=100,
            with_payload=False
        )
        return [p.id for p in res[0]]