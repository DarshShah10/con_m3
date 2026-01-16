import logging
from typing import Dict, Any, List, Set
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore

logger = logging.getLogger("Conclave.IdentityResolver")

class IdentityResolver:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vs = vector_store
        self.gs = graph_store
        # Local mapping of entity_id -> metadata for fast lookup during a session
        self.entity_index: Dict[str, Dict[str, Any]] = {}

    def get_canonical_id(self, entity_id: str) -> str:
        """In a production multi-agent system, this resolves aliased IDs."""
        return entity_id # Currently passthrough; placeholder for global alias table

    def deep_merge_entities(self, video_id: str, primary_id: str, secondary_id: str):
        """
        SOTA Merge Operation:
        1. Refactor Neo4j (move all edges from secondary to primary).
        2. Update Qdrant payloads (re-tag all points).
        3. Clear secondary from local index.
        """
        if primary_id == secondary_id: return
        
        logger.warning(f"INITIATING DEEP MERGE: {secondary_id} -> {primary_id}")

        # 1. Graph Refactor (Neo4j)
        # Moves APPEARED_IN and MENTIONS relationships
        merge_query = """
        MATCH (p:Entity {id: $primary_id}), (s:Entity {id: $secondary_id})
        MATCH (s)-[r:APPEARED_IN]->(c:Clip)
        MERGE (p)-[new_r:APPEARED_IN {obs_id: r.obs_id}]->(c)
        ON CREATE SET new_r.ts_ms = r.ts_ms
        WITH s, p
        MATCH (m:Memory)-[r:MENTIONS]->(s)
        MERGE (m)-[:MENTIONS]->(p)
        WITH s
        DETACH DELETE s
        """
        self.gs.run_query(merge_query, {"primary_id": primary_id, "secondary_id": secondary_id})

        # 2. Vector Payload Update (Qdrant)
        # Re-tag every Face and Voice vector associated with the old ID
        for collection in ["face_memories", "voice_memories"]:
            # Logic: Scroll Qdrant for all points with secondary_id, then update payload
            points_to_update = self.vs.client.scroll(
                collection_name=collection,
                scroll_filter={"must": [{"key": "entity_id", "match": {"value": secondary_id}}]},
                with_payload=False
            )[0]
            
            if points_to_update:
                ids = [p.id for p in points_to_update]
                self.vs.client.set_payload(
                    collection_name=collection,
                    payload={"entity_id": primary_id},
                    points=ids
                )
                logger.info(f"Updated {len(ids)} vector points in {collection}")

        # 3. Update local index
        if secondary_id in self.entity_index:
            del self.entity_index[secondary_id]