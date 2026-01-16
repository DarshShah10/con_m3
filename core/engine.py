import logging
import json
from typing import List, Dict, Any, Optional
from conclave.core.schemas import (
    FaceObservation, VoiceObservation, VisualObservation, 
    MemoryNode, MemoryType, EntityType
)
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore

logger = logging.getLogger("ConclaveEngine")

class ConclaveEngine:
    def __init__(self, video_id: str, config_path: str = "configs/api_config.json"):
        with open(config_path) as f:
            self.api_config = json.load(f)
            
        self.video_id = video_id
        self.vector_store = VectorStore(self.api_config["qdrant"])
        self.graph_store = GraphStore(self.api_config["neo4j"])
        
        # Collections defined in init_qdrant.py
        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories",
            "visual": "visual_memories",
            "text": "text_memories"
        }

    def ingest_face(self, obs: FaceObservation):
        """Standard pipeline for storing a face appearance and linking to Graph."""
        # 1. Ensure temporal structure exists in Neo4j
        self.graph_store.create_clip_structure(self.video_id, obs.clip_id)
        
        # 2. Resolve or Create Entity ID (IdentityManager logic will call this)
        if not obs.entity_id:
            obs.entity_id = f"ent_{obs.obs_id}" # Fallback if resolution skipped

        # 3. Save Vector
        payload = {
            "video_id": self.video_id,
            "clip_id": obs.clip_id,
            "entity_id": obs.entity_id,
            "quality": obs.quality_score
        }
        self.vector_store.upsert(self.collections["face"], obs.obs_id, obs.embedding, payload)
        
        # 4. Save Graph Node & Link
        self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, self.video_id)
        # Link this specific observation to the Clip
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id}), (e:Entity {id: $entity_id})
        MERGE (e)-[:APPEARED_IN {ts_ms: $ts, obs_id: $obs_id}]->(c)
        """
        self.graph_store.run_query(query, {
            "clip_id": obs.clip_id, "video_id": self.video_id, 
            "entity_id": obs.entity_id, "ts": obs.ts_ms, "obs_id": obs.obs_id
        })
        return obs.entity_id

    def add_memory(self, mem: MemoryNode):
        """Commit an LLM-generated caption and link it to entities found in text."""
        # 1. Save Vector
        payload = {
            "video_id": self.video_id,
            "clip_id": mem.clip_id,
            "content": mem.content,
            "type": mem.mem_type.value
        }
        self.vector_store.upsert(self.collections["text"], mem.mem_id, mem.embedding, payload)
        
        # 2. Save Graph Node
        self.graph_store.create_memory_node(
            mem.mem_id, mem.content, mem.mem_type.value, 
            self.video_id, mem.clip_id
        )
        
        # 3. Create Links (GraphRAG Backbone)
        for entity_id in mem.linked_entities:
            logger.info(f"Linking Memory {mem.mem_id[:8]} -> Entity {entity_id[:8]}")
            self.graph_store.link_memory_to_entity(mem.mem_id, entity_id, "MENTIONS")

    def get_hybrid_context(self, query_vector: List[float], top_k: int = 5):
        """Perform Vector search then expand context via Graph."""
        # 1. Find the most relevant moments (Qdrant)
        vectors = self.vector_store.search(
            self.collections["text"], 
            query_vector, 
            {"video_id": self.video_id}, 
            limit=top_k
        )
        
        results = []
        for v in vectors:
            # 2. For each moment, find who was involved (Neo4j)
            query = """
            MATCH (m:Memory {id: $mem_id})-[:MENTIONS]->(e:Entity)
            RETURN e.id as id, e.type as type
            """
            entities = self.graph_store.run_query(query, {"mem_id": v.id})
            
            results.append({
                "content": v.payload["content"],
                "clip_id": v.payload["clip_id"],
                "score": v.score,
                "involved_entities": entities
            })
            
        return results