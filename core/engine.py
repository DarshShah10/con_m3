import logging
import json
import uuid
import numpy as np
from typing import List
from qdrant_client import models
from conclave.core.schemas import MemoryNode
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore
from conclave.core.embedding_service import EmbeddingService

logger = logging.getLogger("Conclave.Engine")

class ConclaveEngine:
    def __init__(self, video_id: str, config_path: str = "configs/api_config.json"):
        with open(config_path) as f: self.api_config = json.load(f)
        self.video_id = video_id
        self.vector_store = VectorStore(self.api_config["qdrant"])
        self.graph_store = GraphStore(self.api_config["neo4j"])
        self.embedding_service = EmbeddingService(self.api_config["text-embedding-3-large"])
        self.collections = {"text": "text_memories"}
        
        # Ensure indexes exist on startup
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Creates critical Qdrant indexes required for filtering/consolidation."""
        try:
            # 1. Text Memories
            self.vector_store.client.create_payload_index(
                collection_name="text_memories",
                field_name="type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.vector_store.client.create_payload_index(
                collection_name="text_memories",
                field_name="video_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # 2. Face/Voice Memories
            for col in ["face_memories", "voice_memories"]:
                self.vector_store.client.create_payload_index(
                    collection_name=col,
                    field_name="entity_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.vector_store.client.create_payload_index(
                    collection_name=col,
                    field_name="video_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
            logger.info("✅ Qdrant Indexes Verified.")
        except Exception as e:
            logger.warning(f"Index check note: {e}")

    def ingest_semantic_memory(self, memory: MemoryNode):
        """M3-Style: Reinforce if similar exists, else create."""
        if not memory.embedding:
            memory.embedding = self.embedding_service.get_embeddings_batched([memory.content])[0]
        
        primary_entity = memory.linked_entities[0] if memory.linked_entities else None
        create_new = True

        if primary_entity:
            nodes = self.graph_store.get_connected_semantic_nodes(primary_entity)
            if nodes:
                points = self.vector_store.client.retrieve("text_memories", [n['id'] for n in nodes], with_vectors=True)
                for p in points:
                    if not p.vector: continue
                    sim = np.dot(memory.embedding, p.vector)
                    if sim > 0.85:
                        self.graph_store.update_edge_weight(p.id, primary_entity, 1.0)
                        create_new = False
                        break
                    elif sim < 0.0:
                        self.graph_store.update_edge_weight(p.id, primary_entity, -0.5)

        if create_new: self.add_memory(memory)

    def add_memory(self, mem: MemoryNode):
        if not mem.mem_id: mem.mem_id = self.embedding_service.generate_deterministic_id(mem.content)
        if not mem.embedding: mem.embedding = self.embedding_service.get_embeddings_batched([mem.content])[0]
        
        self.graph_store.create_memory_node(mem.mem_id, mem.content, mem.mem_type.value, self.video_id, mem.clip_id)
        for eid in mem.linked_entities: self.graph_store.link_memory_to_entity(mem.mem_id, eid, "MENTIONS")
        
        self.vector_store.upsert("text_memories", mem.mem_id, mem.embedding, {
            "video_id": self.video_id, "clip_id": mem.clip_id, "content": mem.content, "type": mem.mem_type.value
        })

    def add_memories_batched(self, memories: List[MemoryNode]):
        if not memories: return
        texts = [m.content for m in memories]
        embs = self.embedding_service.get_embeddings_batched(texts)
        points = []
        for m, emb in zip(memories, embs):
            m.embedding = emb
            m.mem_id = self.embedding_service.generate_deterministic_id(m.content)
            self.graph_store.create_memory_node(m.mem_id, m.content, m.mem_type.value, self.video_id, m.clip_id)
            for eid in m.linked_entities: self.graph_store.link_memory_to_entity(m.mem_id, eid, "MENTIONS")
            points.append(models.PointStruct(id=m.mem_id, vector=emb, payload={
                "video_id": self.video_id, "clip_id": m.clip_id, "content": m.content, "type": m.mem_type.value
            }))
        self.vector_store.upsert_batch("text_memories", points)

    def get_hybrid_context(self, vec):
        self.graph_store.flush()
        hits = self.vector_store.search("text_memories", vec, {"video_id": self.video_id}, limit=5)
        res = []
        for h in hits:
            ents = self.graph_store.run_query("MATCH (m {id:$i})-[:MENTIONS]->(e) RETURN e.id", {"i":h.id})
            res.append({"content":h.payload["content"], "clip_id":h.payload["clip_id"], "involved_entities":ents})
        return res

    def ingest_dialogue_event(self, vid, cid, eid, txt, ts):
        vec = self.embedding_service.get_embeddings_batched([txt])[0]
        
        # 🔥 FIX: Generate a valid UUID from the string components
        unique_str = f"dial_{vid}_{cid}_{ts}"
        mid = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        
        self.vector_store.upsert("text_memories", mid, vec, {"video_id":vid, "clip_id":cid, "content":txt, "type":"dialogue", "entity_id":eid})
        self.graph_store.ingest_dialogue(vid, cid, eid, txt, ts)