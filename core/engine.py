import logging
import json
import uuid
from typing import List, Dict, Any, Optional
import numpy as np  # Ensure numpy is imported
from qdrant_client import models
from conclave.core.schemas import (
    FaceObservation, VoiceObservation, 
    MemoryNode, MemoryType, EntityType
)
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore
from conclave.core.embedding_service import EmbeddingService

logger = logging.getLogger("Conclave.Engine")

class ConclaveEngine:
    def __init__(self, video_id: str, config_path: str = "configs/api_config.json"):
        with open(config_path) as f:
            self.api_config = json.load(f)
            
        self.video_id = video_id
        self.vector_store = VectorStore(self.api_config["qdrant"])
        self.graph_store = GraphStore(self.api_config["neo4j"])
        
        # New Batched Embedding Service
        self.embedding_service = EmbeddingService(self.api_config["text-embedding-3-large"])
        
        # Collections defined in init_qdrant.py
        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories",
            "visual": "visual_memories",
            "text": "text_memories"
        }

    def ingest_semantic_memory(self, memory: MemoryNode):
        """
        M3-Style Smart Ingestion:
        Instead of creating duplicates, we REINFORCE existing memories if they are similar.
        """
        # 1. Get embedding for the new memory
        if not memory.embedding:
            memory.embedding = self.embedding_service.get_embeddings_batched([memory.content])[0]

        # 2. Check for existing related memories (The "Systematic" Check)
        # We look at the primary entity involved
        primary_entity = memory.linked_entities[0] if memory.linked_entities else None
        
        should_create_new = True
        
        if primary_entity:
            # Get existing semantic nodes for this entity
            existing_nodes = self.graph_store.get_connected_semantic_nodes(primary_entity)
            
            # Retrieve their vectors to compare (We batch fetch from Qdrant for speed)
            if existing_nodes:
                existing_ids = [n['id'] for n in existing_nodes]
                vectors = self.vector_store.client.retrieve(
                    collection_name="text_memories",
                    ids=existing_ids,
                    with_vectors=True
                )
                
                # Compare Logic
                for existing_point in vectors:
                    if not existing_point.vector: continue
                    
                    # Cosine Similarity
                    sim = np.dot(memory.embedding, existing_point.vector)
                    
                    if sim > 0.85:
                        logger.info(f"💪 Reinforcing Memory '{existing_point.payload.get('content')}' (Sim: {sim:.2f})")
                        # M3 Logic: Reinforce edge weight, don't create new node
                        self.graph_store.update_edge_weight(existing_point.id, primary_entity, 1.0)
                        should_create_new = False
                        break # Found a match, stop looking
                    elif sim < 0.0:
                        # Contradiction? Weaken it.
                        self.graph_store.update_edge_weight(existing_point.id, primary_entity, -0.5)

        # 3. Create only if unique
        if should_create_new:
            self.add_memory(memory)

    def ingest_face(self, obs: FaceObservation):
        """
        Aligned with main.py: Ensures temporal structure exists and 
        pre-creates nodes if necessary.
        """
        # 1. Ensure temporal structure exists in Neo4j
        self.graph_store.create_clip_structure(self.video_id, obs.clip_id)
        
        # 2. Ensure entity node exists (Idempotent)
        if obs.entity_id:
            self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, self.video_id)
        else:
            logger.warning(f"Face observation {obs.obs_id} missing entity_id - should be resolved by IdentityManager first")

    def add_memory(self, mem: MemoryNode):
        """
        Interface Parity: Handles the MemoryNode output from ReasoningEngine.
        Uses deterministic hashing for mem_id to prevent duplicates.
        """
        # Generate deterministic ID and embedding if not already present
        if not mem.embedding:
            mem.mem_id = self.embedding_service.generate_deterministic_id(mem.content)
            mem.embedding = self.embedding_service.get_embeddings_batched([mem.content])[0]
        elif not mem.mem_id:
            mem.mem_id = self.embedding_service.generate_deterministic_id(mem.content)

        # 1. Idempotent Graph Write
        self.graph_store.create_memory_node(
            mem.mem_id, mem.content, mem.mem_type.value, 
            self.video_id, mem.clip_id
        )

        # 2. Link mentioned entities (GraphRAG Backbone)
        for entity_id in mem.linked_entities:
            logger.info(f"Linking Memory {mem.mem_id[:8]} -> Entity {entity_id[:8]}")
            self.graph_store.link_memory_to_entity(mem.mem_id, entity_id, "MENTIONS")

        # 3. Idempotent Vector Write
        payload = {
            "video_id": self.video_id,
            "clip_id": mem.clip_id,
            "content": mem.content,
            "type": mem.mem_type.value,
            "model": self.embedding_service.model
        }
        self.vector_store.upsert(self.collections["text"], mem.mem_id, mem.embedding, payload)

    def add_memories_batched(self, memories: List[MemoryNode]):
        """
        Principal-level batched ingestion with deduplication and validation.
        """
        if not memories:
            return
        
        # 1. Deduplicate locally based on content + filter out nodes already in Qdrant
        unique_nodes = []
        texts_to_embed = []
        
        for mem in memories:
            # Generate deterministic ID based on content
            mem.mem_id = self.embedding_service.generate_deterministic_id(mem.content)
            
            # Check Qdrant for existence (The 'No Local Storage' deduplication)
            if not self.vector_store.point_exists(self.collections["text"], mem.mem_id):
                unique_nodes.append(mem)
                texts_to_embed.append(mem.content)
        
        if not unique_nodes:
            logger.info("All memories in batch already exist. Skipping.")
            return
        
        # 2. Batch Embedding Call
        embeddings = self.embedding_service.get_embeddings_batched(texts_to_embed)
        
        # 3. Prepare Vector and Graph payloads
        qdrant_points = []
        for node, emb in zip(unique_nodes, embeddings):
            node.embedding = emb
            
            # Vector Payload
            payload = {
                "video_id": self.video_id,
                "clip_id": node.clip_id,
                "content": node.content,
                "type": node.mem_type.value,
                "model": self.embedding_service.model
            }
            qdrant_points.append(models.PointStruct(id=node.mem_id, vector=emb, payload=payload))
            
            # 4. Synchronous Graph Commit
            # This ensures that if the vector exists, the graph relationship also exists
            self.graph_store.create_memory_node(
                node.mem_id, node.content, node.mem_type.value, 
                self.video_id, node.clip_id
            )
            for entity_id in node.linked_entities:
                self.graph_store.link_memory_to_entity(node.mem_id, entity_id, "MENTIONS")
        
        # 5. Atomic Batch Vector Upsert
        self.vector_store.upsert_batch(self.collections["text"], qdrant_points)
        logger.info(f"Successfully ingested {len(qdrant_points)} new memories.")

    def get_hybrid_context(self, query_vector: List[float], top_k: int = 5):
        """Perform Vector search then expand context via Graph."""
        
        # 🔥 CRITICAL FIX: Ensure graph is up to date before querying
        self.graph_store.flush()
        
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
            MATCH (m:Memory {id: $mem_id})
            // Optional: Get entities mentioned
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            RETURN e.id as id, e.type as type
            """
            entities = self.graph_store.run_query(query, {"mem_id": v.id})
            
            # Filter out None if optional match fails
            valid_entities = [e for e in entities if e['id'] is not None]

            results.append({
                "content": v.payload["content"],
                "clip_id": v.payload["clip_id"],
                "score": v.score,
                "involved_entities": valid_entities
            })
            
        return results

    def ingest_dialogue_event(self, video_id: str, clip_id: int, entity_id: str, text: str, ts_ms: int):
        """
        Handles full lifecycle of a spoken line with VALID QDRANT ID.
        """
        if not text or len(text) < 2: return

        # 1. Clean Text
        clean_text = text.replace('"', '').replace("'", "").strip()
        
        # 2. Embed Text
        text_vector = self.embedding_service.get_embeddings_batched([clean_text])[0]
        
        # 3. GENERATE VALID UUID (CRITICAL FIX)
        # We use uuid5 with a seed string to make it deterministic (same text+time = same ID)
        seed_string = f"dial_{video_id}_{clip_id}_{ts_ms}_{entity_id}"
        mem_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_string)) 
        
        # 4. Store in Qdrant 'text_memories'
        self.vector_store.upsert(
            "text_memories",
            mem_id,  # <--- Now passing a proper UUID string
            text_vector,
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "entity_id": entity_id,
                "content": clean_text,
                "type": "dialogue",
                "ts_ms": ts_ms,
                "readable_id": seed_string # Store original ID in payload for debugging
            }
        )
        
        # 5. Store in Neo4j
        self.graph_store.ingest_dialogue(video_id, clip_id, entity_id, clean_text, ts_ms)
        
        logger.info(f"🗣️ DIALOGUE: {entity_id} said '{clean_text}' (Saved to Graph+Vec)")