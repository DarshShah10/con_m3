import logging
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set
from conclave.core.schemas import FaceObservation, VoiceObservation, EntityType
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore

logger = logging.getLogger("Conclave.Identity")

class IdentityManager:
    """
    Robust Identity Resolution System.
    
    Key Features:
    1. Persistent Aliasing: Loads merges from Graph on startup.
    2. Non-Destructive Merging: Retains old nodes as 'tombstones' for audit.
    3. Canonical Resolution: Always returns the primary ID, even for merged entities.
    """
    
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Configuration
        self.face_threshold = config.get("face_match_threshold", 0.65)
        self.voice_threshold = config.get("voice_match_threshold", 0.70)
        
        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories"
        }
        
        # Identity Cache: { "alias_id": "canonical_id" }
        self._alias_map: Dict[str, str] = {}
        
        # Load existing aliases from the graph (Persistence)
        self._load_alias_map()

    def _load_alias_map(self):
        """Rehydrate the alias map from Neo4j to ensure state persists across restarts."""
        query = """
        MATCH (alias:Entity)-[:MERGED_INTO]->(canonical:Entity)
        RETURN alias.id as alias, canonical.id as canonical
        """
        try:
            results = self.graph_store.run_query(query)
            count = 0
            for r in results:
                self._alias_map[r['alias']] = r['canonical']
                count += 1
            if count > 0:
                logger.info(f"♻️  Restored {count} identity aliases from Graph.")
        except Exception as e:
            logger.warning(f"Could not load aliases on init (Graph might be empty): {e}")

    def get_canonical_id(self, entity_id: str) -> str:
        """Recursive lookup to find the true ID of an entity."""
        # Prevent infinite loops with a depth limit
        depth = 0
        curr = entity_id
        while curr in self._alias_map and depth < 10:
            curr = self._alias_map[curr]
            depth += 1
        return curr

    def _generate_entity_id(self, prefix: str = "ent") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def resolve_face(self, obs: FaceObservation) -> str:
        """
        Finds the best matching face or creates a new one.
        Returns the CANONICAL ID.
        """
        # 1. Vector Search
        results = self.vector_store.search(
            collection=self.collections["face"],
            vector=obs.embedding,
            filter_kv={"video_id": obs.video_id},
            limit=1
        )
        
        matched_id = None
        if results and results[0].score >= self.face_threshold:
            raw_id = results[0].payload.get("entity_id")
            matched_id = self.get_canonical_id(raw_id)
            logger.info(f"👤 Face matched: {matched_id} (Score: {results[0].score:.2f})")
        
        if matched_id:
            obs.entity_id = matched_id
        else:
            # Create new Entity
            new_id = self._generate_entity_id("ent_face")
            self.graph_store.create_entity_node(new_id, EntityType.PERSON.value, obs.video_id)
            obs.entity_id = new_id
            logger.info(f"👤 New Face detected: {new_id}")

        return obs.entity_id

    def resolve_voice(self, obs: VoiceObservation) -> str:
        """
        Finds the best matching voice or creates a new one.
        Returns the CANONICAL ID.
        """
        # 1. Vector Search
        results = self.vector_store.search(
            collection=self.collections["voice"],
            vector=obs.embedding,
            filter_kv={"video_id": obs.video_id},
            limit=1
        )
        
        matched_id = None
        if results and results[0].score >= self.voice_threshold:
            raw_id = results[0].payload.get("entity_id")
            matched_id = self.get_canonical_id(raw_id)
            logger.info(f"🎙️ Voice matched: {matched_id} (Score: {results[0].score:.2f})")
        
        if matched_id:
            obs.entity_id = matched_id
        else:
            # Fallback: Create new Voice Entity
            new_id = self._generate_entity_id("ent_voice")
            self.graph_store.create_entity_node(new_id, EntityType.PERSON.value, obs.video_id)
            obs.entity_id = new_id
            logger.info(f"🎙️ New Voice detected: {new_id}")
        
        return obs.entity_id

    def register_observation(self, obs: Union[FaceObservation, VoiceObservation]):
        """
        Links the observation to the entity in both Graph and Vector Store.
        Uses the resolved (canonical) entity_id.
        """
        if not obs.entity_id:
            logger.warning("Observation has no entity_id. Skipping registration.")
            return

        # 1. Update Graph (Async)
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (e)-[r:APPEARED_IN {obs_id: $obs_id}]->(c)
        ON CREATE SET r.ts_ms = $ts
        """
        self.graph_store.execute_async(query, {
            "clip_id": obs.clip_id,
            "video_id": obs.video_id,
            "entity_id": obs.entity_id,
            "obs_id": obs.obs_id,
            "ts": obs.ts_ms
        })

        # 2. Update Vector Store (Sync/Fast)
        collection = self.collections["face"] if isinstance(obs, FaceObservation) else self.collections["voice"]
        payload = {
            "video_id": obs.video_id,
            "entity_id": obs.entity_id,
            "clip_id": obs.clip_id,
            "obs_id": obs.obs_id,
            "type": "resolved_identity"
        }
        self.vector_store.upsert(collection, obs.obs_id, obs.embedding, payload)

    # --- SOTA Maintenance Features ---

    def consolidate_identities(self, video_id: str):
        """
        Scans for entities that are extremely similar and merges them.
        Fixes the 'Fragmented Identity' problem (e.g., same person, slightly different lighting).
        """
        # Ensure latest graph writes are visible before we start querying logic
        self.graph_store.flush()
        
        # We process voices specifically as they are prone to fragmentation
        self._consolidate_collection("voice_memories", video_id, threshold=0.85)
        # We can also do faces
        self._consolidate_collection("face_memories", video_id, threshold=0.80)

    def _consolidate_collection(self, collection_name: str, video_id: str, threshold: float):
        """
        Internal logic to fetch vectors, cluster them, and trigger merges.
        """
        # 1. Fetch representatives
        # We scroll to get vectors. For large videos, we'd use a better sampling strategy.
        try:
            points, _ = self.vector_store.client.scroll(
                collection_name=collection_name,
                scroll_filter=self.vector_store._build_filter({"video_id": video_id}),
                limit=150, # Check last 150 observations
                with_vectors=True
            )
        except Exception:
            return

        if not points: return

        # Group by Entity ID to get a "Centroid" or list of vectors per entity
        entity_vectors = {}
        for p in points:
            eid = self.get_canonical_id(p.payload['entity_id'])
            if eid not in entity_vectors: entity_vectors[eid] = []
            entity_vectors[eid].append(p.vector)

        # 2. Compare Entities
        # Simple centroid comparison
        unique_ids = list(entity_vectors.keys())
        centroids = [np.mean(entity_vectors[uid], axis=0) for uid in unique_ids]
        
        merged_in_this_pass = set()

        for i in range(len(unique_ids)):
            id_a = unique_ids[i]
            if id_a in merged_in_this_pass: continue

            for j in range(i + 1, len(unique_ids)):
                id_b = unique_ids[j]
                if id_b in merged_in_this_pass: continue
                
                # Cosine Similarity
                vec_a = centroids[i]
                vec_b = centroids[j]
                sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                
                if sim > threshold:
                    logger.info(f"🔄 Auto-Merging: {id_b} -> {id_a} (Sim: {sim:.3f})")
                    self._merge_entities_safe(keep_id=id_a, drop_id=id_b, video_id=video_id)
                    merged_in_this_pass.add(id_b)

    def _merge_entities_safe(self, keep_id: str, drop_id: str, video_id: str):
        """
        Perform a Robust Merge:
        1. Update Alias Map (RAM)
        2. Create Graph Link (Audit)
        3. Move Graph Relationships (History)
        4. Update Vector Payloads (Search)
        """
        # 1. Update In-Memory Map
        self._alias_map[drop_id] = keep_id

        # 2. Graph Rewiring (Async but critical)
        # We use APOC if available, or manual Cypher rewriting
        query = """
        MATCH (winner:Entity {id: $keep_id})
        MATCH (loser:Entity {id: $drop_id})
        
        // 1. Create audit trail
        MERGE (loser)-[:MERGED_INTO {ts: timestamp()}]->(winner)
        SET loser.active = false
        
        // 2. Move 'APPEARED_IN' relationships
        WITH winner, loser
        MATCH (loser)-[r:APPEARED_IN]->(clip)
        MERGE (winner)-[r2:APPEARED_IN {obs_id: r.obs_id}]->(clip)
        SET r2.ts_ms = r.ts_ms
        DELETE r
        
        // 3. Move 'SPOKE' relationships (Dialogue)
        WITH winner, loser
        MATCH (loser)-[r:SPOKE]->(dial)
        MERGE (winner)-[r2:SPOKE]->(dial)
        DELETE r
        
        // 4. Move 'MENTIONS' from Memories
        WITH winner, loser
        MATCH (mem)-[r:MENTIONS]->(loser)
        MERGE (mem)-[r2:MENTIONS]->(winner)
        DELETE r
        """
        
        self.graph_store.execute_async(query, {"keep_id": keep_id, "drop_id": drop_id})
        
        # 3. Vector Update (Qdrant)
        # Retag all vectors of the 'drop_id' to 'keep_id'
        for col in [self.collections["face"], self.collections["voice"], "text_memories"]:
            try:
                self.vector_store.update_point_entity(col, drop_id, keep_id, video_id)
            except Exception as e:
                # text_memories might not have entity_id in all points, ignore errors
                pass

    def detect_pov_operator(self, video_id: str):
        """
        Identify the 'Invisible Protagonist'.
        Logic: The entity that speaks the most but is seen the least (or never).
        """
        # Ensure consistency
        self.graph_store.flush()
        
        query = """
        MATCH (e:Entity {video_id: $video_id})
        WHERE NOT (e)-[:MERGED_INTO]->() // Ignore merged ghosts
        
        // Count appearances
        OPTIONAL MATCH (e)-[r_face:APPEARED_IN]->() WHERE r_face.obs_id CONTAINS 'face'
        WITH e, count(r_face) as face_count
        
        // Count speech
        OPTIONAL MATCH (e)-[r_voice:APPEARED_IN]->() WHERE r_voice.obs_id CONTAINS 'voice'
        WITH e, face_count, count(r_voice) as voice_count
        
        RETURN e.id as id, face_count, voice_count
        """
        
        results = self.graph_store.run_query(query, {"video_id": video_id})
        
        best_candidate = None
        max_voice = 0
        
        for r in results:
            fid = r['id']
            f_count = r['face_count']
            v_count = r['voice_count']
            
            # Heuristic: Lots of voice (>5 clips), Zero faces
            if v_count > 5 and f_count == 0:
                if v_count > max_voice:
                    max_voice = v_count
                    best_candidate = fid
        
        if best_candidate:
            logger.info(f"🕵️ POV DETECTED: {best_candidate} identified as Camera Operator.")
            # Tag in Graph
            self.graph_store.execute_async("""
                MATCH (e:Entity {id: $eid})
                SET e.type = 'pov_user', e.alias = 'Camera Operator'
            """, {"eid": best_candidate})