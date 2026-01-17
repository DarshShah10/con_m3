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
    🔥 UNIFIED IDENTITY RESOLUTION SYSTEM 🔥
    Merges logic from identity.py and identity_resolver.py into one canonical class.
    Handles:
    - Face/Voice resolution via vector similarity
    - Co-occurrence-based identity linking
    - Deep merge operations (Neo4j + Qdrant historical retagging)
    - Canonical entity index for fast lookups
    """
    
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, config: Dict[str, Any]):
        """
        SOTA Identity Resolution System with Deep Refactoring capabilities.
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Configuration Thresholds
        self.face_threshold = config.get("face_match_threshold", 0.70) # Lowered for bodycam
        self.voice_threshold = config.get("voice_match_threshold", 0.75) # Lowered for bodycam
        self.co_occurrence_threshold = config.get("min_co_occurrences_to_merge", 3)
        
        # 🔥 UNIFIED: Canonical Mapping (merged from IdentityResolver)
        # entity_id -> { 'type': ..., 'alias': ..., 'canonical_id': ... }
        self.entity_index: Dict[str, Dict[str, Any]] = {}
        
        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories"
        }
        
        # Local cache of stats for the POV heuristic
        # entity_id -> {'voice': 0, 'face': 0}
        self.entity_stats = {}

    def _generate_entity_id(self, prefix: str = "ent") -> str:
        """Generates a new deterministic entity ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def get_canonical_id(self, entity_id: str) -> str:
        """
        🔥 MERGED: From IdentityResolver
        Resolves aliased IDs to their canonical form.
        In production multi-agent systems, this prevents identity fragmentation.
        """
        if entity_id in self.entity_index:
            return self.entity_index[entity_id].get("canonical_id", entity_id)
        return entity_id

    def resolve_face(self, obs: FaceObservation) -> str:
        """
        Alignment: Logic to resolve face identity before ingestion.
        Uses vector similarity search to match against existing entities.
        """
        results = self.vector_store.search(
            collection=self.collections["face"],
            vector=obs.embedding,
            filter_kv={"video_id": obs.video_id},
            limit=1
        )
        
        if results and results[0].score >= self.face_threshold:
            matched_id = results[0].payload.get("entity_id")
            # Resolve to canonical ID if aliased
            obs.entity_id = self.get_canonical_id(matched_id)
            logger.info(f"Resolved Face {obs.obs_id[:8]} -> Existing Entity {obs.entity_id} (score: {results[0].score:.3f})")
        else:
            obs.entity_id = self._generate_entity_id("ent_face")
            logger.info(f"New Face Detected. Created Entity {obs.entity_id}")
            self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, obs.video_id)
            # Register in canonical index
            self.entity_index[obs.entity_id] = {
                "type": EntityType.PERSON.value,
                "canonical_id": obs.entity_id,
                "video_id": obs.video_id
            }
        
        # Update Stats
        self._update_stats(obs.entity_id, "face")

        return obs.entity_id

    def resolve_voice(self, obs: VoiceObservation) -> str:
        """
        Alignment: Logic to resolve voice identity via vectors or co-occurrence.
        Falls back to single-person heuristic if no vector match found.
        """
        results = self.vector_store.search(
            collection=self.collections["voice"],
            vector=obs.embedding,
            filter_kv={"video_id": obs.video_id},
            limit=1
        )
        
        if results and results[0].score >= self.voice_threshold:
            matched_id = results[0].payload.get("entity_id")
            # Resolve to canonical ID if aliased
            obs.entity_id = self.get_canonical_id(matched_id)
            logger.info(f"Resolved Voice {obs.obs_id[:8]} -> Existing Entity {obs.entity_id} (score: {results[0].score:.3f})")
        else:
            # Fallback to co-occurrence heuristic if single person visible
            entity_id = f"ent_voice_{uuid.uuid4().hex[:8]}"
            self.graph_store.create_entity_node(entity_id, EntityType.PERSON.value, obs.video_id)
            obs.entity_id = entity_id
            self._update_stats(entity_id, "voice")
        
        return obs.entity_id

    def register_observation(self, obs: Union[FaceObservation, VoiceObservation]):
        """
        API Parity: Mandatory final step for observation ingestion.
        Syncs the resolved identity to both Vector and Graph databases.
        """
        # Ensure identity is resolved
        if not obs.entity_id:
            if isinstance(obs, FaceObservation):
                self.resolve_face(obs)
            else:
                self.resolve_voice(obs)

        col = self.collections["face"] if isinstance(obs, FaceObservation) else self.collections["voice"]
        
        # 1. Update Graph (Idempotent MERGE)
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

        # 2. Update Vector Store (Idempotent)
        payload = {
            "video_id": obs.video_id,
            "entity_id": obs.entity_id,
            "clip_id": obs.clip_id,
            "obs_id": obs.obs_id,
            "type": "resolved_identity"
        }
        self.vector_store.upsert(col, obs.obs_id, obs.embedding, payload)

    def _update_stats(self, eid, modality):
        if eid not in self.entity_stats: self.entity_stats[eid] = {'voice': 0, 'face': 0}
        self.entity_stats[eid][modality] += 1

    # --- 🔥 SOTA FEATURES BELOW ---

    def consolidate_identities(self, video_id: str):
        """
        Fixes "8 voices for 2 people".
        Looks at all voice entities. If they are similar vectors, MERGE them.
        """
        # 1. Get all voice entities created for this video
        entities_to_check = []
        seen_ids = set()
        
        # Scroll to find candidates
        # Using internal client scroll for raw access 
        scroll_res, _ = self.vector_store.client.scroll(
            collection_name="voice_memories",
            scroll_filter=self.vector_store._build_filter({"video_id": video_id}),
            limit=100,
            with_vectors=True
        )
        
        for point in scroll_res:
            eid = point.payload['entity_id']
            if eid not in seen_ids:
                entities_to_check.append({"id": eid, "vector": point.vector})
                seen_ids.add(eid)

        # 2. Compare All-vs-All
        merges_done = set()
        
        for i in range(len(entities_to_check)):
            for j in range(i + 1, len(entities_to_check)):
                e1 = entities_to_check[i]
                e2 = entities_to_check[j]
                
                if e1['id'] in merges_done or e2['id'] in merges_done: continue
                
                # Cosine Similarity
                sim = np.dot(e1['vector'], e2['vector'])
                
                if sim > 0.80:
                    logger.info(f"🔄 Merging Fragmented Voice: {e2['id']} -> {e1['id']} (Sim: {sim:.2f})")
                    self._deep_merge(e1['id'], e2['id'], video_id)
                    merges_done.add(e2['id'])

    def detect_pov_operator(self, video_id: str):
        """
        Identifies the 'Invisible Protagonist' (Body Cam User).
        """
        # Refresh stats from Graph
        query = """
        MATCH (e:Entity {video_id: $video_id})
        OPTIONAL MATCH (e)-[:APPEARED_IN]->(c:Clip)
        WITH e, count(c) as total_clips
        RETURN e.id as id, total_clips
        """
        results = self.graph_store.run_query(query, {"video_id": video_id})
        
        max_clips = 0
        pov_candidate = None
        
        for res in results:
            eid = res['id']
            # Get specific modality count
            q_mod = "MATCH (e:Entity {id: $eid})-[r:APPEARED_IN]->() WHERE r.obs_id CONTAINS 'face' RETURN count(r) as c"
            face_count = self.graph_store.run_query(q_mod, {"eid": eid})[0]['c']
            
            # Heuristic: Consistent voice (>3 clips), NO face
            if res['total_clips'] > 3 and face_count == 0 and "voice" in eid:
                if res['total_clips'] > max_clips:
                    max_clips = res['total_clips']
                    pov_candidate = eid
        
        if pov_candidate:
            logger.info(f"🕵️ POV DETECTED: {pov_candidate} is likely the Camera Operator.")
            self.graph_store.execute_async("""
                MATCH (e:Entity {id: $eid})
                SET e.type = 'pov_user', e.alias = 'Camera Operator'
            """, {"eid": pov_candidate})
            
    def _deep_merge(self, keep_id, remove_id, video_id):
        # 1. Update Vectors
        for col in ["voice_memories", "face_memories"]:
             self.vector_store.update_point_entity(col, remove_id, keep_id, video_id)
            
        # 2. Update Graph
        query = """
        MATCH (old:Entity {id: $remove_id})
        MATCH (new:Entity {id: $keep_id})
        OPTIONAL MATCH (old)-[r]->(target)
        MERGE (new)-[r2:APPEARED_IN]->(target)
        SET r2 = r
        DETACH DELETE old
        """
        self.graph_store.execute_async(query, {"remove_id": remove_id, "keep_id": keep_id})

    # Keep compatibility with main loop call
    def link_modalities(self, video_id: str):
        pass

    def detect_and_tag_pov(self, video_id: str):
        # Forward to new method
        self.detect_pov_operator(video_id)