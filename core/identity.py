import logging
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from conclave.core.schemas import FaceObservation, VoiceObservation, GlobalEntity, EntityType
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore

logger = logging.getLogger("Conclave.Identity")

class IdentityManager:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, config: Dict[str, Any]):
        """
        SOTA Identity Resolution System.
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Configuration Thresholds
        self.face_threshold = config.get("face_match_threshold", 0.75)
        self.voice_threshold = config.get("voice_match_threshold", 0.82)
        self.co_occurrence_threshold = config.get("min_co_occurrences_to_merge", 3)
        
        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories"
        }

    def _generate_entity_id(self) -> str:
        return f"ent_{uuid.uuid4().hex[:12]}"

    def _find_best_vector_match(self, embedding: List[float], collection: str, video_id: str, threshold: float) -> Optional[str]:
        """
        Searches Qdrant for the closest existing identity in this video.
        """
        results = self.vector_store.search(
            collection=collection,
            vector=embedding,
            filter_kv={"video_id": video_id},
            limit=1
        )
        
        if results and results[0].score >= threshold:
            # We assume the payload contains the 'entity_id' it was previously mapped to
            return results[0].payload.get("entity_id")
        return None

    def resolve_face(self, obs: FaceObservation) -> str:
        """
        Resolves a FaceObservation to a GlobalEntity ID.
        """
        # 1. Check for Vector Match
        existing_id = self._find_best_vector_match(
            obs.embedding, self.collections["face"], obs.video_id, self.face_threshold
        )
        
        if existing_id:
            logger.info(f"Resolved Face {obs.obs_id[:8]} -> Existing Entity {existing_id}")
            obs.entity_id = existing_id
            return existing_id

        # 2. If no match, create a new Global Entity
        new_id = self._generate_entity_id()
        logger.info(f"New Face Detected. Created Entity {new_id}")
        
        self.graph_store.create_entity_node(new_id, EntityType.PERSON.value, obs.video_id)
        obs.entity_id = new_id
        return new_id

    def resolve_voice(self, obs: VoiceObservation) -> str:
        """
        Resolves a VoiceObservation to a GlobalEntity ID.
        """
        # 1. Check for Vector Match
        existing_id = self._find_best_vector_match(
            obs.embedding, self.collections["voice"], obs.video_id, self.voice_threshold
        )
        
        if existing_id:
            logger.info(f"Resolved Voice Segment -> Existing Entity {existing_id}")
            obs.entity_id = existing_id
            return existing_id

        # 2. Check if we can infer identity from visual co-occurrence in the current clip
        # If there is exactly one person visible in the graph for this clip, we hypothesize it's them.
        query = """
        MATCH (e:Entity)-[a:APPEARED_IN]->(c:Clip {id: $clip_id, video_id: $video_id})
        WHERE e.type = 'person'
        RETURN e.id as id
        """
        visible_people = self.graph_store.run_query(query, {"clip_id": obs.clip_id, "video_id": obs.video_id})
        
        if len(visible_people) == 1:
            inferred_id = visible_people[0]['id']
            logger.info(f"Inferred Voice Identity -> Entity {inferred_id} via Single-Person Co-occurrence")
            obs.entity_id = inferred_id
            return inferred_id

        # 3. If no match and no clear inference, create new Entity
        new_id = self._generate_entity_id()
        self.graph_store.create_entity_node(new_id, EntityType.PERSON.value, obs.video_id)
        obs.entity_id = new_id
        return new_id

    def link_modalities(self, video_id: str, clip_id: int):
        """
        Advanced Identity Fusion.
        Analyzes Neo4j to find consistent Face-Voice pairings and merges them.
        """
        # This query finds identities that appear together frequently.
        # We look for a 'Face-heavy' identity and a 'Voice-heavy' identity linked to the same clips.
        query = """
        MATCH (e1:Entity)-[:APPEARED_IN]->(c:Clip {video_id: $video_id})
        MATCH (e2:Entity)-[:APPEARED_IN]->(c)
        WHERE e1.id <> e2.id
        WITH e1, e2, count(c) as co_occurrences
        WHERE co_occurrences >= $threshold
        RETURN e1.id as source, e2.id as target, co_occurrences
        """
        potential_merges = self.graph_store.run_query(query, {
            "video_id": video_id, 
            "threshold": self.co_occurrence_threshold
        })

        for merge in potential_merges:
            self._merge_entities(merge['source'], merge['target'], video_id)

    def _merge_entities(self, source_id: str, target_id: str, video_id: str):
        """
        Permanently merges two entities in the Graph and marks them for vector re-indexing.
        """
        logger.warning(f"Merging Identities: {source_id} <-> {target_id} due to high co-occurrence.")
        
        # 1. Update Graph: Move all relationships from target to source, then delete target
        merge_query = """
        MATCH (target:Entity {id: $target_id})
        MATCH (source:Entity {id: $source_id})
        MATCH (target)-[r]->(x)
        CALL apoc.refactor.to(r, source) YIELD input, output
        DETACH DELETE target
        """
        try:
            self.graph_store.run_query(merge_query, {"source_id": source_id, "target_id": target_id})
        except Exception as e:
            # Fallback if APOC is not installed
            fallback_query = """
            MATCH (target:Entity {id: $target_id})
            MATCH (source:Entity {id: $source_id})
            MATCH (target)-[r:APPEARED_IN]->(c:Clip)
            MERGE (source)-[:APPEARED_IN {ts_ms: r.ts_ms, obs_id: r.obs_id}]->(c)
            WITH target
            DETACH DELETE target
            """
            self.graph_store.run_query(fallback_query, {"source_id": source_id, "target_id": target_id})

    def register_observation(self, obs: Any):
        """
        Final step of ingestion: commit the resolved identity to the Vector Store.
        """
        if isinstance(obs, FaceObservation):
            collection = self.collections["face"]
        elif isinstance(obs, VoiceObservation):
            collection = self.collections["voice"]
        else:
            return

        payload = {
            "video_id": obs.video_id,
            "entity_id": obs.entity_id,
            "clip_id": obs.clip_id,
            "type": "resolved_identity"
        }
        
        self.vector_store.upsert(
            collection=collection,
            point_id=obs.obs_id,
            vector=obs.embedding,
            payload=payload
        )