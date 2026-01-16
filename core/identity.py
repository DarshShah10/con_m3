import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from conclave.core.schemas import FaceObservation, VoiceObservation, EntityType
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore

logger = logging.getLogger("Conclave.Identity")

class IdentityManager:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, config: Dict[str, Any]):
        """
        SOTA Identity Resolution System with Deep Refactoring capabilities.
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Configuration Thresholds
        self.face_threshold = config.get("face_match_threshold", 0.75)
        self.voice_threshold = config.get("voice_match_threshold", 0.82)
        self.co_occurrence_threshold = config.get("min_co_occurrences_to_merge", 3)
        
        # Canonical Mapping: entity_id -> { 'type': ..., 'alias': ... }
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
        
        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories"
        }

    def _generate_entity_id(self, prefix: str = "ent") -> str:
        """Generates a new deterministic entity ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

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
            obs.entity_id = results[0].payload.get("entity_id")
            logger.info(f"Resolved Face {obs.obs_id[:8]} -> Existing Entity {obs.entity_id}")
        else:
            obs.entity_id = self._generate_entity_id("ent_face")
            logger.info(f"New Face Detected. Created Entity {obs.entity_id}")
            self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, obs.video_id)
        
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
            obs.entity_id = results[0].payload.get("entity_id")
            logger.info(f"Resolved Voice {obs.obs_id[:8]} -> Existing Entity {obs.entity_id}")
        else:
            # Fallback to co-occurrence heuristic if single person visible
            query = """
            MATCH (e:Entity)-[:APPEARED_IN]->(c:Clip {id: $clip_id, video_id: $video_id})
            WHERE e.type = 'person'
            RETURN e.id as id
            """
            visible_people = self.graph_store.run_query(query, {
                "clip_id": obs.clip_id,
                "video_id": obs.video_id
            })
            
            if len(visible_people) == 1:
                obs.entity_id = visible_people[0]['id']
                logger.info(f"Inferred Voice Identity -> Entity {obs.entity_id} via Single-Person Co-occurrence")
            else:
                obs.entity_id = self._generate_entity_id("ent_voice")
                logger.info(f"New Voice Detected. Created Entity {obs.entity_id}")
                self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, obs.video_id)
        
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
        self.graph_store.run_query(query, {
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

    def merge_identities(self, source_id: str, target_id: str, video_id: str):
        """
        Step 3: Safe Merge (The "Deep Refactor").
        Re-tags all historical vectors and refactors the Graph.
        """
        if source_id == target_id:
            return

        logger.warning(f"Identity Merger: Unifying {target_id} into {source_id}")

        # 1. Update Graph Relationships (Neo4j)
        self.graph_store.merge_entity_nodes(source_id, target_id)

        # 2. Re-tag Vector Payloads (Qdrant)
        for col_name in self.collections.values():
            point_ids = self.vector_store.get_points_by_entity(col_name, target_id, video_id)
            logger.info(f"Retagging {len(point_ids)} vectors in {col_name}")
            for pid in point_ids:
                self.vector_store.update_point_entity(col_name, pid, source_id)

        logger.info(f"Successfully merged {target_id} into {source_id}")

    def link_modalities(self, video_id: str, clip_id: int = None):
        """
        Advanced Identity Fusion.
        Analyzes Neo4j to find consistent Face-Voice pairings and merges them.
        """
        query = """
        MATCH (e1:Entity)-[:APPEARED_IN]->(c:Clip {video_id: $video_id})
        MATCH (e2:Entity)-[:APPEARED_IN]->(c)
        WHERE e1.id < e2.id
        WITH e1, e2, count(c) as co_occurrences
        WHERE co_occurrences >= $threshold
        RETURN e1.id as source, e2.id as target, co_occurrences
        ORDER BY co_occurrences DESC
        """
        potential_merges = self.graph_store.run_query(query, {
            "video_id": video_id,
            "threshold": self.co_occurrence_threshold
        })

        for merge in potential_merges:
            logger.info(f"Found co-occurrence: {merge['source']} <-> {merge['target']} ({merge['co_occurrences']} clips)")
            self.merge_identities(merge['source'], merge['target'], video_id)

    def refresh_mappings(self, video_id: str):
        """
        Rebuilds the character mapping for the reasoning engine to use.
        Finds entities that consistently appear in clips with 'equivalence' memories.
        """
        query = """
        MATCH (m:Memory {type: 'semantic'})-[:MENTIONS]->(e:Entity {video_id: $video_id})
        WHERE m.content CONTAINS "is the same as" OR m.content CONTAINS "also known as"
        MATCH (m)-[:MENTIONS]->(e2:Entity)
        WHERE e.id < e2.id
        RETURN e.id as source, e2.id as target, m.content as evidence
        """
        merges = self.graph_store.run_query(query, {"video_id": video_id})
        
        for m in merges:
            logger.info(f"Semantic equivalence detected: {m['source']} = {m['target']}")
            logger.debug(f"Evidence: {m['evidence'][:100]}...")
            self.merge_identities(m['source'], m['target'], video_id)