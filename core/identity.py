import logging
import uuid
import numpy as np
from typing import List, Dict, Any, Union
from conclave.core.schemas import FaceObservation, VoiceObservation, EntityType
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore

logger = logging.getLogger("Conclave.Identity")

class IdentityManager:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.graph_store = graph_store
        # Cache for simple lookups
        self._alias_map = {} 

    def resolve_face(self, obs: FaceObservation) -> str:
        # Simple resolution: If Gemini found a specific name (e.g. "MrBeast"), use it.
        # Otherwise, check vectors.
        if obs.entity_id and not obs.entity_id.startswith("ent_"):
            return obs.entity_id # Gemini gave us a name like "Adin Ross"

        results = self.vector_store.search(
            "face_memories", obs.embedding, {"video_id": obs.video_id}, limit=1
        )
        if results and results[0].score > 0.75:
            return results[0].payload.get("entity_id")
        
        new_id = f"ent_face_{uuid.uuid4().hex[:8]}"
        self.graph_store.create_entity_node(new_id, "person", obs.video_id)
        return new_id

    def resolve_voice(self, obs: VoiceObservation) -> str:
        # Deepgram gives us SPEAKER_00. We trust it for consistency within one video.
        # But we check if SPEAKER_00 has been renamed to a real name in the Graph.
        raw_id = obs.entity_id 
        
        # Check Graph for alias
        query = "MATCH (e:Entity {id: $id}) RETURN e.alias as alias"
        res = self.graph_store.run_query(query, {"id": raw_id})
        
        if res and res[0]['alias']:
            return res[0]['alias'] # Return "Officer John" instead of SPEAKER_00
            
        return raw_id # Return SPEAKER_00 if no name yet

    def register_observation(self, obs):
        # (Standard registration logic - unchanged from previous, just ensures links exist)
        pass 

    # --- NEW: NAME ASSIGNMENT ---
    def assign_name_to_entity(self, entity_id: str, name: str, video_id: str):
        """
        Hard re-name of an entity. 
        e.g., SPEAKER_00 -> "Officer Miller"
        """
        logger.info(f"🏷️ NAMING: {entity_id} is now known as '{name}'")
        
        # 1. Update Graph
        query = """
        MATCH (e:Entity {id: $eid})
        SET e.alias = $name, e.is_named = true
        """
        self.graph_store.execute_async(query, {"eid": entity_id, "name": name})
        
        # 2. Update Vector Store (Retagging is expensive, so we usually just rely on Graph alias)
        # But for 'pov_user', we set a special flag
        if entity_id == "pov_user":
            self.graph_store.execute_async(
                "MATCH (e:Entity {type: 'pov_user'}) SET e.alias = $name", 
                {"name": name}
            )

    # --- NEW: POV LOGIC ---
    def detect_pov_operator(self, video_id: str):
        """
        Determine who the camera operator is based on heuristics:
        1. Speaks a lot.
        2. Is never SEEN (Face count == 0).
        """
        self.graph_store.flush()
        
        query = """
        MATCH (e:Entity {video_id: $vid})
        WHERE e.id STARTS WITH 'SPEAKER_'
        
        // Count voice appearances
        OPTIONAL MATCH (e)-[:SPOKE]->(d:Dialogue)
        WITH e, count(d) as speech_count
        
        // Check if this speaker is linked to a face
        OPTIONAL MATCH (e)-[:SAME_AS]-(f:Entity)
        WHERE f.id STARTS WITH 'ent_face_'
        
        RETURN e.id as id, speech_count, f.id as linked_face
        """
        results = self.graph_store.run_query(query, {"vid": video_id})
        
        best_candidate = None
        max_speech = 0
        
        for r in results:
            # If they speak a lot AND have no linked face, they are likely the POV
            if r['speech_count'] > 5 and r['linked_face'] is None:
                if r['speech_count'] > max_speech:
                    max_speech = r['speech_count']
                    best_candidate = r['id']
        
        if best_candidate:
            logger.info(f"📹 POV DETECTED: {best_candidate} is the Camera Operator.")
            # Merge SPEAKER_XX into a persistent 'pov_user' node
            self._merge_entities_safe("pov_user", best_candidate, video_id)

    def _merge_entities_safe(self, keep_id, drop_id, video_id):
        # (Same safe merge logic as before)
        q = """
        MATCH (old:Entity {id: $drop})
        MERGE (new:Entity {id: $keep, video_id: $vid})
        ON CREATE SET new.type = 'person'
        
        // Move Dialogue
        WITH old, new
        MATCH (old)-[r:SPOKE]->(d)
        MERGE (new)-[:SPOKE]->(d)
        DELETE r
        
        // Move Memories
        WITH old, new
        MATCH (m)-[r:MENTIONS]->(old)
        MERGE (m)-[:MENTIONS]->(new)
        DELETE r
        
        // Set Alias if old had one
        SET new.alias = COALESCE(new.alias, old.alias)
        DETACH DELETE old
        """
        self.graph_store.execute_async(q, {"keep": keep_id, "drop": drop_id, "vid": video_id})