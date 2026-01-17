import json
import logging
import re
from typing import List, Dict, Any, Set
import openai
import numpy as np
from sklearn.cluster import DBSCAN

from conclave.core.schemas import (
    FaceObservation,
    VoiceObservation,
    HierarchicalFrameObservation,
    MemoryNode,
    MemoryType,
)

logger = logging.getLogger("Conclave.ReasoningAgent")

class ReasoningAgent:
    """
    Grounded Reasoning Engine.
    Ensures that memories generated are:
    1. Valid JSON
    2. Grounded in actual observations (No imaginary entities)
    3. Strictly Typed (Episodic vs Semantic)
    """

    def __init__(self, config: Dict[str, Any]):
        api_key = config.get("openai_api_key") or config.get("api_key")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o") # Default to stronger model for reasoning
        self.max_retries = config.get("max_retries", 3)

        # Matches <ent_...> tags
        self.entity_pattern = re.compile(r"<(ent_[a-zA-Z0-9_\-]+)>")

    def _safe_json_load(self, raw: str) -> Any:
        """Robust JSON parser that handles common LLM markdown errors."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```[a-zA-Z]*\n?", "", cleaned)
            cleaned = cleaned.rstrip("```")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    def _validate_grounding(self, content: str, valid_ids: Set[str]) -> List[str]:
        """
        Extracts entity tags from text and verifies they actually exist in this clip.
        Returns a list of VALID linked entities.
        """
        found_tags = set(self.entity_pattern.findall(content))
        
        # Intersection: Only keep tags that are actually in the scene
        valid_tags = found_tags.intersection(valid_ids)
        
        if len(found_tags) != len(valid_tags):
            invalid = found_tags - valid_ids
            # We don't discard the memory, but we warn and don't link the invalid tags in the graph
            if invalid:
                logger.warning(f"⚠️ Hallucination Warning: LLM invented entities {invalid}. Ignoring links.")
        
        return list(valid_tags)

    def generate_episodic_memory(
        self,
        video_id: str,
        clip_id: int,
        visuals: List[HierarchicalFrameObservation],
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> List[MemoryNode]:
        
        # 1. Gather all legitimate IDs visible/audible in this clip
        valid_entity_ids = set()
        for f in faces: 
            if f.entity_id: valid_entity_ids.add(f.entity_id)
        for v in voices: 
            if v.entity_id: valid_entity_ids.add(v.entity_id)

        # 2. Build Context
        context_str = self._prepare_multimodal_context(visuals, faces, voices)
        
        if not context_str.strip():
            return []

        # 3. Prompt
        system_prompt = f"""
You are the Conclave Grounding Engine.
Your goal is to synthesize distinct "Episodic Memories" from the raw perception data.

CRITICAL RULES:
1. **Output JSON ONLY**: {{ "memories": [ "memory text...", ... ] }}
2. **Use Tags**: You MUST refer to people by their ID tags provided (e.g., <ent_face_123>).
3. **Grounding**: Do NOT invent entity IDs. Only use those listed in the 'Entities Present' section.
4. **Style**: Objective, factual, past tense. e.g., "Person <ent_x> entered the room and argued with <ent_y>."
"""

        user_prompt = f"""
Video ID: {video_id} | Clip: {clip_id}

=== ENTITIES PRESENT (USE THESE IDS) ===
{', '.join(valid_entity_ids) if valid_entity_ids else "None detected"}

=== PERCEPTION STREAM ===
{context_str}

Generate episodic memories now.
"""

        # 4. Call LLM
        response_data = self._llm_call_with_retry(system_prompt, user_prompt, "memories")
        
        # 5. Convert to Objects
        nodes: List[MemoryNode] = []
        for text in response_data:
            # Check for hallucinated entities
            linked_entities = self._validate_grounding(text, valid_entity_ids)
            
            nodes.append(
                MemoryNode(
                    video_id=video_id,
                    clip_id=clip_id,
                    content=text,
                    mem_type=MemoryType.EPISODIC, # Ensure strictly typed enum
                    linked_entities=linked_entities,
                )
            )

        return nodes

    def _prepare_multimodal_context(self, visuals, faces, voices) -> str:
        blocks = []
        
        # Visual Summary
        for v in visuals:
            desc = v.scene_description or "Scene visible"
            ocr = ", ".join([t.content for obj in v.objects for t in obj.linked_text])
            ocr_text = f" | Visible Text: {ocr}" if ocr else ""
            blocks.append(f"[Time {v.ts_ms}ms] Visual: {desc}{ocr_text}")

        # Action Summary (Grouping by Entity)
        # We group by entity to help the LLM understand "Agent continuity"
        entity_actions = {}
        
        for v in voices:
            if v.entity_id:
                if v.entity_id not in entity_actions: entity_actions[v.entity_id] = []
                entity_actions[v.entity_id].append(f"Spoke: '{v.asr_text}'")
                
        for eid, actions in entity_actions.items():
            blocks.append(f"[Entity <{eid}>] " + " | ".join(actions))

        return "\n".join(blocks)

    def _llm_call_with_retry(self, system: str, user: str, key: str) -> List[str]:
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1 # Low temp for factual consistency
                )
                data = self._safe_json_load(response.choices[0].message.content)
                if key in data and isinstance(data[key], (list, str)): # Allow string return (for fact)
                    return data[key]
            except Exception as e:
                logger.warning(f"Reasoning LLM error: {e}")
        return []

    # --- NEW: SYSTEMATIC SEMANTIC CLUSTERING (M3-Style) ---
    def consolidate_semantic_memory(self, video_id: str, engine_instance) -> int:
        """
        Retrieves ALL episodic memories, clusters them by vector similarity,
        and generates high-level 'Semantic' memories for each cluster.
        """
        # 1. Fetch all episodic memory vectors from Qdrant
        # We scroll through the collection
        points = []
        offset = None
        while True:
            res, offset = engine_instance.vector_store.client.scroll(
                collection_name="text_memories",
                scroll_filter=engine_instance.vector_store._build_filter({"video_id": video_id, "type": "episodic"}),
                limit=100,
                with_vectors=True,
                with_payload=True,
                offset=offset
            )
            points.extend(res)
            if offset is None: break
            
        if not points: return 0
        
        vectors = np.array([p.vector for p in points])
        
        # 2. DBSCAN Clustering (Find clusters of similar events)
        # eps=0.3 is strict, 0.5 is loose. M3 uses dynamic thresholds, we stick to 0.4
        clustering = DBSCAN(eps=0.4, min_samples=3, metric="cosine").fit(vectors)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"🧩 Found {n_clusters} semantic clusters from {len(points)} episodes.")
        
        new_semantic_memories = []
        
        for label in range(n_clusters):
            # Get all texts in this cluster
            cluster_indices = [i for i, x in enumerate(labels) if x == label]
            cluster_texts = [points[i].payload['content'] for i in cluster_indices]
            
            # 3. Summarize Cluster (LLM)
            summary_prompt = f"""
            Synthesize these related observations into a SINGLE high-level semantic fact.
            Use Entity IDs <ent_...> where possible.
            
            Observations:
            {json.dumps(cluster_texts[:10])} # Limit to 10 to fit context
            """
            
            fact = self._llm_call_with_retry(
                "You are a Memory Consolidator. Output JSON: {'fact': string}", 
                summary_prompt, 
                "fact"
            )
            
            if fact:
                if isinstance(fact, list): fact = fact[0] # Handle list return
                
                # Check if this fact already exists to avoid duplicates? 
                # For now, just create it. The vector store handles ID dedupe if we used deterministic IDs.
                new_semantic_memories.append(MemoryNode(
                    video_id=video_id,
                    clip_id=-1, # Semantic memories are timeless or span whole video
                    content=fact,
                    mem_type=MemoryType.SEMANTIC
                ))
                
        # 4. Ingest Semantic Memories
        if new_semantic_memories:
            engine_instance.add_memories_batched(new_semantic_memories)
            
        return len(new_semantic_memories)

    def detect_equivalences(self, video_id: str, clip_id: int, context_str: str) -> List[Dict[str, str]]:
        """
        M3-Style Identity Logic:
        Asks the LLM specifically if it can deduce that two different IDs 
        actually belong to the same person based on context (e.g. lip sync).
        """
        system_prompt = """
        You are an Identity Detective.
        Analyze the perception stream. 
        Look for evidence that a Face ID and a Voice ID belong to the same person.
        (e.g., "Face <ent_face_1> is moving lips while Voice <ent_voice_2> is speaking").
        
        Output JSON: {"equivalences": [ {"source": "<ent_face_...>", "target": "<ent_voice_...>"} ]}
        Return empty list if unsure.
        """
        
        user_prompt = f"Perception Data:\n{context_str}"
        
        data = self._llm_call_with_retry(system_prompt, user_prompt, "equivalences")
        
        valid_merges = []
        for item in data:
            src = item.get("source")
            tgt = item.get("target")
            # Strip brackets if LLM left them in
            if src: src = src.strip("<>")
            if tgt: tgt = tgt.strip("<>")
            
            if src and tgt and src != tgt:
                valid_merges.append({"source": src, "target": tgt})
                
        return valid_merges
