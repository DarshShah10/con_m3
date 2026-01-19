import json
import logging
import re
import numpy as np
from typing import List, Dict, Any, Set
import openai
from sklearn.cluster import DBSCAN
from qdrant_client import models # Explicit import for filters

from conclave.core.schemas import (
    FaceObservation,
    VoiceObservation,
    HierarchicalFrameObservation,
    MemoryNode,
    MemoryType,
)

logger = logging.getLogger("Conclave.ReasoningAgent")

class ReasoningAgent:
    def __init__(self, config: Dict[str, Any]):
        api_key = config.get("openai_api_key") or config.get("api_key")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o")
        self.max_retries = config.get("max_retries", 3)
        self.entity_pattern = re.compile(r"<(ent_[a-zA-Z0-9_\-]+)>")

    def _safe_json_load(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```[a-zA-Z]*\n?", "", cleaned)
            cleaned = cleaned.rstrip("```")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    def _validate_grounding(self, content: str, valid_ids: Set[str]) -> List[str]:
        found_tags = set(self.entity_pattern.findall(content))
        valid_tags = found_tags.intersection(valid_ids)
        if len(found_tags) != len(valid_tags):
            invalid = found_tags - valid_ids
            if invalid:
                logger.warning(f"⚠️ Hallucination Warning: LLM invented {invalid}. Ignoring links.")
        return list(valid_tags)

    def generate_episodic_memory(
        self,
        video_id: str,
        clip_id: int,
        visuals: List[HierarchicalFrameObservation],
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> List[MemoryNode]:
        
        valid_entity_ids = set()
        for f in faces: 
            if f.entity_id: valid_entity_ids.add(f.entity_id)
        for v in voices: 
            if v.entity_id: valid_entity_ids.add(v.entity_id)

        context_str = self._prepare_multimodal_context(visuals, faces, voices)
        
        if not context_str.strip():
            return []

        system_prompt = f"""
You are the Conclave Grounding Engine.
Synthesize "Episodic Memories" from perception data.
RULES:
1. Output JSON ONLY: {{ "memories": [ "text...", ... ] }}
2. Use Tags: Refer to people by ID (e.g. <ent_face_123>).
3. Grounding: ONLY use IDs listed in 'ENTITIES PRESENT'.
"""
        user_prompt = f"""
Video ID: {video_id} | Clip: {clip_id}
=== ENTITIES PRESENT ===
{', '.join(valid_entity_ids) if valid_entity_ids else "None"}

=== PERCEPTION STREAM ===
{context_str}
"""
        response_data = self._llm_call_with_retry(system_prompt, user_prompt, "memories")
        
        nodes: List[MemoryNode] = []
        for text in response_data:
            linked_entities = self._validate_grounding(text, valid_entity_ids)
            nodes.append(MemoryNode(
                video_id=video_id, clip_id=clip_id, content=text,
                mem_type=MemoryType.EPISODIC, linked_entities=linked_entities
            ))

        return nodes

    def _prepare_multimodal_context(self, visuals, faces, voices) -> str:
        blocks = []
        
        for v in visuals:
            desc = v.scene_description or "Scene visible"
            ocr = ", ".join([t.content for obj in v.objects for t in obj.linked_text])
            ocr_text = f" | Visible Text: {ocr}" if ocr else ""
            blocks.append(f"[Time {v.ts_ms}ms] Visual: {desc}{ocr_text}")

        entity_actions = {}
        
        for v in voices:
            if v.entity_id:
                if v.entity_id not in entity_actions: entity_actions[v.entity_id] = []
                entity_actions[v.entity_id].append(f"Spoke: '{v.asr_text}'")

        for f in faces:
            if f.entity_id:
                if f.entity_id not in entity_actions: entity_actions[f.entity_id] = []
                if not any("Appeared" in a for a in entity_actions[f.entity_id]):
                    entity_actions[f.entity_id].append(f"Appeared (Quality: {f.quality_score:.1f})")

        for eid, actions in entity_actions.items():
            blocks.append(f"[Entity <{eid}>] " + " | ".join(actions))

        return "\n".join(blocks)

    def consolidate_semantic_memory(self, video_id: str, engine_instance) -> int:
        """Clusters episodes to form high-level semantic facts."""
        points = []
        offset = None
        
        # Explicit Filter Construction to prevent 400 Bad Request
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id)),
                models.FieldCondition(key="type", match=models.MatchValue(value="episodic"))
            ]
        )

        while True:
            try:
                res, offset = engine_instance.vector_store.client.scroll(
                    collection_name="text_memories",
                    scroll_filter=scroll_filter,
                    limit=50, # Reduced batch size for safety
                    with_vectors=True,
                    with_payload=True,
                    offset=offset
                )
                points.extend(res)
                if offset is None: break
            except Exception as e:
                logger.error(f"Semantic Consolidation Scroll Error: {e}")
                break
            
        if not points: return 0
        
        vectors = np.array([p.vector for p in points])
        if len(vectors) < 3: return 0

        # Cluster events
        clustering = DBSCAN(eps=0.4, min_samples=2, metric="cosine").fit(vectors)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        new_memories = []
        for label in range(n_clusters):
            cluster_indices = [i for i, x in enumerate(labels) if x == label]
            cluster_texts = [points[i].payload['content'] for i in cluster_indices]
            
            prompt = f"""
            You are an Expert Video Analyst.
            I will give you a list of raw observations from a sequence of events.
            Your job is to synthesize them into a SINGLE, DETAILED Semantic Fact.
            
            Format: Narrative style.
            Focus on: Who, What, Where, and Intent.
            
            Raw Observations:
            {json.dumps(cluster_texts[:15])}
            
            Output JSON: {{'fact': "The narrative description..."}}
            """
            fact = self._llm_call_with_retry("You are an Expert Video Analyst. Output JSON: {'fact': string}", prompt, "fact")
            
            if fact:
                if isinstance(fact, list): fact = fact[0]
                new_memories.append(MemoryNode(
                    video_id=video_id, clip_id=-1, content=fact, mem_type=MemoryType.SEMANTIC
                ))
                
        if new_memories:
            engine_instance.add_memories_batched(new_memories)
        return len(new_memories)

    def detect_equivalences(self, video_id: str, clip_id: int, context_str: str) -> List[Dict[str, str]]:
        system = """You are an Identity Detective. Look for evidence that a Face ID and Voice ID are the same person.
Output JSON: {"equivalences": [ {"source": "<ent_face_...>", "target": "<ent_voice_...>"} ]}"""
        data = self._llm_call_with_retry(system, f"Perception:\n{context_str}", "equivalences")
        
        valid = []
        for item in data:
            s, t = item.get("source", "").strip("<>"), item.get("target", "").strip("<>")
            if s and t and s != t: valid.append({"source": s, "target": t})
        return valid

    def _llm_call_with_retry(self, system: str, user: str, key: str) -> Any:
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                data = self._safe_json_load(response.choices[0].message.content)
                if key in data: return data[key]
            except Exception as e:
                logger.warning(f"LLM Error: {e}")
        return []

    # --- NEW: NAME EXTRACTION ---
    def extract_names_from_dialogue(self, video_id: str, engine) -> Dict[str, str]:
        """
        Scans dialogue for self-identification (e.g. "My name is John").
        Returns: {'SPEAKER_01': 'John', 'SPEAKER_00': 'Officer Smith'}
        """
        # Get recent dialogue
        query = """
        MATCH (e:Entity)-[:SPOKE]->(d:Dialogue)
        RETURN e.id as speaker, d.content as text
        """
        results = engine.graph_store.run_query(query)
        
        if not results: return {}
        
        # Group text by speaker
        speaker_text = {}
        for r in results:
            sid = r['speaker']
            if sid not in speaker_text: speaker_text[sid] = []
            speaker_text[sid].append(r['text'])
            
        # Analyze each speaker
        found_names = {}
        for speaker, texts in speaker_text.items():
            # Only analyze if they said enough words
            full_text = " ".join(texts)
            if len(full_text) < 20: continue
            
            prompt = f"""
            Analyze this transcript spoken by ONE person.
            Did they reveal their name or role?
            Look for: "I am...", "My name is...", "This is Officer...", or others calling them by name.
            
            Transcript: "{full_text[:2000]}"
            
            Output JSON: {{ "name": "Extracted Name OR Role", "confidence": float }}
            If unsure, return confidence 0.
            """
            
            try:
                res = self.client.chat.completions.create(
                    model=self.model, messages=[{"role":"user", "content":prompt}],
                    response_format={"type": "json_object"}
                )
                data = json.loads(res.choices[0].message.content)
                if data.get("confidence", 0) > 0.8:
                    found_names[speaker] = data["name"]
            except: pass
            
        return found_names

    # --- NEW: ENTITY PROFILING (Better Semantic Memory) ---
    def build_semantic_profiles(self, video_id: str, engine):
        """
        Instead of random clusters, we build a "Character Sheet" for every main entity.
        """
        # 1. Find Main Characters (Entities with > 3 memories)
        query = """
        MATCH (e:Entity {video_id: $vid})<-[:MENTIONS]-(m:Memory)
        WITH e, count(m) as cnt, collect(m.content) as memories
        WHERE cnt > 3
        RETURN e.id as id, e.alias as alias, memories
        """
        characters = engine.graph_store.run_query(query, {"vid": video_id})
        
        new_facts = []
        
        for char in characters:
            entity_id = char['id']
            name = char.get('alias', entity_id)
            
            prompt = f"""
            You are a Profiler. Build a Semantic Profile for: {name}
            Based on these specific episodic memories:
            {json.dumps(char['memories'][:20])}
            
            Create 1-2 high-level Semantic Facts.
            - NOT: "He opened the door." (Too specific)
            - YES: "He is aggressive and suspects the driver of hiding something." (Semantic/Psychological)
            - YES: "He is the vehicle owner and drives a Red Honda." (Factual)
            
            Output JSON: {{ "facts": ["fact 1", "fact 2"] }}
            """
            
            try:
                res = self.client.chat.completions.create(
                    model=self.model, messages=[{"role":"user", "content":prompt}],
                    response_format={"type": "json_object"}
                )
                data = json.loads(res.choices[0].message.content)
                
                for fact in data.get("facts", []):
                    new_facts.append(MemoryNode(
                        video_id=video_id, clip_id=-1,
                        content=f"[{name}] {fact}",
                        mem_type=MemoryType.SEMANTIC,
                        linked_entities=[entity_id]
                    ))
            except: pass
            
        # Ingest the profiles
        if new_facts:
            logger.info(f"🧠 Generated {len(new_facts)} Semantic Profiles.")
            # Use the "Smart Ingest" to avoid duplicates
            for mem in new_facts:
                engine.ingest_semantic_memory(mem)