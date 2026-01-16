import json
import logging
import re
from typing import List, Dict, Any, Tuple
import openai  # Using OpenAI as the standard interface for GPT-4o/Gemini
from conclave.core.schemas import (
    FaceObservation, VoiceObservation, VisualObservation, 
    MemoryNode, MemoryType
)

logger = logging.getLogger("Conclave.Reasoning")

class ReasoningEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        State-of-the-Art Reasoning Engine.
        Uses structured prompting to ensure Graph-LLM alignment.
        """
        self.api_key = config.get("api_key")
        self.model_name = config.get("model", "gpt-4o")
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Regex to capture UUIDs tagged by the LLM in format <ent_uuid>
        self.entity_pattern = re.compile(r"<([a-f0-9\-]{12,36}|ent_[a-z0-9]+)>")

    def _prepare_multimodal_context(
        self, 
        visuals: List[VisualObservation], 
        faces: List[FaceObservation], 
        voices: List[VoiceObservation]
    ) -> str:
        """
        Consolidates raw perception into a structured textual context for the LLM.
        """
        context_blocks = []
        
        # 1. Visual Context (Florence-2 descriptions + YOLO + PaddleOCR)
        for v in visuals:
            spatial = getattr(v, 'spatial_metadata', {})
            desc = spatial.get('dense_description', 'No description')
            ocr = ", ".join([t['text'] for t in v.ocr_tokens])
            context_blocks.append(
                f"[Time: {v.ts_ms}ms] Visual: {desc}. Text seen: [{ocr}]."
            )

        # 2. Identified People (InsightFace UUIDs)
        person_map = {}
        for f in faces:
            if f.entity_id:
                person_map.setdefault(f.entity_id, []).append(f"Face seen at {f.ts_ms}ms")
        
        # 3. Identified Voices (WeSpeaker UUIDs + Whisper ASR)
        for v in voices:
            if v.entity_id:
                person_map.setdefault(v.entity_id, []).append(
                    f"Spoke at {v.ts_ms}ms: '{v.asr_text}'"
                )

        # Build Person Metadata Block
        for eid, events in person_map.items():
            event_str = " | ".join(events)
            context_blocks.append(f"[Entity: <{eid}>] Activities: {event_str}")

        return "\n".join(context_blocks)

    def generate_episodic_memory(
        self, 
        video_id: str,
        clip_id: int,
        visuals: List[VisualObservation], 
        faces: List[FaceObservation], 
        voices: List[VoiceObservation]
    ) -> List[MemoryNode]:
        """
        Generates episodic memories using SOTA structured reasoning.
        """
        context = self._prepare_multimodal_context(visuals, faces, voices)
        
        system_prompt = """
        You are the 'Conclave' Multimodal Reasoning Engine. 
        Your task is to generate short, factual descriptions of video clips.
        
        RULES:
        1. Use EXACT entity tags like <ent_uuid> when referring to people or unique objects.
        2. Be concise. Focus on actions, interactions, and visible text.
        3. Output your response as a JSON list of strings.
        4. If an entity tag is provided in context, you MUST use it instead of generic names.
        """
        
        user_prompt = f"Video ID: {video_id}, Clip ID: {clip_id}\nPerception Data:\n{context}\n\nGenerate episodic memories:"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # OpenAI/Gemini often wrap in a key like "memories"
            raw_output = json.loads(response.choices[0].message.content)
            memory_texts = raw_output.get("memories", raw_output.get("episodic_memories", []))
            if isinstance(raw_output, list): memory_texts = raw_output

            nodes = []
            for text in memory_texts:
                # Extract UUIDs mentioned in the text to build Graph edges
                mentions = list(set(self.entity_pattern.findall(text)))
                
                nodes.append(MemoryNode(
                    video_id=video_id,
                    clip_id=clip_id,
                    content=text,
                    mem_type=MemoryType.EPISODIC,
                    linked_entities=mentions
                ))
            return nodes

        except Exception as e:
            logger.error(f"Reasoning failure: {e}")
            return []

    def distillation_pass(
        self, 
        video_id: str, 
        episodic_nodes: List[MemoryNode], 
        existing_semantic_context: str
    ) -> List[MemoryNode]:
        """
        Second Pass: Converts episodes into Semantic Memory (Long-term facts).
        """
        episodes_str = "\n".join([n.content for n in episodic_nodes])
        
        prompt = f"""
        Analyze these new events and existing world knowledge. 
        Extract long-term facts (Semantic Memory).
        
        Existing Knowledge: {existing_semantic_context}
        New Events: {episodes_str}
        
        Format: JSON list of strings. Include entity tags <ent_uuid>.
        Example: ["<ent_uuid> is the owner of the white dog.", "The office is located on the 5th floor."]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            raw_output = json.loads(response.choices[0].message.content)
            facts = raw_output.get("semantic_memories", raw_output.get("facts", []))
            
            semantic_nodes = []
            for fact in facts:
                mentions = list(set(self.entity_pattern.findall(fact)))
                semantic_nodes.append(MemoryNode(
                    video_id=video_id,
                    content=fact,
                    mem_type=MemoryType.SEMANTIC,
                    linked_entities=mentions
                ))
            return semantic_nodes
        except:
            return []