import os
import sys
import json
import logging
import argparse
import cv2
import ffmpeg

# Add parent directory to path so 'conclave' package can be found

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager
from conclave.agent.reasoning import ReasoningAgent
from conclave.core.schemas import VoiceObservation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("Conclave.Orchestrator")

class ConclaveOrchestrator:
    def __init__(self, config_path: str, video_id: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, "r") as f: self.config = json.load(f)
        self.video_id = video_id
        
        # 1. Engine & Data
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        self.identity_manager = IdentityManager(self.engine.vector_store, self.engine.graph_store, self.config.get("processing", {}))
        self.reasoning_agent = ReasoningAgent(self.config.get("api", {}))

        # 2. Processors
        self.audio_pipeline = self._load_audio()
        self.scene_proc = self._load_scene()
        self.face_proc = self._load_face()

    def _load_audio(self):
        from conclave.perception.audio.voice import GlobalAudioPipeline
        return GlobalAudioPipeline(self.config.get("processing", {}))

    def _load_scene(self):
        from conclave.perception.vision.scene import AdvancedSceneProcessor
        return AdvancedSceneProcessor(self.config.get("processing", {}))

    def _load_face(self):
        from conclave.perception.vision.face import FaceProcessor
        return FaceProcessor(self.config.get("processing", {}))

    def run_pipeline(self, video_path: str):
        if not os.path.exists(video_path): 
            logger.error(f"Video not found: {video_path}")
            return

        # --- PHASE 1: AUDIO & DIALOGUE (Pre-computation) ---
        logger.info("=== PHASE 1: AUDIO ANALYSIS ===")
        
        # Extract audio track to temp file
        audio_path = f"temp_full_audio_{self.video_id}.wav"
        self._extract_audio(video_path, audio_path)
        
        # Run SOTA Pipeline
        audio_data = self.audio_pipeline.process_full_video(audio_path, self.video_id)
        
        # Resolve Identities immediately (Map SPEAKER_01 -> ent_voice_7d8s)
        speaker_map = {} # { 'SPEAKER_01': 'ent_voice_real_id' }
        
        for raw_spk, embedding in audio_data["embeddings"].items():
            # Create a dummy observation to utilize IdentityManager logic
            dummy_obs = VoiceObservation(
                video_id=self.video_id, clip_id=-1, ts_ms=0,
                embedding=embedding, asr_text="", start_sec=0, end_sec=0
            )
            canonical_id = self.identity_manager.resolve_voice(dummy_obs)
            # Register it strictly so we know this embedding belongs to this ID
            dummy_obs.entity_id = canonical_id
            self.identity_manager.register_observation(dummy_obs)
            
            speaker_map[raw_spk] = canonical_id
            logger.info(f"🔗 Mapped {raw_spk} -> {canonical_id}")

        # Ingest Dialogue into Graph/Vector Store
        dialogue_timeline = audio_data["timeline"]
        for line in dialogue_timeline:
            # Swap temp ID for real ID
            real_id = speaker_map.get(line.entity_id, "UNKNOWN_SPEAKER")
            line.entity_id = real_id
            
            self.engine.ingest_dialogue_event(
                self.video_id, line.clip_id, real_id, line.text, line.start_ts
            )

        if os.path.exists(audio_path): os.remove(audio_path)

        # --- PHASE 2: VISUALS & REASONING (Streaming) ---
        logger.info("=== PHASE 2: VISUAL ANALYSIS ===")
        
        chunk_size = 30.0
        duration = self._get_duration(video_path)
        curr = 0.0
        clip_idx = 0
        
        while curr < duration:
            logger.info(f"🎞️ Processing Clip {clip_idx} ({curr:.1f}s - {curr+chunk_size:.1f}s)")
            
            # 1. Extract Frames
            clip_data = self._extract_frames(video_path, curr, chunk_size)
            if not clip_data: break
            
            # 2. Visual Perception
            hierarchical_obs = self.scene_proc.process_batch(
                clip_data, self.video_id, clip_idx, int(curr*1000), 500
            )
            
            # 3. Face Perception
            faces = self.face_proc.extract_from_frames(clip_data[::2], self.video_id, clip_idx)
            for f in faces:
                self.identity_manager.resolve_face(f)
                self.identity_manager.register_observation(f)

            # 4. Ingest Visuals
            for obs in hierarchical_obs:
                self.engine.graph_store.ingest_hierarchical_obs(obs)
                if obs.scene_description:
                    self.engine.vector_store.upsert(
                        "visual_memories", obs.obs_id, obs.clip_embedding,
                        {"video_id": self.video_id, "clip_id": clip_idx, "desc": obs.scene_description}
                    )

            # 5. Retrieve Dialogue Context for this specific clip
            # We filter the global timeline for lines occurring NOW
            clip_start_ms = int(curr * 1000)
            clip_end_ms = int((curr + chunk_size) * 1000)
            
            # Create Mock Voice Observations from the timeline for the Agent
            current_voices = []
            for line in dialogue_timeline:
                # Check overlap
                if not (line.end_ts < clip_start_ms or line.start_ts > clip_end_ms):
                    current_voices.append(VoiceObservation(
                        video_id=self.video_id, clip_id=clip_idx, ts_ms=line.start_ts,
                        embedding=[], # Not needed, identity already resolved
                        asr_text=line.text, start_sec=line.start_ts/1000, end_sec=line.end_ts/1000,
                        entity_id=line.entity_id
                    ))

            # 6. Flush & Reason
            self.engine.graph_store.flush()
            
            episodes = self.reasoning_agent.generate_episodic_memory(
                self.video_id, clip_idx, hierarchical_obs, faces, current_voices
            )
            
            if episodes:
                # E. Smart Semantic Ingestion (M3 Style)
                for mem in episodes:
                    # Robust check for Enum or String
                    is_semantic = False
                    if isinstance(mem.mem_type, str) and mem.mem_type == "semantic": is_semantic = True
                    elif hasattr(mem.mem_type, "value") and mem.mem_type.value == "semantic": is_semantic = True
                    
                    if is_semantic:
                         self.engine.ingest_semantic_memory(mem)
                    else:
                         self.engine.add_memory(mem)

            # F. Logic-Based Identity Merging (M3 Equivalence)
            try:
                merges = self.reasoning_agent.detect_equivalences(
                    self.video_id, clip_idx, 
                    self.reasoning_agent._prepare_multimodal_context(hierarchical_obs, faces, [])
                )
                
                for m in merges:
                    logger.info(f"🔗 Agent Deducted Identity Merge: {m['source']} == {m['target']}")
                    self.identity_manager._merge_entities_safe(
                        keep_id=m['source'], 
                        drop_id=m['target'], 
                        video_id=self.video_id
                    )
            except Exception as e:
                logger.error(f"Error in equivalence detection: {e}")

            curr += chunk_size
            clip_idx += 1

        # --- PHASE 3: CONSOLIDATION (The "Sleep" Phase) ---
        logger.info("=== PHASE 3: SEMANTIC CONSOLIDATION ===")
        # This mimics M3's memory_processing logic where it clusters nodes
        count = self.reasoning_agent.consolidate_semantic_memory(self.video_id, self.engine)
        logger.info(f"✅ Consolidated {count} high-level semantic facts.")

        self.engine.graph_store.close()
        logger.info("✅ Pipeline Complete.")

    def _extract_audio(self, video_path, out_path):
        try:
            ffmpeg.input(video_path).output(out_path, acodec='pcm_s16le', ar='16000', ac='1', loglevel="quiet").run(overwrite_output=True)
        except Exception as e:
            logger.error(f"FFmpeg audio extraction failed: {e}")

    def _get_duration(self, video_path):
        try:
            return float(ffmpeg.probe(video_path)['format']['duration'])
        except: return 0.0

    def _extract_frames(self, video_path, start, duration):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(fps / 2)) # 2 FPS
        
        frames = []
        max_frames = int(duration * 2)
        count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if count % step == 0:
                h, w = frame.shape[:2]
                if h > 640:
                    scale = 640/h
                    frame = cv2.resize(frame, (int(w*scale), 640))
                frames.append(frame)
            count += 1
        cap.release()
        return frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/api_config.json")
    args = parser.parse_args()
    
    ConclaveOrchestrator(args.config, args.video_id).run_pipeline(args.video)