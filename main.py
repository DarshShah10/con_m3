import os
import sys
import json
import logging
import argparse
import concurrent.futures
import queue
import cv2
import ffmpeg

import os
import sys

# Add parent directory to path for package imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

    
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager
from conclave.agent.reasoning import ReasoningAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("Conclave.Orchestrator")

class ConclaveOrchestrator:
    def __init__(self, config_path: str, video_id: str):
        with open(config_path, "r") as f: self.config = json.load(f)
        self.video_id = video_id
        
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        self.identity_manager = IdentityManager(self.engine.vector_store, self.engine.graph_store, self.config.get("processing", {}))
        self.reasoning_agent = ReasoningAgent(self.config.get("api", {}))

        # Parallel Load
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            self.scene_proc = executor.submit(self._load_scene).result()
            self.face_proc = executor.submit(self._load_face).result()
            self.voice_proc = executor.submit(self._load_voice).result()

    def _load_scene(self):
        from conclave.perception.vision.scene import AdvancedSceneProcessor
        logger.info("-> Loading SceneProcessor...")
        conf = self.config.get("processing", {}).copy()
        if "gemini" in self.config: conf["gemini"] = self.config["gemini"]
        return AdvancedSceneProcessor(conf)

    def _load_face(self):
        from conclave.perception.vision.face import FaceProcessor
        return FaceProcessor(self.config.get("processing", {}))

    def _load_voice(self):
        from conclave.perception.audio.voice import VoiceProcessor
        return VoiceProcessor(self.config.get("processing", {}))

    def run_pipeline(self, video_path: str):
        if not os.path.exists(video_path): return logger.error("Video not found")
        
        clip_queue = queue.Queue(maxsize=3)
        
        def producer():
            try: duration = float(ffmpeg.probe(video_path)['format']['duration'])
            except: duration = 60.0
            curr = 0.0
            idx = 0
            while curr < duration:
                # Extract 30s clips at 2 FPS
                data = self._fast_extract(video_path, curr, 30.0)
                clip_queue.put({"id": idx, "data": data, "start": curr})
                curr += 30.0
                idx += 1
            clip_queue.put(None)

        concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(producer)

        while True:
            item = clip_queue.get()
            if item is None: break
            
            clip_id = item["id"]
            data = item["data"]
            start_ts = item["start"]
            
            logger.info(f"⚡ Processing Clip {clip_id} ({start_ts:.1f}s)...")

            # 1. Advanced Vision
            # Raw frames are at 2 FPS, so we process them all
            hierarchical_obs = self.scene_proc.process_batch(
                data["raw_frames"], self.video_id, clip_id, int(start_ts*1000), 500
            )
            
            for obs in hierarchical_obs:
                # Ingest Graph Data
                self.engine.graph_store.ingest_hierarchical_obs(obs)
                # Ingest Vector Data
                if obs.scene_description:
                    self.engine.vector_store.upsert(
                        "visual_memories", obs.obs_id, obs.clip_embedding,
                        {"video_id": self.video_id, "clip_id": clip_id, "desc": obs.scene_description}
                    )
                for obj in obs.objects:
                    for txt in obj.linked_text:
                        logger.info(f"🔗 LINKED: '{obj.label}' -> Text '{txt.content}'")

            # 2. Faces & Voices
            # Extract 1 FPS for faces (stride 2)
            faces = self.face_proc.extract_from_frames(data["raw_frames"][::2], self.video_id, clip_id)
            voices = self.voice_proc.process_clip_audio(data["audio_bytes"], self.video_id, clip_id)

            for f in faces:
                self.identity_manager.resolve_face(f)
                self.identity_manager.register_observation(f)
            
            for v in voices:
                # 1. Who is speaking? (Speaker ID)
                self.identity_manager.resolve_voice(v)
                self.identity_manager.register_observation(v)
                
                # 2. What did they say? (Dialogue Content)
                # 🔥 NEW: Ingest the meaning of the text
                self.engine.ingest_dialogue_event(
                    self.video_id, 
                    clip_id, 
                    v.entity_id, 
                    v.asr_text, 
                    v.ts_ms
                )
            
            # --- 🔥 SOTA IDENTITY MAINTENANCE ---
            # Every 5 clips (approx 2.5 minutes of video), clean up the graph
            if clip_id > 0 and clip_id % 5 == 0:
                logger.info("🧠 Running Identity Consolidation & POV Detection...")
                # 1. Merge fragmented voices (Fixes "8 voices")
                self.identity_manager.consolidate_identities(self.video_id)
                # 2. Identify Body Cam User
                self.identity_manager.detect_pov_operator(self.video_id)
            # ------------------------------------
            
            # 3. Reasoning (Episodic Memory)
            episodes = self.reasoning_agent.generate_episodic_memory(
                self.video_id, clip_id, hierarchical_obs, faces, voices
            )
            if episodes: self.engine.add_memories_batched(episodes)

        logger.info("✅ Pipeline Complete.")

    def _fast_extract(self, video_path, start, duration):
        clip = {"audio_bytes": None, "raw_frames": []}
        try:
            out, _ = ffmpeg.input(video_path, ss=start, t=duration).output('pipe:', format='wav', acodec='pcm_s16le', ar='16000', ac='1', loglevel="quiet").run(capture_stdout=True)
            clip["audio_bytes"] = out
        except: pass

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        
        # Force 2 FPS extraction
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(fps / 2)) 
        
        count = 0
        read_frames = 0
        while read_frames < (duration * 2):
            ret, frame = cap.read()
            if not ret: break
            if count % step == 0:
                h, w = frame.shape[:2]
                if h > 640:
                    scale = 640/h
                    frame = cv2.resize(frame, (int(w*scale), 640))
                clip["raw_frames"].append(frame)
                read_frames += 1
            count += 1
        return clip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    args = parser.parse_args()
    ConclaveOrchestrator("configs/api_config.json", args.video_id).run_pipeline(args.video)