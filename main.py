import os
import sys
import json
import logging
import argparse
import torch
import numpy as np
import concurrent.futures
import queue
import time
import cv2
import ffmpeg
from typing import Dict, Any

# Dynamic path handling
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

# Now the imports should work
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager
from conclave.agent.reasoning import ReasoningAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Conclave.Orchestrator")

class ConclaveOrchestrator:
    def __init__(self, config_path: str, video_id: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # 1. Initialize Engine (Fast)
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # 2. Initialize Identity Manager (Fast)
        self.identity_manager = IdentityManager(
            self.engine.vector_store, 
            self.engine.graph_store, 
            self.config.get("processing", {})
        )
        
        # 3. PARALLEL MODEL LOADING (The Startup Fix)
        # Instead of waiting for one model to load before starting the next,
        # we load them all simultaneously.
        logger.info("🚀 Bootstrapping AI Models in PARALLEL...")
        t0 = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_face = executor.submit(self._load_face_model)
            future_voice = executor.submit(self._load_voice_model)
            future_scene = executor.submit(self._load_scene_model)
            
            self.face_proc = future_face.result()
            self.voice_proc = future_voice.result()
            self.scene_proc = future_scene.result()
            
        logger.info(f"✅ All Models Loaded in {time.time() - t0:.2f}s")
        
        # 4. Initialize Reasoning Agent
        self.reasoning_agent = ReasoningAgent(self.config.get("api", {}))

    def _load_face_model(self):
        from conclave.perception.vision.face import FaceProcessor
        logger.info("-> Loading FaceProcessor...")
        return FaceProcessor(self.config.get("processing", {}))

    def _load_voice_model(self):
        from conclave.perception.audio.voice import VoiceProcessor
        logger.info("-> Loading VoiceProcessor...")
        return VoiceProcessor(self.config.get("processing", {}))

    def _load_scene_model(self):
        from conclave.perception.vision.scene import SceneProcessor
        logger.info("-> Loading SceneProcessor...")
        return SceneProcessor(self.config.get("processing", {}))

    def run_pipeline(self, video_path: str, window_size: int = 30, overlap: int = 5):
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return

        logger.info(f"🎬 STARTING PIPELINE: {self.video_id}")
        
        # Get total duration using ffprobe (fastest)
        try:
            probe = ffmpeg.probe(video_path)
            total_duration = float(probe['format']['duration'])
        except Exception as e:
            logger.error(f"FFProbe failed: {e}")
            return

        # ----------------------------------------------------------------
        # 1. High-Speed Producer (FFmpeg Pipe)
        # ----------------------------------------------------------------
        # Increase buffer to 5 to ensure GPU is never starved
        clip_queue = queue.Queue(maxsize=5)
        
        def producer():
            curr = 0.0
            idx = 0
            while curr < total_duration:
                try:
                    # Extract directly to memory (No Disk I/O)
                    data = self._fast_extract(video_path, curr, window_size)
                    clip_queue.put({"id": idx, "data": data, "start_time": curr})
                    curr += (window_size - overlap)
                    idx += 1
                except Exception as e:
                    logger.error(f"Producer error at {curr}s: {e}")
                    break
            clip_queue.put(None)

        prefetcher = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        prefetcher.submit(producer)

        # ----------------------------------------------------------------
        # 2. Inference Consumer (A40 Optimized)
        # ----------------------------------------------------------------
        # Use a small thread pool to run Scene/Face/Voice in parallel per-clip
        perception_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        while True:
            # If queue is empty, GPU is starving -> Bad.
            if clip_queue.empty():
                logger.warning("⚠️ GPU Starvation Warning: Waiting for video decode...")
            
            item = clip_queue.get()
            if item is None:
                break
                
            clip_id = item["id"]
            data = item["data"]
            current_start = item["start_time"]
            
            logger.info(f"⚡ Processing Clip {clip_id} [{current_start:.1f}s] (Buffer: {clip_queue.qsize()})...")
            
            # --- Parallel Tasks: Scene / Face / Voice ---
            def task_scene():
                # Scene: run every ~5.0s. With extraction at 5 FPS, stride=25 -> 5s.
                stride = 25
                selected = data["raw_frames"][::stride]
                if not selected:
                    return []
                interval_ms = int((1000 / 5) * stride)
                return self.scene_proc.process_batch(
                    selected, self.video_id, clip_id, int(current_start * 1000), interval_ms
                )

            def task_face():
                # Face: target ~0.5s (2 FPS). With 5 FPS extraction, stride=2 gives ~2.5 FPS (good balance).
                stride = 2
                selected = data["raw_frames"][::stride]
                if not selected:
                    return []
                return self.face_proc.extract_from_frames(selected, self.video_id, clip_id)

            def task_voice():
                if data.get("audio_bytes"):
                    return self.voice_proc.process_clip_audio(data["audio_bytes"], self.video_id, clip_id)
                return []

            future_scene = perception_executor.submit(task_scene)
            future_face = perception_executor.submit(task_face)
            future_voice = perception_executor.submit(task_voice)

            visuals = future_scene.result()
            faces = future_face.result()
            voices = future_voice.result()

            # Commit visuals
            for v in visuals:
                self.engine.vector_store.upsert(
                    "visual_memories", v.obs_id, v.clip_embedding,
                    {"video_id": self.video_id, "clip_id": clip_id, "desc": v.detected_objects[0]}
                )

            # Faces: cluster & resolve
            for f in self.face_proc.cluster_clip_faces(faces):
                self.identity_manager.resolve_face(f)
                self.engine.ingest_face(f)
                self.identity_manager.register_observation(f)

            # Voices
            for v in voices:
                self.identity_manager.resolve_voice(v)
                self.identity_manager.register_observation(v)

            # --- D. REASONING ---
            episodes = self.reasoning_agent.generate_episodic_memory(
                self.video_id, clip_id, visuals, faces, voices
            )
            if episodes:
                self.engine.add_memories_batched(episodes)

            if clip_id % 2 == 0: 
                self.identity_manager.link_modalities(self.video_id)
            
            # Optimized Cleanup: Only clear cache if VRAM is actually tight
            # On A40 (48GB), we barely need this.
            if clip_id % 50 == 0:
                torch.cuda.empty_cache()

        perception_executor.shutdown()
        prefetcher.shutdown()
        logger.info("✅ Pipeline Execution Complete.")

    def _fast_extract(self, video_path: str, start: float, duration: float):
        """
        Extracts raw frames and audio bytes directly into RAM using FFmpeg pipes.
        10x faster than MoviePy because it avoids re-encoding and temp files.
        """
        clip_data = {"audio_bytes": None, "raw_frames": []}
        
        # 1. Extract Audio to RAM (pcm_s16le wav)
        try:
            out, _ = (
                ffmpeg
                .input(video_path, ss=start, t=duration)
                .output('pipe:', format='wav', acodec='pcm_s16le', ar='16000', ac='1', loglevel="quiet")
                .run(capture_stdout=True)
            )
            clip_data["audio_bytes"] = out
        except ffmpeg.Error:
            pass # No audio stream or error

        # 2. Extract Frames to RAM (decoding to BGR numpy)
        # Using OpenCV VideoCapture is faster for sequential reading than random seek
        # But for clips, seeking is needed.
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25 # Fallback
        
        target_fps = 5
        step = max(1, int(fps / target_fps))
        
        frames_needed = int(duration * target_fps)
        frames_read = 0

        count = 0
        while frames_read < frames_needed:
            ret, frame = cap.read()
            if not ret:
                break

            # Take every Nth frame based on step
            if count % step == 0:
                # Resize to max-height 640 to save memory and speed up CNNs
                h, w = frame.shape[:2]
                if h > 640:
                    scale = 640 / float(h)
                    new_w = int(w * scale)
                    frame = cv2.resize(frame, (new_w, 640), interpolation=cv2.INTER_AREA)

                clip_data["raw_frames"].append(frame)
                frames_read += 1

            count += 1

            # Safety break if we go way past duration
            if (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) > (start + duration + 1):
                break
                
        cap.release()
        return clip_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    args = parser.parse_args()
    ConclaveOrchestrator("configs/api_config.json", args.video_id).run_pipeline(args.video)