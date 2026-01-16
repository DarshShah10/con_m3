import os
import json
import logging
import argparse
import torch
import numpy as np
from typing import Dict, Any, List
from moviepy.editor import VideoFileClip

from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager
from conclave.agent.reasoning import ReasoningAgent # Note: Using ReasoningAgent logic
from conclave.perception.vision.face import FaceProcessor
from conclave.perception.audio.voice import VoiceProcessor
from conclave.perception.vision.scene import SceneProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Conclave.Main")

class ConclaveOrchestrator:
    def __init__(self, config_path: str, video_id: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        self.identity_manager = IdentityManager(self.engine.vector_store, self.engine.graph_store, self.config["processing"])
        
        self.face_proc = FaceProcessor(self.config["processing"])
        self.voice_proc = VoiceProcessor(self.config["processing"])
        self.scene_proc = SceneProcessor(self.config["processing"])
        self.reasoning_agent = ReasoningAgent(self.config["api"])

    def run_pipeline(self, video_path: str, window_size: int = 30, overlap: int = 5):
        logger.info(f"STARTING CONCLAVE: {self.video_id}")
        video_clip = VideoFileClip(video_path)
        total_duration = video_clip.duration
        
        current_start = 0.0
        clip_id = 0

        while current_start < total_duration:
            logger.info(f"Processing Clip {clip_id}...")
            # Extraction logic (assumed integrated into orchestrator)
            data = self._extract_clip_data(video_path, current_start, window_size)
            
            # 1. Perception & Identity
            faces = self.face_proc.extract_from_frames(data["frames_b64"], self.video_id, clip_id)
            for f in self.face_proc.cluster_clip_faces(faces):
                self.identity_manager.resolve_face(f)
                self.engine.ingest_face(f)
                self.identity_manager.register_observation(f)

            voices = self.voice_proc.process_clip_audio(data["audio_b64"], self.video_id, clip_id) if data["audio_b64"] else []
            for v in voices:
                self.identity_manager.resolve_voice(v)
                self.identity_manager.register_observation(v)

            # 2. Scene Logic
            visuals = []
            for i in [0, len(data["raw_frames"])//2, -1]:
                if abs(i) < len(data["raw_frames"]):
                    v_obs = self.scene_proc.process_frame(data["raw_frames"][i], self.video_id, clip_id, int(current_start*1000))
                    visuals.append(v_obs)
                    # Vector search ingestion
                    self.engine.vector_store.upsert("visual_memories", v_obs.obs_id, v_obs.clip_embedding, 
                                                   {"video_id": self.video_id, "clip_id": clip_id, "desc": v_obs.detected_objects[0]})

            # 3. Reasoning & Batched Storage
            episodes = self.reasoning_agent.generate_episodic_memory(self.video_id, clip_id, visuals, faces, voices)
            if episodes:
                self.engine.add_memories_batched(episodes)

            if clip_id % 2 == 0: self.identity_manager.link_modalities(self.video_id)

            current_start += (window_size - overlap)
            clip_id += 1
            torch.cuda.empty_cache() # Aggressive VRAM cleanup for 8GB

    def _extract_clip_data(self, video_path: str, start: float, duration: float):
        # Implementation from previous orchestrator (Complete and consistent)
        import cv2, base64
        clip_data = {"frames_b64": [], "audio_b64": None, "raw_frames": []}
        with VideoFileClip(video_path) as video:
            sub = video.subclip(start, min(start + duration, video.duration))
            for t in np.arange(0, sub.duration, 0.2): # 5 FPS
                f = cv2.cvtColor(sub.get_frame(t), cv2.COLOR_RGB2BGR)
                clip_data["raw_frames"].append(f)
                clip_data["frames_b64"].append(base64.b64encode(cv2.imencode('.jpg', f)[1]).decode('utf-8'))
            if sub.audio:
                temp_a = f"temp_{self.video_id}.wav"
                sub.audio.write_audiofile(temp_a, fps=16000, codec='pcm_s16le', logger=None)
                clip_data["audio_b64"] = base64.b64encode(open(temp_a, "rb").read()).decode('utf-8')
                os.remove(temp_a)
        return clip_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--video_id", required=True)
    args = parser.parse_args()
    ConclaveOrchestrator("configs/api_config.json", args.video_id).run_pipeline(args.video)