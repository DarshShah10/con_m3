import os
import json
import logging
import argparse
import cv2
import base64
import torch
import numpy as np
from typing import Dict, Any, List

# Core System Imports
from conclave.core.schemas import (
    FaceObservation, VoiceObservation, VisualObservation, 
    MemoryNode, MemoryType
)
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager
from conclave.agent.reasoning import ReasoningEngine

# Perception Module Imports
from conclave.perception.vision.face import FaceProcessor
from conclave.perception.audio.voice import VoiceProcessor
from conclave.perception.vision.scene import SceneProcessor

# Utils
from moviepy.editor import VideoFileClip

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Conclave.Orchestrator")

class ConclaveOrchestrator:
    def __init__(self, config_path: str, video_id: str):
        """
        Principal-level Orchestrator. 
        Initializes the entire SOTA stack.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.video_id = video_id
        
        # 1. Initialize Storage Engine (Qdrant + Neo4j)
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # 2. Initialize Identity Manager
        self.identity_manager = IdentityManager(
            vector_store=self.engine.vector_store,
            graph_store=self.engine.graph_store,
            config=self.config.get("processing", {})
        )

        # 3. Initialize Perception Stack
        logger.info("Loading SOTA Perception Models...")
        self.face_proc = FaceProcessor(self.config.get("processing", {}))
        self.voice_proc = VoiceProcessor(self.config.get("processing", {}))
        self.scene_proc = SceneProcessor(self.config.get("processing", {}))

        # 4. Initialize Reasoning Engine
        self.reasoning_engine = ReasoningEngine(self.config.get("api", {}))

    def _extract_clip_data(self, video_path: str, start_sec: float, duration: float) -> Dict[str, Any]:
        """
        Extracts frames and audio from a specific window.
        """
        clip_data = {"frames_b64": [], "audio_b64": None, "raw_frames": []}
        
        with VideoFileClip(video_path) as video:
            subclip = video.subclip(start_sec, min(start_sec + duration, video.duration))
            
            # Extract Frames at 5 FPS for perception efficiency
            fps = 5
            for t in np.arange(0, subclip.duration, 1.0 / fps):
                frame = subclip.get_frame(t)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                clip_data["raw_frames"].append(frame_bgr)
                
                _, buffer = cv2.imencode('.jpg', frame_bgr)
                clip_data["frames_b64"].append(base64.b64encode(buffer).decode('utf-8'))

            # Extract Audio as Base64 WAV
            audio_path = f"temp_audio_{self.video_id}.wav"
            if subclip.audio:
                subclip.audio.write_audiofile(audio_path, fps=16000, codec='pcm_s16le', logger=None)
                with open(audio_path, "rb") as f:
                    clip_data["audio_b64"] = base64.b64encode(f.read()).decode('utf-8')
                os.remove(audio_path)
        
        return clip_data

    def run_pipeline(self, video_path: str, window_size: int = 30, overlap: int = 5):
        """
        The Main Integrated Loop. 
        Sliding window processing with identity fusion.
        """
        logger.info(f"Starting Conclave Pipeline for Video: {video_path}")
        
        video_clip = VideoFileClip(video_path)
        total_duration = video_clip.duration
        video_clip.close()

        current_start = 0.0
        clip_id = 0

        while current_start < total_duration:
            logger.info(f"--- Processing Clip {clip_id} [{current_start}s - {current_start + window_size}s] ---")
            
            # 1. Extraction
            data = self._extract_clip_data(video_path, current_start, window_size)
            
            # 2. Perception: Faces
            logger.info("Perception: Extracting and Clustering Faces...")
            face_obs_list = self.face_proc.extract_from_frames(data["frames_b64"], self.video_id, clip_id)
            face_obs_list = self.face_proc.cluster_clip_faces(face_obs_list)
            
            for f_obs in face_obs_list:
                # Identity Resolution
                self.identity_manager.resolve_face(f_obs)
                # Engine Ingestion (Storage + Graph Link)
                self.engine.ingest_face(f_obs)
                # Global Registration
                self.identity_manager.register_observation(f_obs)

            # 3. Perception: Voice
            voice_obs_list = []
            if data["audio_b64"]:
                logger.info("Perception: Diarizing and Transcribing Voice...")
                voice_obs_list = self.voice_proc.process_clip_audio(data["audio_b64"], self.video_id, clip_id)
                for v_obs in voice_obs_list:
                    # Identity Resolution (with Graph-based inference)
                    self.identity_manager.resolve_voice(v_obs)
                    # Storage
                    self.identity_manager.register_observation(v_obs)

            # 4. Perception: Scene (SigLIP + Florence-2 + YOLO11 + PaddleOCR)
            logger.info("Perception: Analyzing Scene Context...")
            visual_obs_list = []
            # Sample 3 key frames per 30s clip for scene understanding to maintain speed
            sample_indices = [0, len(data["raw_frames"]) // 2, len(data["raw_frames"]) - 1]
            for idx in sample_indices:
                if idx < len(data["raw_frames"]):
                    v_obs = self.scene_proc.process_frame(
                        data["raw_frames"][idx], self.video_id, clip_id, ts_ms=int(current_start * 1000)
                    )
                    visual_obs_list.append(v_obs)
                    # Visual Search Ingestion
                    self.engine.vector_store.upsert(
                        "visual_memories", v_obs.obs_id, v_obs.clip_embedding, 
                        {"video_id": self.video_id, "clip_id": clip_id, "description": v_obs.detected_objects[0]}
                    )

            # 5. Reasoning: Dual-Pass Knowledge Generation
            logger.info("Reasoning: Generating Episodic and Semantic Memories...")
            episodes = self.reasoning_engine.generate_episodic_memory(
                self.video_id, clip_id, visual_obs_list, face_obs_list, voice_obs_list
            )
            
            for ep_node in episodes:
                # Generate Embedding for Vector Search
                # In production, we'd use a dedicated batch embedder; here we use the ReasoningEngine's model
                ep_node.embedding = self.engine.vector_store.client.embed(ep_node.content) # Placeholder for actual embed call
                self.engine.add_memory(ep_node)

            # 6. Global Identity Fusion
            # Periodically merge identities based on co-occurrence graph
            if clip_id % 2 == 0:
                self.identity_manager.link_modalities(self.video_id, clip_id)

            # Advance Window
            current_start += (window_size - overlap)
            clip_id += 1

        logger.info("--- Pipeline Execution Complete. Knowledge Base Rebuilt. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conclave Orchestrator")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--video_id", type=str, required=True, help="Unique ID for this video session")
    parser.add_argument("--config", type=str, default="configs/api_config.json", help="Path to config file")
    
    args = parser.parse_args()
    
    orchestrator = ConclaveOrchestrator(config_path=args.config, video_id=args.video_id)
    orchestrator.run_pipeline(args.video)