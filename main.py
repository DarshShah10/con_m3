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
from moviepy.editor import VideoFileClip
# -------------------------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------------------------
# Add parent directory to sys.path to support 'conclave.' package imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------------------------------------------
# CORE IMPORTS (Package-style)
# -------------------------------------------------------------------------
try:
    from conclave.core.engine import ConclaveEngine
    from conclave.core.identity import IdentityManager
    from conclave.agent.reasoning import ReasoningAgent
    from conclave.perception.vision.face import FaceProcessor
    from conclave.perception.audio.voice import VoiceProcessor
    from conclave.perception.vision.scene import SceneProcessor
except ImportError as e:
    logger = logging.getLogger("Conclave.Bootstrap")
    logger.error(f"❌ Failed to import core components: {e}")
    logger.error(f"   Current sys.path: {sys.path}")
    sys.exit(1)
# -------------------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Conclave.Orchestrator")
# -------------------------------------------------------------------------
# ORCHESTRATOR CLASS
# -------------------------------------------------------------------------
class ConclaveOrchestrator:
    def __init__(self, config_path: str, video_id: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # 1. Initialize Engine
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # 2. Initialize Identity Manager
        self.identity_manager = IdentityManager(
            self.engine.vector_store, 
            self.engine.graph_store, 
            self.config.get("processing", {})
        )
        
        # --- 🔥 FIX: Inject API Token into Processing Config ---
        processing_conf = self.config.get("processing", {})
        api_conf = self.config.get("api", {})
        
        # Ensure HF token is available to Voice Processor
        if "hf_token" in api_conf:
            processing_conf["hf_token"] = api_conf["hf_token"]
        # -----------------------------------------------------

        # 3. Initialize Perception Models
        logger.info("🚀 Loading SOTA Perception Models (A40 Optimized)...")
        self.face_proc = FaceProcessor(processing_conf)
        self.voice_proc = VoiceProcessor(processing_conf)
        self.scene_proc = SceneProcessor(processing_conf)
        
        # 4. Initialize Reasoning Agent
        self.reasoning_agent = ReasoningAgent(self.config.get("api", {}))
    def run_pipeline(self, video_path: str, window_size: int = 30, overlap: int = 5):
        """
        High-Performance Producer-Consumer Pipeline.
        Decouples video extraction (CPU) from Model Inference (GPU).
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return
        logger.info(f"🎬 STARTING PIPELINE: {self.video_id}")
        video_clip = VideoFileClip(video_path)
        total_duration = video_clip.duration
        
        # ----------------------------------------------------------------
        # 1. Prefetching Setup (Producer)
        # ----------------------------------------------------------------
        # Buffer size 2 is sufficient to keep GPU busy without exploding RAM
        clip_queue = queue.Queue(maxsize=2) 
        
        def producer():
            curr = 0.0
            idx = 0
            while curr < total_duration:
                try:
                    # Extract raw data (Optimized: No Base64)
                    data = self._extract_clip_data_optimized(video_path, curr, window_size)
                    clip_queue.put({"id": idx, "data": data, "start_time": curr})
                    curr += (window_size - overlap)
                    idx += 1
                except Exception as e:
                    logger.error(f"Producer failed at {curr}s: {e}")
                    break
            clip_queue.put(None) # Sentinel to signal end
        # Start Producer in background thread
        prefetcher = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        prefetcher.submit(producer)
        # ----------------------------------------------------------------
        # 2. Inference Loop (Consumer)
        # ----------------------------------------------------------------
        while True:
            item = clip_queue.get()
            if item is None:
                break
                
            clip_id = item["id"]
            data = item["data"]
            current_start = item["start_time"]
            
            logger.info(f"⚡ Processing Clip {clip_id} [{current_start:.1f}s] (Buffer: {clip_queue.qsize()})...")
            
            # --- A. SCENE UNDERSTANDING (Batch GPU) ---
            # Sample every 10th frame (approx every 2s) for visual context
            stride = 10
            selected_frames = data["raw_frames"][::stride]
            
            visuals = []
            if selected_frames:
                visuals = self.scene_proc.process_batch(
                    selected_frames, 
                    self.video_id, 
                    clip_id, 
                    int(current_start*1000), 
                    200 * stride # Interval in ms
                )
                # Ingest Visual Vectors
                for v in visuals:
                    self.engine.vector_store.upsert(
                        "visual_memories", 
                        v.obs_id, 
                        v.clip_embedding, 
                        {"video_id": self.video_id, "clip_id": clip_id, "desc": v.detected_objects[0]}
                    )
            # --- B. FACE PERCEPTION (Optimized) ---
            # extract_from_frames now accepts raw numpy arrays (from previous fix)
            faces = self.face_proc.extract_from_frames(data["raw_frames"], self.video_id, clip_id)
            
            # Cluster & Ingest
            for f in self.face_proc.cluster_clip_faces(faces):
                self.identity_manager.resolve_face(f)
                self.engine.ingest_face(f)
                self.identity_manager.register_observation(f)
            # --- C. AUDIO PERCEPTION ---
            voices = []
            if data["audio_bytes"]:
                # Process raw bytes
                voices = self.voice_proc.process_clip_audio(data["audio_bytes"], self.video_id, clip_id)
                for v in voices:
                    self.identity_manager.resolve_voice(v)
                    self.identity_manager.register_observation(v)
            # --- D. REASONING (LLM) ---
            episodes = self.reasoning_agent.generate_episodic_memory(
                self.video_id, clip_id, visuals, faces, voices
            )
            
            if episodes:
                self.engine.add_memories_batched(episodes)
            # --- E. IDENTITY FUSION ---
            # Run graph-based merging every few clips
            if clip_id % 2 == 0: 
                self.identity_manager.link_modalities(self.video_id)
            
            # --- F. VRAM CLEANUP ---
            # Periodic cleanup to prevent fragmentation on long runs
            if clip_id % 10 == 0:
                 torch.cuda.empty_cache()
        prefetcher.shutdown()
        logger.info("✅ Pipeline Execution Complete.")
    def _extract_clip_data_optimized(self, video_path: str, start: float, duration: float):
        """
        Fast extraction using MoviePy + OpenCV. 
        Returns raw numpy arrays (BGR) and raw audio bytes.
        """
        clip_data = {"audio_bytes": None, "raw_frames": []}
        
        with VideoFileClip(video_path) as video:
            end = min(start + duration, video.duration)
            sub = video.subclip(start, end)
            
            # 1. Extract Audio
            if sub.audio:
                 temp_a = f"temp_{self.video_id}_{start:.2f}.wav"
                 # Verbose=False to keep logs clean
                 sub.audio.write_audiofile(temp_a, fps=16000, codec='pcm_s16le', logger=None, verbose=False)
                 with open(temp_a, "rb") as f:
                     clip_data["audio_bytes"] = f.read()
                 if os.path.exists(temp_a):
                    os.remove(temp_a)
            # 2. Extract Frames (5 FPS)
            # np.arange creates time points: 0.0, 0.2, 0.4...
            for t in np.arange(0, sub.duration, 0.2):
                f = sub.get_frame(t)
                # Convert RGB (MoviePy) -> BGR (OpenCV/InsightFace standard)
                f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                clip_data["raw_frames"].append(f_bgr)
                
        return clip_data
# -------------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Conclave: SOTA Multimodal Memory System")
    
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (mp4/mkv/mov)")
    parser.add_argument("--video_id", type=str, required=True, help="Unique ID for this video session")
    parser.add_argument("--config", type=str, default="configs/api_config.json", help="Path to config file")
    parser.add_argument("--window", type=int, default=30, help="Processing window size in seconds")
    parser.add_argument("--overlap", type=int, default=5, help="Window overlap in seconds")
    
    args = parser.parse_args()
    # 1. Config Validation
    if not os.path.exists(args.config):
        logger.error(f"❌ Config file not found: {args.config}")
        logger.error("   Run 'mkdir configs && nano configs/api_config.json' to create it.")
        sys.exit(1)
    # 2. Video Validation
    if not os.path.exists(args.video):
        logger.error(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    print(f"--- 🧠 CONCLAVE SYSTEM INITIALIZED ---")
    print(f"    Video: {args.video}")
    print(f"    ID:    {args.video_id}")
    print(f"--------------------------------------")
    try:
        orchestrator = ConclaveOrchestrator(
            config_path=args.config, 
            video_id=args.video_id
        )
        
        orchestrator.run_pipeline(
            video_path=args.video, 
            window_size=args.window, 
            overlap=args.overlap
        )
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user. Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"❌ CRITICAL PIPELINE FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()