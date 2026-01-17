import cv2
import torch
import numpy as np
import logging
import base64
import os
import concurrent.futures
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO
from transformers import SiglipVisionModel, SiglipProcessor
import easyocr
import openai
from core.schemas import VisualObservation

logger = logging.getLogger("Conclave.Vision.Scene")

class SceneProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 1. SETUP GEMINI API
        gemini_conf = config.get("gemini", {})
        api_key = gemini_conf.get("api_key")
        base_url = gemini_conf.get("base_url")
        model = gemini_conf.get("model", "gemini-2.5-flash-lite-preview-02-05")

        self.client = None
        self.vlm_model_name = model
        
        if api_key and base_url:
            logger.info(f"🌐 Connecting to Gemini API: {model}")
            self.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=10.0,
                max_retries=1  # Gemini is usually stable, 1 retry is okay
            )
        else:
            logger.warning("⚠️ Gemini API credentials missing. Captions will be skipped.")
        
        # Logic: Caption every 3rd frame (Every ~3 seconds at 1FPS)
        self.caption_interval = 3
        
        logger.info(f"⚡ Loading Lightweight Stack (YOLO + SigLIP) on {self.device}...")

        # 2. SIGLIP (Embeddings) - Keep on GPU
        self.siglip_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(self.device, dtype=self.torch_dtype).eval()
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        
        # 3. OCR (Local)
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'), verbose=False)

        # 4. YOLO (Local) - With Warmup
        self.yolo_model = YOLO("yolo11s.pt")
        if self.device.type == 'cuda':
            # Run dummy inference to initialize CUDA context to prevent asserts later
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.yolo_model(dummy, verbose=False, device=self.device)
            except Exception as e:
                logger.warning(f"YOLO Warmup warning: {e}")

    def _encode_image(self, image_np: np.ndarray) -> str:
        h, w = image_np.shape[:2]
        # Resize if too big (speed optimization for API upload)
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode('.jpg', image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return base64.b64encode(buffer).decode('utf-8')

    def _get_caption_gemini(self, img: np.ndarray) -> str:
        if not self.client: return "No API Key"
        try:
            b64 = self._encode_image(img)
            resp = self.client.chat.completions.create(
                model=self.vlm_model_name,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Describe this scene concisely in one sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=60
            )
            if resp.choices:
                content = resp.choices[0].message.content
                # 🔥 ADD THIS TO SEE THE CAPTION
                logger.info(f"👀 GEMINI CAPTION: {content}")
                return content
            return "No description."
        except Exception as e:
            logger.warning(f"Gemini API Error: {e}")
            return "Visual processing skipped."

    def process_batch(self, frames_np: List[np.ndarray], video_id: str, clip_id: int, start_ts: int, interval_ms: int):
        if not frames_np: return []

        # --- 1. SANITIZE INPUTS ---
        # Filter out corrupted frames (0-byte) to prevent CUDA crashes
        valid_frames = []
        valid_indices = []
        for idx, f in enumerate(frames_np):
            if f is not None and f.size > 0 and f.shape[0] > 10 and f.shape[1] > 10:
                valid_frames.append(f)
                valid_indices.append(idx)
        
        if not valid_frames:
            return []

        # --- 2. API CALLS (Parallel) ---
        # Pick keyframes (0, 3, 6...)
        key_imgs = [valid_frames[i] for i in range(0, len(valid_frames), self.caption_interval)]
        
        # Use ThreadPool to send requests to Google while GPU is busy
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            key_captions = list(pool.map(self._get_caption_gemini, key_imgs))

        # --- 3. LOCAL GPU WORK (YOLO + SigLIP) ---
        pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in valid_frames]
        
        # Batch SigLIP
        with torch.no_grad():
            inputs = self.siglip_processor(images=pil_imgs, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
            out = self.siglip_model(**inputs)
            embeds = out.pooler_output if hasattr(out, 'pooler_output') else out.last_hidden_state.mean(dim=1)
            norms = torch.linalg.norm(embeds, ord=2, dim=1, keepdim=True)
            visual_vecs = (embeds / (norms + 1e-6)).tolist()

        # Batch YOLO
        try:
            yolo_results = self.yolo_model(valid_frames, verbose=False, stream=False, device=self.device)
        except Exception as e:
            logger.error(f"YOLO Batch Failed: {e}")
            yolo_results = [None] * len(valid_frames)

        # --- 4. AGGREGATE ---
        final_obs = []
        for i, original_idx in enumerate(valid_indices):
            # Map back to caption
            key_idx = min(i // self.caption_interval, len(key_captions) - 1)
            caption = key_captions[key_idx] if key_captions else "Scene"
            
            # Extract Objects
            objects = []
            if yolo_results[i] is not None:
                y_res = yolo_results[i]
                img_area = y_res.orig_shape[0] * y_res.orig_shape[1]
                objects = [y_res.names[int(b.cls)] for b in y_res.boxes if ((b.xywh[0][2]*b.xywh[0][3])/img_area) > 0.01]

            obs = VisualObservation(
                video_id=video_id, clip_id=clip_id, ts_ms=start_ts + (original_idx * interval_ms),
                clip_embedding=visual_vecs[i], ocr_tokens=[], detected_objects=[caption] + objects
            )
            obs.__dict__['spatial_metadata'] = {"dense_description": caption}
            final_obs.append(obs)

        return final_obs