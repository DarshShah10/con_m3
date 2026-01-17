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
from conclave.core.schemas import HierarchicalFrameObservation, DetectedObject, LinkedText

logger = logging.getLogger("Conclave.Vision.Scene")

class AdvancedSceneProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 1. Gemini API Setup
        gemini_conf = config.get("gemini", {})
        self.api_key = gemini_conf.get("api_key")
        self.base_url = gemini_conf.get("base_url")
        self.client = None
        if self.api_key:
            logger.info("🌐 Connecting to Gemini...")
            self.client = openai.OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=10.0, max_retries=1
            )

        # 2. YOLO (Object Detection)
        self.yolo = YOLO("yolo11s.pt")
        # Warmup to prevent CUDA errors
        if self.device.type == 'cuda':
            self.yolo(np.zeros((640,640,3), dtype=np.uint8), verbose=False, device=self.device)
        
        # 3. EasyOCR (Text Reading)
        self.ocr = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'), verbose=False)
        
        # 4. SigLIP (Embeddings)
        self.siglip_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(self.device, dtype=self.torch_dtype).eval()
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

    def _enhance_for_ocr(self, img_crop: np.ndarray) -> np.ndarray:
        if img_crop.size == 0: return img_crop
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _process_single_frame(self, frame: np.ndarray, ts: int) -> HierarchicalFrameObservation:
        obs = HierarchicalFrameObservation(video_id="", clip_id=0, ts_ms=ts)
        
        # A. Detect Objects
        results = self.yolo(frame, verbose=False, device=self.device)[0]
        h_frame, w_frame = frame.shape[:2]

        for box in results.boxes:
            conf = float(box.conf)
            if conf < 0.5: continue
            
            label = results.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detected_obj = DetectedObject(label=label, confidence=conf, bbox=[x1, y1, x2, y2])

            # B. Hierarchical Logic: If object is big enough, scan for text
            obj_w = x2 - x1
            obj_h = y2 - y1
            
            if obj_w > 50 and obj_h > 20:
                pad_x, pad_y = int(obj_w * 0.05), int(obj_h * 0.05)
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w_frame, x2 + pad_x)
                crop_y2 = min(h_frame, y2 + pad_y)
                
                raw_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                clean_crop = self._enhance_for_ocr(raw_crop)
                
                try:
                    ocr_results = self.ocr.readtext(clean_crop, detail=1)
                    for (bbox, text, ocr_conf) in ocr_results:
                        if ocr_conf > 0.4 and len(text.strip()) > 1:
                            detected_obj.linked_text.append(LinkedText(
                                content=text, confidence=float(ocr_conf), bbox_relative=bbox
                            ))
                except: pass

            obs.objects.append(detected_obj)
        return obs

    def _get_gemini_description(self, img: np.ndarray) -> str:
        if not self.client: return None
        try:
            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64 = base64.b64encode(buffer).decode('utf-8')
            resp = self.client.chat.completions.create(
                model="gemini-2.0-flash-lite-preview-02-05",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Describe the scene concisely."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}],
                max_tokens=60
            )
            if resp.choices:
                content = resp.choices[0].message.content
                logger.info(f"👀 GEMINI: {content}")
                return content
        except Exception as e:
            logger.warning(f"Gemini API Error: {e}")
        return None

    def process_batch(self, frames: List[np.ndarray], video_id: str, clip_id: int, start_ms: int, interval_ms: int):
        observations = []
        pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        
        # 1. Batch Embeddings
        with torch.no_grad():
            inputs = self.siglip_processor(images=pil_imgs, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
            out = self.siglip_model(**inputs)
            embeds = out.pooler_output if hasattr(out, 'pooler_output') else out.last_hidden_state.mean(dim=1)
            norms = torch.linalg.norm(embeds, ord=2, dim=1, keepdim=True)
            visual_vecs = (embeds / (norms + 1e-6)).tolist()

        # 2. Logic Loop (2 FPS)
        for i, frame in enumerate(frames):
            ts = start_ms + (i * interval_ms)
            obs = self._process_single_frame(frame, ts)
            obs.video_id = video_id
            obs.clip_id = clip_id
            obs.clip_embedding = visual_vecs[i]

            # 3. Scene Description (Every 4th frame -> 2 seconds)
            if i % 4 == 0:
                obs.scene_description = self._get_gemini_description(frame)
            
            observations.append(obs)
            
        return observations