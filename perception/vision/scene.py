import cv2
import torch
import base64
import numpy as np
import logging
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq, SiglipVisionModel, SiglipProcessor
from paddleocr import PaddleOCR
from conclave.core.schemas import VisualObservation

logger = logging.getLogger("Conclave.Vision.Scene")

class SceneProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        High-Performance Vision Pipeline optimized for 8GB VRAM.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 1. SigLIP Base (Significantly smaller than so400m, fits in ~300MB VRAM)
        self.siglip_model = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(self.device, dtype=self.torch_dtype).eval()
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

        # 2. Florence-2 Base (Densely Grounded Captioning)
        self.florence_model = AutoModelForVision2Seq.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        ).to(self.device, dtype=self.torch_dtype).eval()
        self.florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

        # 3. YOLO11s (Small variant: High speed, low VRAM footprint)
        self.yolo_model = YOLO("yolo11s.pt") 

        # 4. PaddleOCR (Offloaded to CPU to preserve VRAM for LLMs)
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)

    @torch.no_grad()
    def _get_siglip_embedding(self, pil_img: Image) -> List[float]:
        inputs = self.siglip_processor(images=pil_img, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
        outputs = self.siglip_model(**inputs)
        embedding = outputs.pooler_output.squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        return (embedding / (norm + 1e-6)).tolist()

    @torch.no_grad()
    def _run_florence_task(self, pil_img: Image, task_prompt: str) -> str:
        inputs = self.florence_processor(text=task_prompt, images=pil_img, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=1 # Beam search 1 for speed and VRAM
        )
        results = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return results

    def process_frame(self, frame_np: np.ndarray, video_id: str, clip_id: int, ts_ms: int) -> VisualObservation:
        pil_img = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        
        # 1. Global Embedding
        visual_vec = self._get_siglip_embedding(pil_img)

        # 2. Detailed Captioning
        dense_caption = self._run_florence_task(pil_img, "<DETAILED_CAPTION>")

        # 3. Object Detection
        yolo_results = self.yolo_model(frame_np, verbose=False)[0]
        objects_structured = []
        for box in yolo_results.boxes:
            objects_structured.append({
                "label": yolo_results.names[int(box.cls)],
                "conf": float(box.conf)
            })

        # 4. CPU-based OCR
        ocr_raw = self.ocr_engine.ocr(frame_np, cls=True)
        ocr_structured = []
        if ocr_raw and ocr_raw[0]:
            for line in ocr_raw[0]:
                ocr_structured.append({"text": line[1][0], "conf": float(line[1][1])})

        obs = VisualObservation(
            video_id=video_id,
            clip_id=clip_id,
            ts_ms=ts_ms,
            clip_embedding=visual_vec,
            ocr_tokens=ocr_structured,
            detected_objects=[dense_caption] + [obj["label"] for obj in objects_structured]
        )
        # Spatial metadata for the graph
        obs.__dict__['spatial_metadata'] = {"dense_description": dense_caption}
        return obs