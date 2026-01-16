import cv2
import torch
import base64
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq, SiglipVisionModel, SiglipProcessor
from paddleocr import PaddleOCR
from conclave.core.schemas import VisualObservation

logger = logging.getLogger("Conclave.Vision.Scene")

class SceneProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        A Professional Multi-Stage Vision Pipeline.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 1. SigLIP for Global Visual Search (Superior to CLIP)
        self.siglip_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device, dtype=self.torch_dtype)
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        # 2. Florence-2 for Dense Grounding & Captioning (The "Pro" edge)
        self.florence_model = AutoModelForVision2Seq.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        ).to(self.device, dtype=self.torch_dtype).eval()
        self.florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

        # 3. YOLOv11 for High-Speed Object Detection
        self.yolo_model = YOLO("yolo11x.pt") # Using 'x' for maximum precision

        # 4. PaddleOCR for Precise Text Recognition
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=torch.cuda.is_available())

    @torch.no_grad()
    def _get_siglip_embedding(self, pil_img: Image) -> List[float]:
        inputs = self.siglip_processor(images=pil_img, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
        outputs = self.siglip_model(**inputs)
        # Use pooled output for a compact 1152-dim representation
        embedding = outputs.pooler_output.squeeze().cpu().numpy()
        return (embedding / np.linalg.norm(embedding)).tolist()

    @torch.no_grad()
    def _run_florence_task(self, pil_img: Image, task_prompt: str) -> str:
        inputs = self.florence_processor(text=task_prompt, images=pil_img, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        results = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return results

    def process_frame(self, frame_np: np.ndarray, video_id: str, clip_id: int, ts_ms: int) -> VisualObservation:
        """
        Executes the hierarchical vision pipeline on a single frame.
        """
        pil_img = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        
        # 1. Global Context (SigLIP)
        visual_vec = self._get_siglip_embedding(pil_img)

        # 2. Dense Captioning (Florence-2)
        # We use <DETAILED_CAPTION> to get spatial relationships standard YOLO misses
        dense_caption = self._run_florence_task(pil_img, "<DETAILED_CAPTION>")

        # 3. Precise Object Detection (YOLOv11)
        # We store full metadata: bbox, class, and confidence
        yolo_results = self.yolo_model(frame_np, verbose=False)[0]
        objects_structured = []
        for box in yolo_results.boxes:
            objects_structured.append({
                "label": yolo_results.names[int(box.cls)],
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "conf": float(box.conf)
            })

        # 4. Professional OCR (PaddleOCR)
        ocr_raw = self.ocr_engine.ocr(frame_np, cls=True)
        ocr_structured = []
        if ocr_raw and ocr_raw[0]:
            for line in ocr_raw[0]:
                ocr_structured.append({
                    "text": line[1][0],
                    "conf": float(line[1][1]),
                    "poly": line[0] # The 4-point polygon for the text location
                })

        # Create Observation
        # We inject the Florence-2 caption into 'detected_objects' as a high-level summary
        # while keeping the raw YOLO results in a metadata dict.
        obs = VisualObservation(
            video_id=video_id,
            clip_id=clip_id,
            ts_ms=ts_ms,
            clip_embedding=visual_vec,
            ocr_tokens=ocr_structured,
            detected_objects=[dense_caption] + [obj["label"] for obj in objects_structured]
        )
        
        # We extend the object with a structured metadata dictionary for the Neo4j Graph
        # This allows queries like: "Find clips where a 'laptop' is to the left of a 'coffee cup'"
        obs.__dict__['spatial_metadata'] = {
            "objects": objects_structured,
            "dense_description": dense_caption
        }

        return obs