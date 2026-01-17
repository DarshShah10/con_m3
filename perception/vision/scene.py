import cv2
import torch
import numpy as np
import logging
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    SiglipVisionModel, 
    SiglipProcessor
)
from qwen_vl_utils import process_vision_info
import easyocr
from conclave.core.schemas import VisualObservation

logger = logging.getLogger("Conclave.Vision.Scene")

class SceneProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading Scene Models (Qwen2-VL + SigLIP + YOLO) on {self.device}...")

        # 1. SigLIP Base (For Embeddings)
        self.siglip_model = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(self.device, dtype=self.torch_dtype).eval()
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

        # 2. Qwen2-VL-2B (For Detailed Captioning - REPLACES FLORENCE-2)
        # Using 'Qwen/Qwen2-VL-2B-Instruct' - SOTA 2B model
        # device_map="auto" is safe here, or explicit to device
        self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=self.torch_dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            attn_implementation="sdpa" # Use PyTorch 2.0 Native Flash Attention
        ).eval()
        
        self.vlm_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct"
        )

        # 3. YOLO11s (For Object Detection)
        self.yolo_model = YOLO("yolo11s.pt") 

        # 4. EasyOCR (For Text)
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'), verbose=False)

    @torch.no_grad()
    def _get_siglip_embedding_batch(self, pil_imgs: List[Image.Image]) -> List[List[float]]:
        # SigLIP expects RGB images
        inputs = self.siglip_processor(images=pil_imgs, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
        outputs = self.siglip_model(**inputs)
        
        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        norms = torch.linalg.norm(embeddings, ord=2, dim=1, keepdim=True)
        return (embeddings / (norms + 1e-6)).tolist()

    @torch.no_grad()
    def _run_qwen_caption_sequential(self, pil_imgs: List[Image.Image], max_tokens: int = 80) -> List[str]:
        """
        Runs Qwen2-VL on a list of images. Sequential is safer for VRAM 
        with VLMs that handle dynamic resolutions.
        Optimized with shorter prompts and token limits for speed.
        """
        captions = []
        prompt_text = "Describe this scene concisely." # Shorter prompt for speed

        for img in pil_imgs:
            # Qwen2-VL Input Format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            # Prepare inputs
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device) # device placement handled by Qwen internals usually, but explicit safe

            # Generate with token limit for speed
            generated_ids = self.vlm_model.generate(
                **inputs, 
                max_new_tokens=max_tokens  # SPEED LIMIT (default 80)
            )
            
            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            captions.append(output_text)
            
        return captions

    def process_batch(self, frames_np: List[np.ndarray], video_id: str, clip_id: int, start_ts: int, interval_ms: int) -> List[VisualObservation]:
        if not frames_np: return []

        # Convert to PIL
        pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_np]
        
        # 1. Embeddings (SigLIP - Batched)
        visual_vecs = self._get_siglip_embedding_batch(pil_imgs)
        
        # 2. Captions (Qwen2-VL)
        # Limit max_new_tokens to 80 for speed (concise descriptions)
        captions = self._run_qwen_caption_sequential(pil_imgs, max_tokens=80)
        
        # 3. Object Detection (YOLO)
        yolo_results = self.yolo_model(frames_np, verbose=False, stream=False) 
        
        # 4. OCR (EasyOCR)
        ocr_results = []
        for f in frames_np:
            try:
                res = self.ocr_reader.readtext(f) 
                lines = []
                for (bbox, text, conf) in res:
                    if conf > 0.4:  # Higher confidence threshold for quality
                        lines.append({"text": text, "conf": float(conf)})
                ocr_results.append(lines)
            except:
                ocr_results.append([])

        # 5. Aggregate with Object Filtering
        observations = []
        for i, (vec, cap, y_res, ocr) in enumerate(zip(visual_vecs, captions, yolo_results, ocr_results)):
            
            # OBJECT FILTER: Ignore objects < 1% of frame area (removes tiny/noise detections)
            img_area = y_res.orig_shape[0] * y_res.orig_shape[1]
            objects = []
            for box in y_res.boxes:
                # Get width and height from xywh format
                w = box.xywh[0][2].item()
                h = box.xywh[0][3].item()
                obj_area = w * h
                
                # Only keep objects larger than 1% of image
                if (obj_area / img_area) > 0.01:
                    objects.append(y_res.names[int(box.cls)])
            
            ts = start_ts + (i * interval_ms)
            
            obs = VisualObservation(
                video_id=video_id,
                clip_id=clip_id,
                ts_ms=ts,
                clip_embedding=vec,
                ocr_tokens=ocr,
                detected_objects=[cap] + objects  # Caption first, then filtered objects
            )
            obs.__dict__['spatial_metadata'] = {"dense_description": cap}
            observations.append(obs)
            
        return observations