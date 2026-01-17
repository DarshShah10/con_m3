import cv2
import torch
import numpy as np
import base64
import logging
from typing import List, Dict, Any, Union
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from conclave.core.schemas import FaceObservation
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

logger = logging.getLogger("Conclave.Vision.Face")

class FaceProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quality_threshold = config.get("face_quality_score_threshold", 22.0)
        self.min_cluster_size = config.get("min_cluster_size", 3)
        
        # 1. Detection: YOLOv8-Face (SOTA Speed)
        # Downloads a specific face-detection fine-tune of YOLOv8n
        logger.info("⚡ Loading YOLOv8-Face (High Speed)...")
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.detector = YOLO(model_path) # Auto-loads to GPU if available

        # 2. Recognition: InceptionResnetV1
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        if hasattr(torch, 'compile'):
            try:
                self.resnet = torch.compile(self.resnet)
            except: pass

    def extract_from_frames(self, frames: List[Union[str, np.ndarray]], video_id: str, clip_id: int) -> List[FaceObservation]:
        raw_observations = []
        batch_crops = []
        batch_meta = [] 

        # Configurable "Meme Filter"
        # Ignore faces smaller than 1.5% of the image area
        MIN_FACE_AREA_RATIO = 0.015 

        for idx, frame_input in enumerate(frames):
            # Process every frame provided (Stride handled in main.py)
            img_bgr = frame_input if isinstance(frame_input, np.ndarray) else None
            if img_bgr is None: continue 

            height, width = img_bgr.shape[:2]
            img_area = height * width

            # YOLO Detect (Verbose=False for speed)
            results = self.detector(img_bgr, verbose=False, device=self.device)

            for r in results:
                for box in r.boxes:
                    # Filter by confidence
                    if float(box.conf) < 0.6: continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # --- MEME FILTER ---
                    # If face is too small (e.g. tiny meme in corner), skip it
                    face_area = (x2 - x1) * (y2 - y1)
                    if (face_area / img_area) < MIN_FACE_AREA_RATIO:
                        continue

                    # Padding logic (Context helps recognition)
                    pad_x = int((x2 - x1) * 0.1)
                    pad_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(width, x2 + pad_x)
                    y2 = min(height, y2 + pad_y)

                    face_crop = img_bgr[y1:y2, x1:x2]
                    if face_crop.size == 0: continue

                    # Prepare for embedding (BGR -> RGB -> PIL -> Tensor)
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb).resize((160, 160))
                    
                    tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
                    
                    # Low-res thumbnail for storage
                    _, buffer = cv2.imencode('.jpg', face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    face_b64 = base64.b64encode(buffer).decode('utf-8')

                    batch_crops.append(tensor)
                    batch_meta.append({
                        "ts": idx * 200, # Approximation
                        "box": [x1, y1, x2, y2],
                        "prob": float(box.conf),
                        "b64": face_b64
                    })

        if not batch_crops:
            return []

        # Batch Inference on GPU
        face_tensor_batch = torch.stack(batch_crops).to(self.device)
        
        with torch.no_grad():
            embeddings_batch = self.resnet(face_tensor_batch).cpu().numpy()

        for i, meta in enumerate(batch_meta):
            emb = embeddings_batch[i]
            norm = np.linalg.norm(emb)
            final_emb = (emb / (norm + 1e-6)).tolist()
            
            obs = FaceObservation(
                video_id=video_id,
                clip_id=clip_id,
                ts_ms=meta["ts"],
                embedding=final_emb,
                bbox=meta["box"],
                base64_img=meta["b64"],
                detection_score=meta["prob"],
                quality_score=meta["prob"] * 100
            )
            raw_observations.append(obs)

        return raw_observations

    def cluster_clip_faces(self, observations: List[FaceObservation]) -> List[FaceObservation]:
        # Same clustering logic as before
        if len(observations) < self.min_cluster_size: return observations
        try:
            embeddings = np.array([o.embedding for o in observations])
            if np.isnan(embeddings).any(): return observations
            similarity = np.dot(embeddings, embeddings.T)
            distances = np.clip(1 - similarity, 0, 1).astype(np.float64)
            clusterer = DBSCAN(eps=0.4, min_samples=self.min_cluster_size, metric="precomputed")
            labels = clusterer.fit_predict(distances)
            for i, obs in enumerate(observations):
                if labels[i] != -1:
                    obs.__dict__['temp_cluster_id'] = int(labels[i])
        except: pass
        return observations