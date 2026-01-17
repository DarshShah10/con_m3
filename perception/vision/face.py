import cv2
import torch
import numpy as np
import base64
import logging
from typing import List, Dict, Any, Union
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from conclave.core.schemas import FaceObservation
# Using standard sklearn clustering instead of hdbscan
from sklearn.cluster import DBSCAN

logger = logging.getLogger("Conclave.Vision.Face")

class FaceProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det_threshold = config.get("face_detection_score_threshold", 0.90)
        self.min_cluster_size = config.get("min_cluster_size", 2)
        
        logger.info(f"⚡ Loading Face Stack (Batch Optimized) on {self.device}...")

        # 1. Detection (MTCNN)
        self.mtcnn = MTCNN(
            keep_all=True, 
            device=self.device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )

        # 2. Recognition (InceptionResnetV1)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # 🚀 OPTIMIZATION: Compile the Resnet for A40
        if hasattr(torch, 'compile'):
            try:
                self.resnet = torch.compile(self.resnet)
                logger.info("✓ Face recognition model compiled")
            except Exception as e:
                logger.warning(f"Torch compile failed (ignoring): {e}")

    def extract_from_frames(self, frames: List[Union[str, np.ndarray]], video_id: str, clip_id: int) -> List[FaceObservation]:
        """
        Batch-Optimized Extraction.
        1. Detects faces in loop (MTCNN is sequential).
        2. Stacks all crops.
        3. Embeds all crops in ONE GPU forward pass.
        """
        raw_observations = []
        batch_crops = []
        batch_meta = []  # Stores (idx, box, prob, b64) to map back after embedding

        # --- Phase 1: Detection & Cropping ---
        for idx, frame_input in enumerate(frames):
            img_bgr = frame_input if isinstance(frame_input, np.ndarray) else None
            if img_bgr is None: 
                continue  # Skip base64 handling for speed, assume producer gives numpy

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            try:
                # MTCNN Detect
                boxes, probs = self.mtcnn.detect(pil_img)
            except Exception as e:
                logger.warning(f"MTCNN detection failed: {e}")
                continue
                
            if boxes is None: 
                continue

            for box, prob in zip(boxes, probs):
                if prob < self.det_threshold: 
                    continue

                box = box.astype(int)
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), box[2], box[3]
                if x2 <= x1 or y2 <= y1: 
                    continue
                
                face_crop = img_rgb[y1:y2, x1:x2]
                if face_crop.size == 0: 
                    continue

                # Prepare for Batching
                face_pil = Image.fromarray(face_crop).resize((160, 160))
                # Standard Norm: (x - 127.5) / 128.0
                tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
                
                # Create Base64 chip for metadata
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                face_b64 = base64.b64encode(buffer).decode('utf-8')

                batch_crops.append(tensor)
                batch_meta.append({
                    "ts": idx * 200,
                    "box": box.tolist(),
                    "prob": float(prob),
                    "b64": face_b64
                })

        if not batch_crops:
            return []

        # --- Phase 2: Batch Recognition (The Speedup) ---
        # Stack all faces: [N, 3, 160, 160]
        face_tensor_batch = torch.stack(batch_crops).to(self.device)
        
        with torch.no_grad():
            # ONE CALL instead of N calls
            embeddings_batch = self.resnet(face_tensor_batch).cpu().numpy()

        # --- Phase 3: Assembly ---
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
        """
        Cluster faces using sklearn's DBSCAN instead of hdbscan.
        Uses cosine distance for face similarity.
        """
        if len(observations) < self.min_cluster_size:
            return observations
        
        try:
            embeddings = np.array([o.embedding for o in observations])
            
            # Simple check for NaN
            if np.isnan(embeddings).any(): 
                return observations
            
            # DBSCAN with Cosine Distance
            # Calculate pairwise cosine similarity, then convert to distance
            similarity = np.dot(embeddings, embeddings.T)
            distances = np.clip(1 - similarity, 0, 1).astype(np.float64)
            
            # eps=0.4 means faces are clustered if cosine similarity > 0.6
            # This is a reasonable threshold for face recognition
            clusterer = DBSCAN(eps=0.4, min_samples=self.min_cluster_size, metric="precomputed")
            labels = clusterer.fit_predict(distances)
            
            for i, obs in enumerate(observations):
                # Only assign if not noise (-1)
                if labels[i] != -1:
                    obs.__dict__['temp_cluster_id'] = int(labels[i])
                    
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            
        return observations