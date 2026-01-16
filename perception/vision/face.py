import cv2
import numpy as np
import base64
import logging
import torch
import hdbscan
from typing import List, Dict, Any
from insightface.app import FaceAnalysis
from conclave.core.schemas import FaceObservation

logger = logging.getLogger("Conclave.Vision.Face")

class FaceProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Replicates the 'buffalo_l' setup from M3-Agent with precision thresholds.
        """
        self.det_threshold = config.get("face_detection_score_threshold", 0.85)
        self.quality_threshold = config.get("face_quality_score_threshold", 22)
        self.min_cluster_size = config.get("min_cluster_size", 2)
        
        # Initialize InsightFace (Buffalo_L)
        # Providers prioritize CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def _get_quality_score(self, embedding: np.ndarray) -> float:
        """Precision quality metric from M3-Agent (L2 Norm of embedding)."""
        return float(np.linalg.norm(embedding, ord=2))

    def _classify_face_type(self, bbox: List[int]) -> str:
        """Classifies face as 'ortho' or 'side' based on aspect ratio."""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = height / (width + 1e-6)
        return "ortho" if 1.0 < aspect_ratio < 1.5 else "side"

    def extract_from_frames(self, frames_b64: List[str], video_id: str, clip_id: int) -> List[FaceObservation]:
        """
        Step 1: Extraction
        Decodes frames and extracts all faces exceeding the detection threshold.
        """
        raw_observations = []

        for idx, img_b64 in enumerate(frames_b64):
            # Decode base64 to OpenCV
            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if img is None: continue

            # InsightFace Detection
            faces = self.face_app.get(img)
            
            for face in faces:
                if face.det_score < self.det_threshold:
                    continue

                bbox = face.bbox.astype(int).tolist()
                q_score = self._get_quality_score(face.embedding)
                
                # Filter by Quality (replicating M3 precision)
                if q_score < self.quality_threshold:
                    continue

                # Crop face for the observation metadata (base64)
                face_img = img[max(0, bbox[1]):bbox[3], max(0, bbox[0]):bbox[2]]
                _, buffer = cv2.imencode('.jpg', face_img)
                face_b64 = base64.b64encode(buffer).decode('utf-8')

                obs = FaceObservation(
                    video_id=video_id,
                    clip_id=clip_id,
                    ts_ms=idx * 200, # Assuming 5fps (1 frame every 200ms)
                    embedding=face.normed_embedding.tolist(),
                    bbox=bbox,
                    base64_img=face_b64,
                    detection_score=float(face.det_score),
                    quality_score=q_score
                )
                raw_observations.append(obs)

        return raw_observations

    def cluster_clip_faces(self, observations: List[FaceObservation]) -> List[FaceObservation]:
        """
        Step 2: Local Clustering (HDBSCAN)
        Groups faces within a single clip to ensure temporal consistency before 
        handing off to the Global Identity Manager.
        """
        if len(observations) < self.min_cluster_size:
            return observations

        embeddings = np.array([o.embedding for o in observations])
        
        # Calculate Cosine Distance (1 - Cosine Similarity)
        # Using precomputed metric for HDBSCAN
        similarity = np.dot(embeddings, embeddings.T)
        distances = np.clip(1 - similarity, 0, 1).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, 
            metric="precomputed"
        )
        labels = clusterer.fit_predict(distances)

        # We tag the observations with their cluster IDs
        # This helps the IdentityManager resolve them as a group
        for i, obs in enumerate(observations):
            # We use a temporary field in the object for the grouping logic
            obs.__dict__['temp_cluster_id'] = int(labels[i])

        return observations