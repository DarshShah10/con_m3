import base64
import cv2
import logging
import json
import openai
from typing import List, Dict, Any
from conclave.core.schemas import HierarchicalFrameObservation, DetectedObject

logger = logging.getLogger("Conclave.Vision.Gemini")

class AdvancedSceneProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("gemini", {})
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url")
        self.model = self.config.get("model", "gemini-2.0-flash-lite-preview-02-05")
        
        if not self.api_key:
            raise ValueError("❌ Missing 'gemini.api_key' in config!")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _encode_image(self, image) -> str:
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buffer).decode('utf-8')

    def process_batch(self, frames: List[Any], video_id: str, clip_id: int, start_ms: int, interval_ms: int) -> List[HierarchicalFrameObservation]:
        if not frames: return []

        # Select 3 representative frames (Start, Mid, End)
        indices = [0, len(frames)//2, len(frames)-1]
        selected_frames = [frames[i] for i in indices if i < len(frames)]
        
        # --- ENHANCED PROMPT FOR DETAILS ---
        prompt_text = """
        Analyze these video frames as a Forensic Video Analyst.
        Return a valid JSON object (NOT a list) with these keys:
        
        1. "summary": Detailed narrative of the scene.
        2. "entities": List of specific people/roles. 
           - Identify Roles (e.g., "Police Officer", "Cyclist", "Camera Operator").
           - Identify Famous People if seen (e.g., "Adin Ross", "MrBeast").
        3. "objects": List of key objects with attributes.
           - For Cars: Include Color, Make, and LICENSE PLATE if visible.
           - For Signs: Include the EXACT TEXT (OCR).
        
        JSON Structure Example:
        {
            "summary": "A police officer stops a red car...",
            "entities": [{"label": "Police Officer", "action": "talking"}, {"label": "Adin Ross", "action": "streaming"}],
            "objects": [{"label": "Car", "color": "Red", "text": "Plate: XYZ-123"}, {"label": "Sign", "text": "STOP"}]
        }
        """
        
        content_payload = [{"type": "text", "text": prompt_text}]
        
        for frame in selected_frames:
            b64_img = self._encode_image(frame)
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content_payload}],
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            # Clean Markdown wrappers if present
            clean_json = raw_content.replace("```json", "").replace("```", "").strip()
            
            try:
                data = json.loads(clean_json)
            except:
                # Fallback: sometimes Gemini returns text before JSON
                start = clean_json.find('{')
                end = clean_json.rfind('}') + 1
                data = json.loads(clean_json[start:end])

            # 🔥 FIX: Handle List vs Dict return
            if isinstance(data, list):
                data = data[0]

            mid_ts = start_ms + (len(frames) // 2 * interval_ms)
            
            obs = HierarchicalFrameObservation(
                video_id=video_id,
                clip_id=clip_id,
                ts_ms=mid_ts,
                scene_description=data.get("summary", "Scene analysis available."),
                clip_embedding=[]
            )
            
            # Merge Entities and Objects into the unified schema
            all_detected = data.get("entities", []) + data.get("objects", [])
            
            for item in all_detected:
                label = item.get("label", "Object")
                
                # Append attributes to label for vector search context
                # e.g., "Car (Color: Red, Text: Plate ABC-123)"
                details = []
                if "color" in item: details.append(f"Color: {item['color']}")
                if "text" in item: details.append(f"Text: {item['text']}")
                if "action" in item: details.append(f"Action: {item['action']}")
                
                full_label = f"{label} ({', '.join(details)})" if details else label
                
                obs.objects.append(DetectedObject(
                    label=full_label,
                    confidence=0.95,
                    bbox=[0,0,0,0] # Gemini doesn't give coords in this mode
                ))

            return [obs]

        except Exception as e:
            logger.error(f"Gemini Vision API Failed: {e}")
            return [HierarchicalFrameObservation(video_id=video_id, clip_id=clip_id, ts_ms=start_ms)]