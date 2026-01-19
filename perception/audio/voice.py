import os
import logging
import json
import re
from typing import List, Dict, Any
from conclave.core.schemas import DialogueLine
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

logger = logging.getLogger("Conclave.Audio")

class GlobalAudioPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.deepgram_key = config.get("deepgram_api_key")
        if not self.deepgram_key:
            raise ValueError("❌ Missing 'deepgram_api_key' in config!")
        
        self.deepgram = DeepgramClient(self.deepgram_key)
        logger.info("🎤 Deepgram API Initialized (Using Nova-2 Latest).")

    def process_full_video(self, audio_path: str, video_id: str) -> Dict[str, Any]:
        logger.info("☁️ Sending audio to Deepgram...")
        try:
            with open(audio_path, "rb") as file:
                buffer_data = file.read()
            
            payload: FileSource = {"buffer": buffer_data}
            
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                diarize=True,
                diarize_version="latest", # 🔥 Force latest diarization engine
                utterances=True,
                punctuate=True,
                filler_words=False # Remove 'um', 'uh' to clean transcript
            )
            
            response = self.deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        except Exception as e:
            logger.error(f"Deepgram API failed: {e}")
            return {"timeline": [], "embeddings": {}}

        logger.info("✅ Deepgram response received. Segmenting...")
        timeline = []
        
        try:
            results = getattr(response, "results", None)
            utterances = getattr(results, "utterances", []) if results else []
            
            for utt in utterances:
                raw_text = getattr(utt, "transcript", "").strip()
                if not raw_text: continue
                
                # Skip tiny blips (< 0.3s) which cause speaker confusion
                start = getattr(utt, "start", 0.0)
                end = getattr(utt, "end", 0.0)
                if (end - start) < 0.3: continue 

                spk_int = getattr(utt, "speaker", 0)
                conf = getattr(utt, "confidence", 0.9)
                speaker_id = f"SPEAKER_{spk_int:02d}"
                
                # Split Logic (Preserved from previous step)
                segments = re.split(r'(?<=[.!?])\s+', raw_text)
                total_len = len(raw_text)
                duration = end - start
                current_char = 0
                
                for seg in segments:
                    if len(seg) < 2: continue
                    
                    seg_len = len(seg)
                    seg_start = start + (current_char / total_len) * duration
                    seg_end = seg_start + (seg_len / total_len) * duration
                    
                    timeline.append(DialogueLine(
                        video_id=video_id,
                        clip_id=int(seg_start // 30),
                        entity_id=speaker_id,
                        text=seg,
                        start_ts=int(seg_start * 1000),
                        end_ts=int(seg_end * 1000),
                        confidence=conf
                    ))
                    current_char += seg_len + 1

        except Exception as e:
            logger.error(f"Error parsing Deepgram response: {e}")

        # Voice Embeddings are NOT needed if Diarization + Logic is strong.
        # We rely on the Reasoning Agent to link "SPEAKER_01" to "Police Officer"
        # based on what they say and the scene context.
        return {"timeline": timeline, "embeddings": {}}