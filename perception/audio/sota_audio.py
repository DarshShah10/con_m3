import torch
import logging
import whisperx
import gc
from typing import List
from conclave.core.schemas import DialogueLine

logger = logging.getLogger("Conclave.Audio.SOTA")

class AdvancedAudioProcessor:
    def __init__(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.hf_token = config.get("hf_token") # Needed for Pyannote Diarization

        logger.info(f"🎤 Loading WhisperX on {self.device}...")
        # 1. Load Transcription Model
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)
        
        # 2. Load Alignment Model (for precise timestamps)
        self.align_model, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)

        # 3. Load Diarization Model (Speaker ID)
        if self.hf_token:
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        else:
            logger.warning("⚠️ No HuggingFace Token! Speaker IDs will not be generated.")
            self.diarize_model = None

    def process_audio(self, audio_path: str, video_id: str) -> List[DialogueLine]:
        """
        Runs the full SOTA pipeline: Transcribe -> Align -> Diarize.
        """
        try:
            # A. Transcribe
            result = self.model.transcribe(audio_path, batch_size=16)
            
            # B. Align (Force sync text to audio waves)
            result = whisperx.align(result["segments"], self.align_model, self.metadata, audio_path, self.device, return_char_alignments=False)
            
            # C. Diarize (Identify Speakers)
            if self.diarize_model:
                diarize_segments = self.diarize_model(audio_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            dialogue_lines = []
            for seg in result["segments"]:
                # Default to "unknown" if diarization failed/skipped
                speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
                
                line = DialogueLine(
                    video_id=video_id,
                    clip_id=int(seg["start"] // 30), # Approximate clip bucket
                    speaker_id=speaker,
                    text=seg["text"].strip(),
                    start_ts=int(seg["start"] * 1000),
                    end_ts=int(seg["end"] * 1000),
                    confidence=0.99
                )
                dialogue_lines.append(line)

            # Cleanup VRAM
            gc.collect()
            torch.cuda.empty_cache()
            
            return dialogue_lines

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return []
