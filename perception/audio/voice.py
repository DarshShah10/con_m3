import os
import torch
import logging
import numpy as np
import whisperx
import gc
from typing import List, Dict, Any
from pydub import AudioSegment
from speechbrain.inference.speaker import EncoderClassifier
from conclave.core.schemas import DialogueLine

logger = logging.getLogger("Conclave.Audio")

class GlobalAudioPipeline:
    """
    Simpler, Better, Faster.
    Processes the ENTIRE audio track at once before video processing starts.
    
    Advantages:
    1. No cut-off sentences (Global context).
    2. Consistent Speaker IDs (Diarization runs on full file).
    3. Better Embeddings (We can average vectors per speaker).
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = config.get("hf_token") # Required for Diarization
        
        logger.info(f"🎤 Loading WhisperX Pipeline on {self.device}...")
        self.model = whisperx.load_model("large-v2", self.device, compute_type="float16" if self.device=="cuda" else "int8")
        self.align_model, self.align_meta = whisperx.load_align_model(language_code="en", device=self.device)
        
        # Diarization (Who is speaking?)
        self.diarize_model = None
        if self.hf_token:
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        else:
            logger.warning("⚠️ No HuggingFace Token! Speaker IDs will not be generated.")

        # Embedding (For linking to Identity System)
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def process_full_video(self, audio_path: str, video_id: str) -> Dict[str, Any]:
        """
        Returns:
        1. 'timeline': List of dialogue lines with timestamps.
        2. 'speaker_embeddings': Dict mapping 'SPEAKER_01' -> [Vector]
        """
        logger.info("⏳ Starting Global Audio Analysis (This may take a moment)...")
        
        # 1. Transcribe
        result = self.model.transcribe(audio_path, batch_size=16)
        
        # 2. Align (Fix timestamps)
        result = whisperx.align(result["segments"], self.align_model, self.align_meta, audio_path, self.device, return_char_alignments=False)
        
        # 3. Diarize (Assign Speaker Labels)
        if self.diarize_model:
            diarize_segments = self.diarize_model(audio_path)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 4. Process Results & Generate Embeddings
        timeline = []
        speaker_samples = {} # Store audio segments to average embeddings later
        
        # Load audio for embedding extraction
        full_audio = AudioSegment.from_file(audio_path)

        for seg in result["segments"]:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            speaker = seg.get("speaker", "UNKNOWN")
            
            if len(text) < 2: continue

            # Add to timeline
            timeline.append(DialogueLine(
                video_id=video_id,
                clip_id=int(start // 30), # Approximate clip bucket
                entity_id=speaker,        # Temporary ID (SPEAKER_01), resolved later
                text=text,
                start_ts=int(start * 1000),
                end_ts=int(end * 1000),
                confidence=0.99
            ))

            # Collect sample for this speaker (if we haven't processed them yet)
            # We take the longest segment for the best voice print
            duration = end - start
            if speaker != "UNKNOWN" and duration > 1.0:
                if speaker not in speaker_samples:
                    speaker_samples[speaker] = (start, end, duration)
                elif duration > speaker_samples[speaker][2]:
                    speaker_samples[speaker] = (start, end, duration)

        # 5. Generate Canonical Embeddings per Speaker
        speaker_embeddings = {}
        for spk, (s, e, _) in speaker_samples.items():
            try:
                # Extract snippet
                chunk_path = f"temp_{video_id}_{spk}.wav"
                chunk = full_audio[int(s*1000):int(e*1000)]
                chunk.export(chunk_path, format="wav")
                
                # Embed
                signal = self.speaker_encoder.load_audio(chunk_path)
                emb = self.speaker_encoder.encode_batch(signal.unsqueeze(0)).squeeze().cpu().numpy()
                norm_emb = (emb / (np.linalg.norm(emb) + 1e-6)).tolist()
                
                speaker_embeddings[spk] = norm_emb
                
                if os.path.exists(chunk_path): os.remove(chunk_path)
            except Exception as ex:
                logger.warning(f"Failed to embed {spk}: {ex}")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"✅ Audio Complete. Found {len(speaker_embeddings)} unique speakers.")
        return {"timeline": timeline, "embeddings": speaker_embeddings}