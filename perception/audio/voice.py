import io
import os
import torch
import logging
from typing import List, Dict, Any, Union
from pydub import AudioSegment
import whisper
from pyannote.audio import Pipeline
# Using SpeechBrain for speaker embeddings (more reliable than wespeaker)
from speechbrain.inference.speaker import EncoderClassifier
from conclave.core.schemas import VoiceObservation

logger = logging.getLogger("Conclave.Audio.Voice")

class VoiceProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading Audio Models on {self.device}...")
        
        # 1. Speaker Embedding (SpeechBrain ECAPA-TDNN)
        try:
            # ECAPA-TDNN is state-of-the-art for speaker verification
            # run_opts forces GPU usage for faster inference
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)}
            )
            logger.info("✓ SpeechBrain speaker model loaded")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain: {e}")
            self.speaker_model = None

        # 2. ASR (Whisper Base)
        try:
            self.asr_model = whisper.load_model("base", device=self.device)
            logger.info("✓ Whisper ASR model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.asr_model = None

        # 3. Diarization (Pyannote)
        token = config.get("hf_token")
        self.diarization_pipeline = None
        
        if token:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token
                )
                if self.diarization_pipeline:
                    self.diarization_pipeline.to(self.device)
                logger.info("✓ Pyannote diarization pipeline loaded")
            except Exception as e:
                logger.error(f"Pyannote init failed: {e}. Check HF token permissions.")
                logger.error("   Make sure you've accepted the terms at:")
                logger.error("   https://huggingface.co/pyannote/speaker-diarization-3.1")
        else:
            logger.warning("No HF token provided. Voice diarization will be skipped.")

        self.min_duration = config.get("min_duration_for_audio", 0.8)

    def _base64_to_wav(self, base64_audio: str) -> io.BytesIO:
        """Convert base64 audio to BytesIO for legacy support"""
        import base64
        audio_data = base64.b64decode(base64_audio)
        return io.BytesIO(audio_data)

    def process_clip_audio(self, audio_input: Union[str, bytes, io.BytesIO], video_id: str, clip_id: int) -> List[VoiceObservation]:
        """
        Process audio clip through diarization, ASR, and speaker embedding.
        
        Args:
            audio_input: Can be bytes (fast path), base64 string (legacy), or BytesIO
            video_id: Unique video identifier
            clip_id: Clip sequence number
            
        Returns:
            List of VoiceObservation objects with embeddings and transcripts
        """
        # Early exit if models not loaded
        if not self.diarization_pipeline or not self.speaker_model or not self.asr_model:
            logger.warning("Audio models not fully loaded, skipping audio processing")
            return []

        # Convert input to BytesIO
        wav_io = None
        if isinstance(audio_input, bytes):
            # Fast path: raw bytes from main.py
            wav_io = io.BytesIO(audio_input)
        elif isinstance(audio_input, str):
            # Legacy path: base64 encoded
            wav_io = self._base64_to_wav(audio_input)
        elif isinstance(audio_input, io.BytesIO):
            wav_io = audio_input
        else:
            logger.warning(f"Unsupported audio input type: {type(audio_input)}")
            return []
            
        temp_wav = f"temp_proc_{video_id}_{clip_id}.wav"
        observations = []

        try:
            # Load and export audio segment
            audio_segment = AudioSegment.from_file(wav_io)
            audio_segment.export(temp_wav, format="wav")

            # 1. Speaker Diarization (Who spoke when?)
            diarization = self.diarization_pipeline(temp_wav)

            # 2. Process each speaker turn
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Skip very short segments
                if (turn.end - turn.start) < self.min_duration: 
                    continue

                # Calculate timestamps
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                
                # Export segment for processing
                seg_path = f"temp_seg_{clip_id}_{start_ms}.wav"
                audio_segment[start_ms:end_ms].export(seg_path, format="wav")

                try:
                    # 3. Transcribe (Whisper)
                    asr = self.asr_model.transcribe(seg_path)
                    text = asr['text'].strip()
                except Exception as e:
                    logger.warning(f"ASR failed for segment: {e}")
                    text = ""

                # Skip empty transcriptions
                if not text: 
                    if os.path.exists(seg_path): 
                        os.remove(seg_path)
                    continue

                # 4. Speaker Embedding (SpeechBrain)
                try:
                    # Load audio with SpeechBrain's loader
                    audio_tensor = self.speaker_model.load_audio(seg_path)
                    
                    # Generate embedding
                    # Returns shape: [1, 1, 192] -> squeeze to [192]
                    embedding = self.speaker_model.encode_batch(
                        audio_tensor.unsqueeze(0)
                    ).squeeze().cpu()
                    
                    # Normalize embedding to unit vector
                    norm = torch.linalg.norm(embedding)
                    norm_emb = (embedding / (norm + 1e-6)).tolist()

                    # Create observation
                    obs = VoiceObservation(
                        video_id=video_id,
                        clip_id=clip_id,
                        ts_ms=start_ms,
                        embedding=norm_emb,
                        asr_text=text,
                        start_sec=turn.start,
                        end_sec=turn.end
                    )
                    observations.append(obs)
                    
                except Exception as e:
                    logger.warning(f"Speaker embedding failed: {e}")
                
                # Cleanup segment file
                if os.path.exists(seg_path): 
                    os.remove(seg_path)

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(temp_wav): 
                os.remove(temp_wav)

        return observations