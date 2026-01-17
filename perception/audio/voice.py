import io
import os
import torch
import logging
import numpy as np
from typing import List, Dict, Any, Union
from pydub import AudioSegment
import whisper
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from conclave.core.schemas import VoiceObservation

logger = logging.getLogger("Conclave.Audio.Voice")

class VoiceProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_duration = 0.5
        
        logger.info(f"🎤 Loading SOTA Audio Stack on {self.device}...")

        # 1. VAD (Silero) - Detects WHEN speech happens
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.vad_model.to(self.device)
            self.get_speech_timestamps = utils[0]
        except Exception as e:
            logger.error(f"Failed to load VAD: {e}")
            raise e

        # 2. Speaker Embedding (SpeechBrain) - Detects WHO is speaking
        try:
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)}
            )
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain: {e}")
            raise e

        # 3. ASR (Whisper) - Detects WHAT is said
        # using 'small.en' is faster/better than base for dialogue
        self.asr_model = whisper.load_model("small.en", device=self.device)

    def process_clip_audio(self, audio_bytes: bytes, video_id: str, clip_id: int) -> List[VoiceObservation]:
        if not audio_bytes: return []
        
        # Convert raw bytes to Tensor for VAD
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_data = np.array(audio.set_frame_rate(16000).set_channels(1).get_array_of_samples())
            wav_float = wav_data.astype(np.float32) / 32768.0
            wav_tensor = torch.from_numpy(wav_float).to(self.device)
        except Exception:
            return []

        # 1. Get precise timestamps
        try:
            timestamps = self.get_speech_timestamps(
                wav_tensor, self.vad_model, sampling_rate=16000, min_speech_duration_ms=500
            )
        except: return []
        
        if not timestamps: return []

        observations = []
        
        # 2. Process each speech segment
        # Using a fixed temp file name pattern to avoid accumulation
        temp_path = f"temp_seg_{video_id}_{clip_id}.wav"
        
        for ts in timestamps:
            start_ms = int(ts['start'] / 16000 * 1000)
            end_ms = int(ts['end'] / 16000 * 1000)
            
            # Extract segment audio
            segment = audio[start_ms:end_ms]
            
            # Export to temp file for libraries that need file path
            segment.export(temp_path, format="wav")

            try:
                # A. Get Voice Fingerprint (Who?)
                signal = self.speaker_model.load_audio(temp_path)
                embedding = self.speaker_model.encode_batch(signal.unsqueeze(0)).squeeze().cpu().numpy()
                norm_emb = (embedding / (np.linalg.norm(embedding) + 1e-6)).tolist()

                # B. Get Text (What?)
                # We use strict timestamp decoding for accuracy
                transcription = self.asr_model.transcribe(
                    temp_path, fp16=(self.device.type == "cuda")
                )['text'].strip()

                if transcription:
                    obs = VoiceObservation(
                        video_id=video_id,
                        clip_id=clip_id,
                        ts_ms=start_ms,
                        embedding=norm_emb,
                        asr_text=transcription,
                        start_sec=start_ms/1000.0,
                        end_sec=end_ms/1000.0
                    )
                    observations.append(obs)

            except Exception as e:
                logger.warning(f"Audio processing error: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

        return observations