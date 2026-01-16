import base64
import io
import os
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Any
from pydub import AudioSegment
import wespeaker
import whisper
from pyannote.audio import Pipeline
from conclave.core.schemas import VoiceObservation

class VoiceProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Precise Voice Processing using WeSpeaker (ERes2Net) and Pyannote.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Speaker Embedding Model (Successor to SpeakerLab)
        # Model: cam++ or eres2net
        self.speaker_model = wespeaker.load_model('english')
        self.speaker_model.set_gpu(0) if torch.cuda.is_available() else None

        # 2. ASR Model (Whisper)
        self.asr_model = whisper.load_model("base", device=self.device)

        # 3. Diarization Pipeline (Pyannote)
        # Requires a HuggingFace token for Pyannote 3.1
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.get("hf_token")
        )
        if torch.cuda.is_available():
            self.diarization_pipeline.to(self.device)

        self.min_duration = config.get("min_duration_for_audio", 0.8)

    def _base64_to_wav(self, base64_audio: str) -> io.BytesIO:
        audio_data = base64.b64decode(base64_audio)
        return io.BytesIO(audio_data)

    def process_clip_audio(self, base64_audio: str, video_id: str, clip_id: int) -> List[VoiceObservation]:
        """
        Full Pipeline: Diarize -> Crop -> Transcribe -> Embed.
        """
        # Convert base64 to temp wav file for processing
        wav_io = self._base64_to_wav(base64_audio)
        audio_segment = AudioSegment.from_file(wav_io)
        
        # Save to temp file because pyannote/wespeaker expect file paths or specific tensors
        temp_wav = f"temp_audio_{clip_id}.wav"
        audio_segment.export(temp_wav, format="wav")

        observations = []

        try:
            # 1. Diarization (Who spoke when)
            diarization = self.diarization_pipeline(temp_wav)

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                if duration < self.min_duration:
                    continue

                # 2. Crop Segment
                start_ms = turn.start * 1000
                end_ms = turn.end * 1000
                segment_audio = audio_segment[start_ms:end_ms]
                
                segment_path = f"temp_seg_{clip_id}.wav"
                segment_audio.export(segment_path, format="wav")

                # 3. Transcribe (Whisper)
                asr_result = self.asr_model.transcribe(segment_path)
                text = asr_result['text'].strip()

                if not text:
                    continue

                # 4. Generate Embedding (WeSpeaker ERes2Net)
                # Resample if necessary (WeSpeaker expects 16k)
                embedding = self.speaker_model.extract_embedding(segment_path)
                
                # Normalize embedding for Cosine Similarity (Vector Store requirement)
                norm = np.linalg.norm(embedding)
                normalized_emb = (embedding / norm).tolist() if norm > 0 else embedding.tolist()

                # 5. Create Schema Observation
                obs = VoiceObservation(
                    video_id=video_id,
                    clip_id=clip_id,
                    ts_ms=int(start_ms),
                    embedding=normalized_emb,
                    asr_text=text,
                    start_sec=float(turn.start),
                    end_sec=float(turn.end)
                )
                observations.append(obs)

        finally:
            # Cleanup temp files
            if os.path.exists(temp_wav): os.remove(temp_wav)
            if os.path.exists(f"temp_seg_{clip_id}.wav"): os.remove(f"temp_seg_{clip_id}.wav")

        return observations