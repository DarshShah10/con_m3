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
        self.min_duration = config.get("min_duration_for_audio", 0.5)
        
        logger.info(f"⚡ Loading Audio Stack on {self.device}...")

        # 1. Silero VAD (Instant load, no auth required)
        # Using torch.hub to get the latest enterprise-grade VAD
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self.vad_model.to(self.device)
            (self.get_speech_timestamps, _, _, _, _) = utils
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise e

        # 2. Speaker Embedding (SpeechBrain ECAPA-TDNN)
        try:
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)}
            )
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain: {e}")
            raise e

        # 3. ASR (Whisper)
        self.asr_model = whisper.load_model("base", device=self.device)

    def _bytes_to_audio_segment(self, audio_input: Union[str, bytes]) -> AudioSegment:
        if isinstance(audio_input, bytes):
            return AudioSegment.from_file(io.BytesIO(audio_input))
        elif isinstance(audio_input, str):
            import base64
            return AudioSegment.from_file(io.BytesIO(base64.b64decode(audio_input)))
        return None

    def process_clip_audio(self, audio_input: Any, video_id: str, clip_id: int) -> List[VoiceObservation]:
        """
        Fast Pipeline: VAD -> Crop -> Batch Embed -> Cluster -> Transcribe
        """
        audio_segment = self._bytes_to_audio_segment(audio_input)
        if not audio_segment: return []

        # 1. Preprocess for VAD (Convert to mono, 16k for Silero)
        # We need a float32 tensor for Silero
        wav_data = np.array(audio_segment.set_frame_rate(16000).set_channels(1).get_array_of_samples())
        wav_float = wav_data.astype(np.float32) / 32768.0
        wav_tensor = torch.from_numpy(wav_float).to(self.device)

        # 2. Get Timestamps (Fast)
        speech_timestamps = self.get_speech_timestamps(
            wav_tensor, 
            self.vad_model, 
            sampling_rate=16000,
            min_speech_duration_ms=int(self.min_duration * 1000)
        )

        if not speech_timestamps:
            return []

        # 3. Extract Segments & Embed
        # We process segments to get embeddings and text
        observations = []
        embeddings = []
        metadata = []

        temp_seg_file = f"temp_seg_{video_id}.wav"

        for ts in speech_timestamps:
            start_ms = int(ts['start'] / 16000 * 1000)
            end_ms = int(ts['end'] / 16000 * 1000)
            
            # Skip micro-segments
            if (end_ms - start_ms) < (self.min_duration * 1000): continue

            # Export for SpeechBrain/Whisper (they prefer file paths or specific formats)
            segment = audio_segment[start_ms:end_ms]
            segment.export(temp_seg_file, format="wav")

            # A. Embed (GPU)
            # SpeechBrain encodes file path directly, handling normalization
            emb_tensor = self.speaker_model.encode_batch(
                self.speaker_model.load_audio(temp_seg_file).unsqueeze(0)
            ).squeeze().cpu()
            
            # B. Transcribe (GPU)
            try:
                transcription = self.asr_model.transcribe(temp_seg_file)['text'].strip()
            except:
                transcription = ""

            if transcription:
                embeddings.append(emb_tensor.numpy())
                metadata.append({
                    "start": start_ms,
                    "end": end_ms,
                    "text": transcription
                })

        if os.path.exists(temp_seg_file): os.remove(temp_seg_file)
        if not embeddings: return []

        # 4. Clustering (Who is who?)
        # If we have enough segments, cluster them. If only 1, it's Speaker 0.
        speaker_labels = [0] * len(embeddings)
        if len(embeddings) > 1:
            try:
                # Cosine distance clustering
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.7, # Cosine distance threshold
                    metric="cosine",
                    linkage="average"
                )
                speaker_labels = clusterer.fit_predict(embeddings)
            except Exception as e:
                logger.warning(f"Clustering failed, assigning all to Speaker 0: {e}")

        # 5. Build Observations
        for i, meta in enumerate(metadata):
            # Normalize embedding for Qdrant
            emb = embeddings[i]
            norm = np.linalg.norm(emb)
            norm_emb = (emb / (norm + 1e-6)).tolist()

            # We create a temporary entity ID based on the cluster
            # The IdentityManager will later resolve this to a global ID
            local_speaker_id = f"local_speaker_{speaker_labels[i]}"

            obs = VoiceObservation(
                video_id=video_id,
                clip_id=clip_id,
                ts_ms=meta["start"],
                embedding=norm_emb,
                asr_text=meta["text"],
                start_sec=meta["start"] / 1000.0,
                end_sec=meta["end"] / 1000.0,
                entity_id=None # Let IdentityManager resolve this
            )
            # Tag the observation so IdentityManager can use the cluster info if needed
            obs.__dict__['temp_cluster_id'] = int(speaker_labels[i])
            observations.append(obs)

        return observations