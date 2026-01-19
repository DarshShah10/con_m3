import os
import logging
from huggingface_hub import snapshot_download
from speechbrain.inference.speaker import EncoderClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Downloader")

def download_all():
    logger.info("⏳ Starting Robust Model Download...")
    
    # 1. Set long timeout for slow connections
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120" # 2 minutes timeout
    
    # 2. Download SpeechBrain (The one failing for you)
    logger.info("⬇️ Downloading SpeechBrain (Speaker Embeddings)...")
    try:
        EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        logger.info("✅ SpeechBrain Downloaded.")
    except Exception as e:
        logger.error(f"❌ SpeechBrain Failed: {e}")

    # 3. Download WhisperX / Pyannote (Requires Token)
    # Note: These are usually downloaded on-the-fly by whisperx, 
    # but setting the timeout env var above will help the main script too.
    
    logger.info("🎉 Download setup complete. Now run main.py")

if __name__ == "__main__":
    download_all()