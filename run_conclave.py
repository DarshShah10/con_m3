import argparse
import sys
from main import ConclaveOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Conclave System Runner")
    parser.add_argument("--video", type=str, required=True, help="Path to raw mp4 file")
    parser.add_argument("--video_id", type=str, required=True, help="Canonical ID for this video (e.g., 'meeting_01')")
    parser.add_argument("--window", type=int, default=30, help="Sliding window size in seconds")
    parser.add_argument("--overlap", type=int, default=5, help="Window overlap in seconds")
    
    args = parser.parse_args()

    # Environment sanity check
    if not os.path.exists("configs/api_config.json"):
        print("[!] Error: configs/api_config.json not found. Create it first.")
        sys.exit(1)

    print(f"--- CONCLAVE STARTING: {args.video_id} ---")
    orchestrator = ConclaveOrchestrator(
        config_path="configs/api_config.json", 
        video_id=args.video_id
    )

    try:
        orchestrator.run_pipeline(
            video_path=args.video, 
            window_size=args.window, 
            overlap=args.overlap
        )
    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import os
    main()