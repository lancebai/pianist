import cv2
import argparse
import sys
import os
import time

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.camera import LiveCameraSource
from sources.video_file import VideoFileSource
from sources.jpeg_source import JpegFileSource
from detector.hand_detector import HandProcessor
from extractor.async_writer import AsyncWriter

def get_source(args):
    """Factory to create the appropriate video source."""
    if args.input_type == 'camera':
        return LiveCameraSource(int(args.input_path)) # Using input_path as index
    elif args.input_type == 'video':
        if not os.path.exists(args.input_path):
            print(f"Error: Video file not found: {args.input_path}")
            sys.exit(1)
        return VideoFileSource(args.input_path)
    elif args.input_type == 'jpeg':
        if not os.path.isdir(args.input_path):
            print(f"Error: Directory not found: {args.input_path}")
            sys.exit(1)
        return JpegFileSource(args.input_path)
    else:
        print(f"Unknown input type: {args.input_type}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pianist: Realtime Hand Tracking & Extraction Debugger")
    parser.add_argument('--input_type', choices=['camera', 'video', 'jpeg'], required=True, help='Type of input')
    parser.add_argument('--input_path', required=True, help='Path to file/dir or camera index (e.g., 0)')
    parser.add_argument('--model', default='hand_landmarker.task', help='Path to MediaPipe task model')
    parser.add_argument('--output_dir', default='output_frames', help='Directory to save processed frames (if extract is on)')
    parser.add_argument('--no_display', action='store_true', help='Disable window display (headless mode)')
    
    args = parser.parse_args()

    # 1. Setup Resources
    print(f"Initializing Source: {args.input_type} -> {args.input_path}")
    source = get_source(args)
    
    print(f"Initializing Detector with model: {args.model}")
    try:
        processor = HandProcessor(model_path=args.model)
    except Exception as e:
        print(f"Failed to load detector: {e}")
        print("Did you download 'hand_landmarker.task'? Download it from MediaPipe website.")
        sys.exit(1)

    writer = AsyncWriter(args.output_dir)
    
    print("Starting Pipeline...")
    print("Press 'q' to quit.")

    # 2. Main Loop
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    try:
        # source.frames() yields (index, frame)
        for frame_idx, frame in source.frames():
            
            # --- Inference ---
            start_proc = time.time()
            processed_frame, events = processor.process(frame)
            proc_time = time.time() - start_proc

            # --- Visualization Info ---
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            cv2.putText(processed_frame, f"FPS: {fps:.1f} | Proc: {proc_time*1000:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Output: Display ---
            if not args.no_display:
                cv2.imshow("Pianist Debugger", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # --- Output: Save ---
            # Save every frame to debug mediapipe results as requested
            # Use async writer to not block
            writer.write(processed_frame)

    except KeyboardInterrupt:
        print("\nInterrupted manually.")
    finally:
        print("Cleaning up...")
        writer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()