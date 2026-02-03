import cv2
import numpy as np
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
import telemetry

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
    parser.add_argument('--log_data', action='store_true', help='Enable logging of landmarks to CSV for training')
    
    args = parser.parse_args()

    # 1. Setup Resources
    print(f"Initializing Source: {args.input_type} -> {args.input_path}")
    source = get_source(args)
    
    print(f"Initializing Detector with model: {args.model}")
    processor = None
    writer = None
    try:
        processor = HandProcessor(model_path=args.model, log_data=args.log_data)
    except Exception as e:
        print(f"Failed to load detector: {e}")
        print("Did you download 'hand_landmarker.task'? Download it from MediaPipe website.")
        sys.exit(1)

    if not isinstance(source, JpegFileSource):
        writer = AsyncWriter(args.output_dir)
        
    # Start Telemetry Server
    telemetry.start_server(port=5000)
    
    print("Starting Pipeline...")
    print("Controls:")
    print("  'q'     : Quit")
    print("  's'     : Single Step / Pause")
    print("  'space' : Toggle Pause/Play")

    # 2. Main Loop
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    paused = False

    try:
        # source.frames() yields (index, frame)
        for frame_idx, frame in source.frames():
            if frame is None:
                # display_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                # cv2.putText(display_frame, "Reconnecting...", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # cv2.imshow("Pianist", display_frame)
                # cv2.waitKey(1)
                print("source disconnected, attempting to reconnect...")
                continue
            t_start = time.time()
            
            # --- Inference ---
            t_infer_start = time.time()
            processed_frame, events = processor.process(frame)
            t_infer = time.time() - t_infer_start

            # --- Visualization Info ---
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                duration = time.time() - fps_start_time
                fps = fps_counter / duration
                fps_counter = 0
                fps_start_time = time.time()
                print(f"[Profile] FPS: {fps:.2f} | Infer: {t_infer*1000:.1f}ms")
                
                # Update Telemetry
                num_hands = len(events) # events contains key presses, but let's approximate or just say 1 if events exist
                # Actually, main.py doesn't track raw hand count from processor easily without modifying processor return.
                # For now, let's just assume 1 hand if we got a processed frame, or 0.
                # Use a better metric if available. Processor events are key presses.
                # Let's count key presses as "hands" for now or just update FPS.
                telemetry.state.update(fps, 0) # TODO: Pass actual hand count if available


            # --- Output: Save ---
            # Save every frame to debug mediapipe results as requested
            # Use async writer to not block
            t_write_start = time.time()
            if writer is not None:
                writer.write(processed_frame)
            t_write = time.time() - t_write_start

            cv2.putText(processed_frame, f"FPS: {fps:.1f} | Infer: {t_infer*1000:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Output: Display ---
            t_display_start = time.time()
            if not args.no_display:
                cv2.imshow("Pianist Debugger", processed_frame)
                
                # Wait interaction
                # If paused, wait indefinitely (0) until key press
                # If running, wait 1ms
                wait_time = 0 if paused else 1
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # specific request: 's' for single step (implies pause)
                    paused = True
                elif key == ord(' '):
                    # standard space to toggle
                    paused = not paused
            t_display = time.time() - t_display_start
            
            # Print detailed warnings if slow
            total_loop = time.time() - t_start
            if total_loop > 0.1: # If taking > 100ms ( < 10 FPS )
                print(f"[Slow Frame] Total: {total_loop*1000:.1f}ms | Infer: {t_infer*1000:.1f}ms | Write: {t_write*1000:.1f}ms | Display: {t_display*1000:.1f}ms")


    except KeyboardInterrupt:
        print("\nInterrupted manually.")
    finally:
        print("Cleaning up...")
        if not isinstance(source, JpegFileSource) and 'writer' in locals():
            writer.stop()
        if processor:
            processor.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()