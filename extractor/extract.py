import cv2
import os
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Extract frames from a video as JPEGs.")
parser.add_argument("video_path", help="Path to the input .MOV or video file")
parser.add_argument("--output_dir", default="frames", help="Directory to save extracted frames")
args = parser.parse_args()

# --- Create Output Folder ---
os.makedirs(args.output_dir, exist_ok=True)

# --- Video Processing ---
cap = cv2.VideoCapture(args.video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(filename, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames to {args.output_dir}/")

