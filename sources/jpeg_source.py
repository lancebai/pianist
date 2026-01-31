import os
import cv2
from .video_source import VideoSource

class JpegFileSource(VideoSource):
    def __init__(self, directory_path):
        self.image_files = sorted([
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')
        ])

    def frames(self):
        for idx, file_path in enumerate(self.image_files):
            frame = cv2.imread(file_path)
            if frame is None:
                continue  # Skip unreadable files
            yield idx, frame

    @property
    def resolution(self):
        for file_path in self.image_files:
            frame = cv2.imread(file_path)
            if frame is not None:
                height, width = frame.shape[:2]
                return width, height
        raise ValueError("No readable images found to determine resolution.")
    
    @property
    def fps(self):
        return 30.0