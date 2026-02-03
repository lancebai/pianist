import cv2
from .video_source import VideoSource
import time

class LiveCameraSource(VideoSource):
    def __init__(self, camera_index=0):
        self.cap = None
        self.camera_idx = camera_index
        
        self._fps = None

    def _connect(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_idx)
            # Force MJPG and 1080p again
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)

            if self.cap.isOpened():
                print(f"Successfully connected to camera {self.camera_idx}")
            else:
                print(f"Failed to open camera {self.camera_idx}")
                self.cap = None
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            self.cap = None
            
        if self.cap:
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)

    def frames(self):
        frame_idx = 0
        while True:

            if self.cap is None or not self.cap.isOpened():
                print("Camera disconnected, attempting to reconnect...")
                self._connect()            

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame_idx += 1
                    yield frame_idx, frame
                    continue
                else:
                    print("Read failed (ret=False). Releasing camera.")
                    self.cap.release()
                    self.cap = None
                    yield frame_idx, None
                    time.sleep(1)
        
        if self.cap:
            self.cap.release()

    @property
    def resolution(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    @property
    def fps(self):
        return self._fps