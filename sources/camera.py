import cv2
from .video_source import VideoSource

class LiveCameraSource(VideoSource):
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        try:
            # Initialize webcam (set resolution to 1080p)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        except:
            print("1080p is not supported")
        # poc: fps of cv2.VideoCapture(0) is usually low. use the JetsonCameraSource for the higher fps
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)


    def frames(self):
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
        self.cap.release()

    @property
    def resolution(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    @property
    def fps(self):
        return self._fps