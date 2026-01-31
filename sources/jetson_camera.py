import cv2
from .video_source import VideoSource

class JetsonCameraSource(VideoSource):
    def __init__(self, device="/dev/video0", width=1920, height=1080, framerate=30):
        self.gst_pipeline = (
            f"v4l2src device={device} ! "
            f"image/jpeg, format=MJPG, width={width}, height={height}, framerate={framerate}/1 ! "
            "nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )

        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        self._fps = framerate
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open Jetson camera with GStreamer pipeline")

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