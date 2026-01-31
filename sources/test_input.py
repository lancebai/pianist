import numpy as np
from .video_source import VideoSource
import cv2

from led_blink_detector.detector import LEDColor 

LED_COLOR_BGR = {
    LEDColor.OFF: (30, 30, 30),         # Dark gray
    LEDColor.YELLOW: (0, 255, 255),     # BGR for yellow
    LEDColor.ORANGE: (0, 165, 255),     # BGR for orange
}

class TestInputSource(VideoSource):
    def __init__(self, num_frames=10, frame_size=(100, 100), color_pattern=None,
                 coord_pattern=None, brightness=None):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.color_pattern = color_pattern or [LEDColor.OFF] * num_frames
        self.coord_pattern = coord_pattern
        self.brightness = brightness or 200  # Default brightness if not provided

    def get_mock_brightness(self, color):
        if color == LEDColor.OFF:
            return 30
        elif color == LEDColor.YELLOW:
            return self.brightness
        elif color == LEDColor.ORANGE:
            return self.brightness - 20
        else:
            return 30

    def apply_brightness(self, frame, brightness, box_coords):
        x1, y1, x2, y2 = box_coords
        roi = frame[y1:y2, x1:x2]
        brightness_scaled = brightness / 255.0
        roi = np.clip(roi * brightness_scaled, 0, 255).astype(np.uint8)
        frame[y1:y2, x1:x2] = roi
        return frame

    def frames(self):
        for frame_idx in range(self.num_frames):
            frame = np.full((*self.frame_size, 3), 255, dtype=np.uint8)

            # Get color(s) and box coord(s) for the current frame
            colors = self.color_pattern[frame_idx]
            if not isinstance(colors, (list, set)):
                colors = [colors]

            if self.coord_pattern:
                coords = self.coord_pattern[frame_idx]
            else:
                coords = [(10 + i*15, 10, 20 + i*15, 20) for i in range(len(colors))]

            for color, box_coords in zip(colors, coords):
                x1, y1, x2, y2 = box_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), LED_COLOR_BGR[color], -1)

                brightness = self.get_mock_brightness(color)
                frame = self.apply_brightness(frame, brightness, box_coords)

            yield frame_idx, frame

    @property
    def resolution(self):        
        return 1920, 1080

    @property
    def fps(self):
        return 30.0