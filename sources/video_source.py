from abc import ABC, abstractmethod

class VideoSource(ABC):
    @abstractmethod
    def frames(self):
        """
        Should yield (frame_idx, frame) tuples.
        """
        pass

    @abstractmethod
    def get_resolution(self):
        """
        Returns (width, height) of the video.
        """
        pass

    @abstractmethod
    def get_fps(self):
        """
        Returns the frame rate (fps) of the video source, or None if unavailable.
        """
        pass