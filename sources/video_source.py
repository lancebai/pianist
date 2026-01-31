from abc import ABC, abstractmethod

class VideoSource(ABC):
    @abstractmethod
    def frames(self):
        """
        Should yield (frame_idx, frame) tuples.
        """
        pass

    @property
    @abstractmethod
    def resolution(self):
        """
        Returns (width, height) of the video.
        """
        pass

    @property
    @abstractmethod
    def fps(self):
        """
        Returns the frame rate (fps) of the video source, or None if unavailable.
        """
        pass