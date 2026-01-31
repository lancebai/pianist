import cv2
import threading
import queue
import time
import os

class AsyncWriter:
    """
    Writes frames to disk in a separate thread to avoid blocking the main processing loop.
    """
    def __init__(self, output_dir, quit_event=None):
        self.output_dir = output_dir
        self.queue = queue.Queue()
        self.stopped = False
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def write(self, frame, filename=None):
        """
        Queue a frame for writing.
        """
        if self.stopped:
            return
        
        if filename is None:
            filename = f"frame_{self.frame_count:05d}.jpg"
            self.frame_count += 1
            
        self.queue.put((frame.copy(), filename))

    def _worker(self):
        while not self.stopped or not self.queue.empty():
            try:
                # Wait for a brief moment to allow checking 'stopped'
                item = self.queue.get(timeout=0.1)
                frame, filename = item
                
                path = os.path.join(self.output_dir, filename)
                cv2.imwrite(path, frame)
                
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error writing frame: {e}")

    def stop(self):
        self.stopped = True
        self.worker_thread.join()
        print(f"Writer stopped. Total extracted: {self.frame_count}")
