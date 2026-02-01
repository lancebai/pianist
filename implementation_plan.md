# System Reliability & Recovery Plan

## Goal
Prevent the application from crashing or exiting when the camera disconnects (e.g., loose cable, driver crash). Instead, enter a "Reconnecting" state and attempt to restore functionality automatically.

## Proposal

### 1. Refactor `LiveCameraSource`
The current `frames()` method breaks the loop immediately if `cap.read()` fails.

**New Behavior:**
- **State Machine**: Introduce logic to handle `connected`, `connecting` (init), and `reconnecting` states.
- **Generator Loop**: Instead of stopping, the loop will continue indefinitely.
- **On Failure (Init or Runtime)**: 
    - Attempt to release and re-initialize `cv2.VideoCapture`.
    - Implement **Exponential Backoff** (wait 1s, 2s, 4s...) to avoid busy-looping on a dead device.
    - While reconnecting, `yield None` to keep the main application loop alive.
    - **Initialization**: Constructor should not raise exception if camera is missing; instead, it starts in `disconnected` state and the first call to `frames()` enters the retry loop.

### 2. Update `main.py`
The main loop currently expects a valid `frame` every iteration.

**New Behavior:**
- **Handle `None`**: If `frame` is `None` (signal that source is down):
    - Skip inference (`processor.process`).
    - Skip saving (`writer.write`).
    - **UI Feedback**: Create a blank (black) frame or reuse the last valid frame.
    - **Status Overlay**: Draw text "Camera Disconnected - Reconnecting..." on this placeholder frame.
    - Call `cv2.waitKey` to ensure the window remains responsive (doesn't freeze).

## Implementation Details

### `sources/camera.py`
```python
class LiveCameraSource(VideoSource):
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        # Don't open here to avoid crashing on init.
        # Initialize in the loop.

    def frames(self):
        while True:
            # 1. Connect if needed
            if self.cap is None or not self.cap.isOpened():
                self._connect()
                
            # 2. Read frame
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    yield frame_idx, frame
                    continue
            
            # 3. Handle Failure
            yield frame_idx, None 
            time.sleep(1) # Simple backoff for example
```

### `main.py`
```python
for frame_idx, frame in source.frames():
    if frame is None:
        # Create black frame for UI
        display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(display_frame, "Reconnecting...", ...)
        cv2.imshow("Pianist", display_frame)
        cv2.waitKey(1)
        continue
        
    # Standard processing...
```

## Verification
1.  **Manual Test**:
    - Start the app with a webcam.
    - Physically unplug the webcam.
    - **Expectation**: App shows "Reconnecting..." and does not crash.
    - Plug the webcam back in.
    - **Expectation**: App resumes tracking hands automatically.
2.  **Init Test**:
    - Start app without webcam.
    - **Expectation**: App starts, shows "Reconnecting...".
    - Plug in webcam.
    - **Expectation**: Connects and starts.
