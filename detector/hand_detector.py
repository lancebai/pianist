import cv2
import mediapipe as mp
import mediapipe.tasks as mp_tasks
from mediapipe.tasks.python import vision
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import platform

# --- Configuration & Constants ---
# MediaPipe Landmark Indices
# Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
# Knuckles (MCP): Thumb: 2, Index: 5, Middle: 9, Ring: 13, Pinky: 17
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 5, 9, 13, 17] # Knuckles (Metacarpophalangeal joints)
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
def get_device_delegate():
    os_name = platform.system()
    
    # 1. macOS (Darwin): Force CPU to avoid the Metal crash
    if os_name == 'Darwin':
        print("🍎 macOS detected: Forcing CPU delegate to avoid known crash.")
        return mp.tasks.BaseOptions.Delegate.CPU
    
    # 2. Linux / Windows: Default to GPU
    else:
        print(f"🐧/🪟 {os_name} detected: Defaulting to GPU delegate.")
        return mp.tasks.BaseOptions.Delegate.GPU

@dataclass
class KeyPressEvent:
    finger_name: str
    start_time: float
    duration: float

class FingerState:
    """
    Tracks the state of a single finger to calculate duration.
    State Machine: IDLE -> PRESSED -> RELEASED -> IDLE
    """
    def __init__(self, name: str):
        self.name = name
        self.is_pressed = False
        self.press_start_time = 0.0
        
    def update(self, pressed_now: bool) -> KeyPressEvent:
        event = None
        
        if pressed_now and not self.is_pressed:
            # Transition: IDLE -> PRESSED
            self.is_pressed = True
            self.press_start_time = time.time()
            # print(f"[{self.name}] Down") # Debug logging
            
        elif not pressed_now and self.is_pressed:
            # Transition: PRESSED -> RELEASED
            self.is_pressed = False
            duration = time.time() - self.press_start_time
            event = KeyPressEvent(self.name, self.press_start_time, duration)
            print(f"[{self.name}] Released. Duration: {duration:.2f}s")
            
        return event

class HandProcessor:
    def __init__(self, model_path='hand_landmarker.task'):
        # Initialize MediaPipe Tasks HandLandmarker with GPU delegate
        try:
            with open(model_path, 'r'): pass
        except FileNotFoundError:
            # Fallback or error if not found. 
            # Assuming widely available task file or user provided path.
            print(f"Warning: Model file '{model_path}' not found at init.")

        base_options = mp_tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=get_device_delegate()
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Initialize state trackers for 5 fingers
        self.finger_states = [FingerState(name) for name in FINGER_NAMES]

    def process(self, frame):
        # MediaPipe Tasks requires mp.Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Timestamp in ms (must be increasing)
        timestamp_ms = int(time.time() * 1000)
        
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        events = []
        
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                # 1. Draw the skeleton
                self._draw_skeleton(frame, hand_lms)
                
                # 2. Analyze geometric logic
                events.extend(self._detect_presses(hand_lms, frame))
                
        return frame, events

    def _draw_skeleton(self, frame, landmarks):
        h, w, _ = frame.shape
        # Draw connections
        for start_idx, end_idx in mp.solutions.hands.HAND_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(frame, 
                     (int(start.x * w), int(start.y * h)), 
                     (int(end.x * w), int(end.y * h)), 
                     (255, 255, 255), 2)
        
        # Draw points
        for lm in landmarks:
             cx, cy = int(lm.x * w), int(lm.y * h)
             cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    def _detect_presses(self, landmarks, frame) -> List[KeyPressEvent]:
        h, w, _ = frame.shape
        events_batch = []

        for i in range(5):
            tip_idx = FINGER_TIPS[i]
            mcp_idx = FINGER_MCP[i] # Knuckle
            
            # Note: landmarks is now a list of NormalizedLandmark objects
            tip_y = landmarks[tip_idx].y
            mcp_y = landmarks[mcp_idx].y
            
            # Simple Heuristic: If Tip Y > Knuckle Y (plus offset), it's curled/pressed.
            is_down = tip_y > (mcp_y + 0.02)
            
            # Update State Machine
            event = self.finger_states[i].update(is_down)
            if event:
                events_batch.append(event)
                
            # Visual Feedback on frame
            color = (0, 255, 0) if is_down else (0, 0, 255)
            # Convert normalized to pixel coords for drawing
            cx, cy = int(landmarks[tip_idx].x * w), int(landmarks[tip_idx].y * h)
            cv2.circle(frame, (cx, cy), 10, color, cv2.FILLED)

        return events_batch
