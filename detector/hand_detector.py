import cv2
import mediapipe as mp
import mediapipe.tasks as mp_tasks
from mediapipe.tasks.python import vision
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import platform
import csv
import os
from datetime import datetime

from abc import ABC, abstractmethod

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

@dataclass
class FingerState:
    """
    Tracks the state of a single finger to calculate duration.
    State Machine: IDLE -> PRESSED -> RELEASED -> IDLE
    """
    name: str
    is_pressed: bool = False
    press_start_time: float = 0.0
        
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

class AbstractFingerDetector(ABC):
    @abstractmethod
    def detect(self, landmarks, frame_shape) -> List[bool]:
        """
        Determines if each of the 5 fingers is pressed.
        Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky].
        """
        pass

class HeuristicFingerDetector(AbstractFingerDetector):
    def detect(self, landmarks, frame_shape) -> List[bool]:
        is_pressed_list = []
        for i in range(5):
            tip_idx = FINGER_TIPS[i]
            mcp_idx = FINGER_MCP[i]
            
            # Note: landmarks is now a list of NormalizedLandmark objects
            tip_y = landmarks[tip_idx].y
            mcp_y = landmarks[mcp_idx].y
            
            # Simple Heuristic: If Tip Y > Knuckle Y (plus offset), it's curled/pressed.
            is_down = tip_y > (mcp_y + 0.02)
            is_pressed_list.append(is_down)
        return is_pressed_list

class TrainedFingerDetector(AbstractFingerDetector):
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        # TODO: Load the trained model (e.g., sklearn pickle, onnx, pytorch)
        if model_path:
            print(f"Loading trained finger detection model from: {model_path}")
            # self.model = load_model(model_path)
        else:
            print("Warning: No model path provided for TrainedFingerDetector.")

    def detect(self, landmarks, frame_shape) -> List[bool]:
        if not self.model_path:
            return [False] * 5

        # TODO: Implement feature extraction
        # features = self._extract_features(landmarks)
        
        # TODO: Run inference
        # predictions = self.model.predict(features)
        
        # Placeholder: Return all False for now
        return [False] * 5

    def _extract_features(self, landmarks):
        # Implement feature extraction logic matching the training phase
        pass

class LandmarkLogger:
    def __init__(self, output_dir="training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename based on timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"landmarks_{timestamp_str}.csv")
        
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write Header
        # timestamp, [x0, y0, z0, ...], [Thumb_tip_y, Thumb_mcp_y, ...], [Thumb_pressed, ...]
        header = ["timestamp_ms"]
        for i in range(21):
            header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
            
        # Add specific features for easy analysis
        for name in FINGER_NAMES:
            header.extend([f"{name}_tip_y", f"{name}_mcp_y"])
            
        for name in FINGER_NAMES:
            header.append(f"{name}_pressed")
            
        self.writer.writerow(header)
        print(f"Logging training data to: {self.filepath}")

    def log(self, timestamp_ms, landmarks, pressed_states):
        row = [timestamp_ms]
        
        # Add all landmarks
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
            
        # Add specific tip and mcp y values
        for i in range(5):
            tip_idx = FINGER_TIPS[i]
            mcp_idx = FINGER_MCP[i]
            row.extend([landmarks[tip_idx].y, landmarks[mcp_idx].y])
            
        # Add pressed/not pressed labels (0 or 1)
        row.extend([1 if p else 0 for p in pressed_states])
        
        self.writer.writerow(row)

    def close(self):
        if self.file:
            self.file.close()

class HandProcessor:
    def __init__(self, model_path='hand_landmarker.task', detector: AbstractFingerDetector = None, log_data: bool = False):
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
        
        # Set detector strategy
        self.detector = detector if detector else HeuristicFingerDetector()
        
        # Initialize Logger
        self.logger = LandmarkLogger() if log_data else None

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
                # Get detection results from the strategy
                pressed_states = self.detector.detect(hand_lms, frame.shape)
                
                # Log data for training
                if self.logger:
                    self.logger.log(timestamp_ms, hand_lms, pressed_states)
                
                # Update state machines and visualize
                h, w, _ = frame.shape
                for i, is_down in enumerate(pressed_states):
                    # Update State Machine
                    event = self.finger_states[i].update(is_down)
                    if event:
                        events.append(event)
                        
                    # Visual Feedback on frame
                    color = (0, 255, 0) if is_down else (0, 0, 255)
                    # Convert normalized to pixel coords for drawing
                    tip_idx = FINGER_TIPS[i]
                    cx, cy = int(hand_lms[tip_idx].x * w), int(hand_lms[tip_idx].y * h)
                    cv2.circle(frame, (cx, cy), 10, color, cv2.FILLED)
                
        return frame, events

    def close(self):
        if self.logger:
            self.logger.close()

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

