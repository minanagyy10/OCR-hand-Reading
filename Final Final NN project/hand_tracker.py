import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def get_index_finger_tip(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            # Get index finger tip coordinates (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            frame_h, frame_w, _ = frame.shape
            x, y = int(index_tip.x * frame_w), int(index_tip.y * frame_h)
            
            # Draw hand landmarks for visualization
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            return (x, y), True
        
        return None, False
        
    def get_finger_height(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Get index finger tip (8) and pip (6) landmarks
            tip = hand_landmarks.landmark[8].y
            pip = hand_landmarks.landmark[6].y
            # Return relative height (lower value means finger is raised)
            return tip - pip
        
        return None
    
    def __del__(self):
        self.hands.close()