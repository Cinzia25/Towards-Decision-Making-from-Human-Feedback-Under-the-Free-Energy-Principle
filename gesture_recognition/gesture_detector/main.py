import numpy as np
from typing import List, Tuple

from gesture_detector.hand_gesture_extractor import HandProcessing
from gesture_detector.drawing_functions import DrawingFunctions


class GestureDetector:
    def __init__(self):
        # Initialize hand processing module with a high detection threshold
        self.hand_detector = HandProcessing(threshold_detection=0.9)

        # Initialize drawing utilities for visual feedback
        self.draw = DrawingFunctions()

    def fingers_interpretation(self, fingers_up: List[int]) -> str:
        # Map finger configurations to control commands
        # Each tuple represents the state of the five fingers (1 = up, 0 = down)
        commands = {
            (0, 0, 0, 0, 0): 'F',  # Forward
            (1, 1, 1, 1, 1): 'S',  # Stop
            (1, 0, 0, 0, 0): 'L',  # Left
            (0, 0, 0, 0, 1): 'R',  # Right
        }

        # Return corresponding command, or empty string if not recognized
        return commands.get(tuple(fingers_up), "")

    def gesture_interpretation(self, img: np.ndarray) -> Tuple[str, np.ndarray]:
        # Create a copy of the input image to avoid modifying the original
        frame = img.copy()

        # Detect hands and optionally draw landmarks
        frame = self.hand_detector.find_hands(frame, draw=True)

        # Extract hand keypoints and bounding box
        hand_list, bbox = self.hand_detector.find_position(frame, draw_box=False)

        # Proceed only if a full hand (21 landmarks) is detected
        if len(hand_list) == 21:
            # Determine which fingers are raised
            fingers_up = self.hand_detector.fingers_up(hand_list)

            # Debug print of finger states
            print(fingers_up)

            # Interpret gesture into a control command
            command = self.fingers_interpretation(fingers_up)

            # Draw corresponding action icon on the frame
            frame = self.draw.draw_actions(command, frame)

            return command, frame

        else:
            # Default behavior: return Stop command if no valid hand is detected
            return "S", frame