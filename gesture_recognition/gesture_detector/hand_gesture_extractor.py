import math
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple


class HandProcessing:
    def __init__(
        self,
        mode=False,
        hands=1,
        model_complexity=0,
        threshold_detection=0.5,
        threshold_tracking=0.5
    ):
        # Configuration parameters for MediaPipe Hands
        self.mode = mode
        self.max_hands = hands
        self.complexity = model_complexity
        self.conf_deteccion = threshold_detection
        self.conf_tracking = threshold_tracking

        # Initialize MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_hands,
            self.complexity,
            self.conf_deteccion,
            self.conf_tracking
        )

        # Utility for drawing landmarks
        self.draw = mp.solutions.drawing_utils

        # Landmark indices corresponding to fingertips
        self.tip = [4, 8, 12, 16, 20]

    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        # Convert image from BGR (OpenCV) to RGB (MediaPipe)
        img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame to detect and track hands
        self.results = self.hands.process(img_color)

        # Draw landmarks and connections if hands are detected
        if self.results.multi_hand_landmarks:
            for mano in self.results.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(
                        frame,
                        mano,
                        self.mp_hands.HAND_CONNECTIONS
                    )

        return frame

    def find_position(
        self,
        frame: np.ndarray,
        hand: int = 0,
        draw_points: bool = True,
        draw_box: bool = True,
        color: List[int] = []
    ) -> Tuple[List[List[int]], Tuple[int, int, int, int]]:
        # Lists to store x and y coordinates of landmarks
        xlist: List[int] = []
        ylist: List[int] = []

        # Bounding box of the detected hand
        bbox: Tuple[int, int, int, int] = ()

        # List of keypoints in format [id, x, y]
        hands_list: List[List[int]] = []

        # Check if any hand is detected
        if self.results.multi_hand_landmarks:
            # Select the specified hand
            my_hand = self.results.multi_hand_landmarks[hand]

            # Iterate over all landmarks
            for id, lm in enumerate(my_hand.landmark):
                alto, ancho, c = frame.shape

                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * ancho), int(lm.y * alto)

                xlist.append(cx)
                ylist.append(cy)
                hands_list.append([id, cx, cy])

                # Optionally draw each landmark
                if draw_points:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 0), cv2.FILLED)

            # Compute bounding box around the hand
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax

            # Optionally draw bounding box
            if draw_box:
                cv2.rectangle(
                    frame,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    color,
                    2
                )

        # Return keypoints list and bounding box
        return hands_list, bbox

    def fingers_up(self, keypoints_list: List[List[int]]) -> List[int]:
        # Determine which fingers are raised (1 = up, 0 = down)
        fingers: List[int] = []

        # Thumb (based on x-coordinate comparison)
        if keypoints_list[self.tip[0]][1] > keypoints_list[self.tip[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers (based on y-coordinate comparison)
        for i in range(1, 5):
            if keypoints_list[self.tip[i]][2] < keypoints_list[self.tip[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def distance(
        self,
        p1: int,
        p2: int,
        frame: np.ndarray,
        draw: bool = True,
        radio: int = 15,
        thickness: int = 3
    ) -> Tuple[float, np.ndarray, list]:
        # Get coordinates of the selected keypoints
        x1, y1 = self.list[p1][1:]
        x2, y2 = self.list[p2][1:]

        # Compute midpoint
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Optionally draw line and points
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
            cv2.circle(frame, (x1, y1), radio, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), radio, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), radio, (0, 0, 255), cv2.FILLED)

        # Compute Euclidean distance between the two points
        length = math.hypot(x2 - x1, y2 - y1)

        # Return distance, updated frame, and additional geometric info
        return length, frame, [x1, y1, x2, y2, cx, cy]