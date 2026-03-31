from pathlib import Path
import cv2
import numpy as np


class DrawingFunctions:
    def __init__(self): 
        # Build path to resources/images relative to this file
        base_path = Path(__file__).parent / "resources" / "images"

        # Load images representing actions
        # cv2.imread returns None if the file is not found or cannot be read
        self.img_forward = cv2.imread(str(base_path / "forward.png"))
        self.img_left = cv2.imread(str(base_path / "left.png"))
        self.img_right = cv2.imread(str(base_path / "right.png"))
        self.img_stop = cv2.imread(str(base_path / "stop.png"))

    def draw_image(self, original_frame: np.ndarray, action_image: np.ndarray):
        # Debug print (not suitable for production)
        print("action_image:", action_image)

        # Check if the image was loaded correctly
        if action_image is None:
            raise FileNotFoundError(
                "The action image was not loaded correctly. Check the file path."
            )

        # Extract dimensions of the action image
        al, an, c = action_image.shape  

        # Extract dimensions of the original frame
        frame_height, frame_width, _ = original_frame.shape

        # Define target size for overlay
        target_height = 120
        target_width = an  # keep original width

        # Resize if height does not match target
        if al != target_height:
            action_image = cv2.resize(action_image, (target_width, target_height))

        # Overlay the action image onto the frame at fixed position
        # (y: 600 → 600 + height, x: 50 → 50 + width)
        original_frame[600:600 + target_height, 50:50 + target_width] = action_image
        
        return original_frame

    def draw_actions(self, action: str, original_frame: np.ndarray) -> np.ndarray:

        # Map action codes to corresponding images
        actions_dict = {
            'F': self.img_forward,
            'S': self.img_stop,
            'L': self.img_left,
            'R': self.img_right,
        }

        # If action is valid, draw corresponding image
        if action in actions_dict:
            movement_image = actions_dict[action]
            original_frame = self.draw_image(original_frame, movement_image)

        return original_frame