import cv2
from gesture_detector.main import GestureDetector
from control_interface.tcp_communication import TCPCommunication

# The system assumes the use of the right hand 

class CarGestureControl:
    def __init__(self):
        # Initialize webcam (device 0 = default camera)
        self.cap = cv2.VideoCapture(0)

        # Set resolution 
        self.cap.set(3, 1280)  # width
        self.cap.set(4, 720)   # height

        # Object responsible for gesture detection and interpretation
        self.hand_gesture = GestureDetector()

        # TCP interface used to send commands to the robot
        self.communication = TCPCommunication()

    def frame_processing(self):
        # Infinite loop for frame acquisition and processing
        while True:
            # Wait 5 ms and capture keyboard input
            # Returns ASCII code of pressed key
            t = cv2.waitKey(5)

            # Capture frame from webcam
            ret, frame = self.cap.read()

            # Process frame to extract gesture and corresponding command
            # command: control signal to be sent
            # draw_frame: frame with visual overlays (e.g., landmarks)
            command, draw_frame = self.hand_gesture.gesture_interpretation(frame)

            # Send command over TCP
            self.communication.sending_data(command)

            # Display processed frame
            cv2.imshow('Car gesture control', draw_frame)

            # Exit condition: ESC key (ASCII 27)
            if t == 27:
                break

        # Proper resource cleanup
        self.communication.close()  # close TCP socket
        self.cap.release()          # release camera
        cv2.destroyAllWindows()     # close OpenCV windows


# Instantiate controller
detector = CarGestureControl()

# Start processing loop
detector.frame_processing()