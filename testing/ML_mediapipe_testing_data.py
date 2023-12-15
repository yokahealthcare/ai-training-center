from pprint import pprint

import cv2
import joblib
import mediapipe as mp
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class PersonDetection:
    def __init__(self, yolo_model):
        self.model = yolo_model
        # We are only intrested in detecting person
        self.model.classes = [0]

        self.frame = None
        self.result = None

    def set_frame(self, frame):
        self.frame = frame

    def detect(self):
        # Inference
        self.result = self.model(self.frame)

    def get_coordinate(self):
        return self.result.xyxy[0]

    def debug(self):
        print("\nDEBUG INFORMATION - Person Detection")
        print("FRAME")
        pprint(self.frame)
        print("RESULT")
        pprint(self.result)


class SinglePoseEstimation:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.frame = None
        self.result = None
        self.landmarks = None

    def set_frame(self, frame):
        self.frame = frame

    def estimate(self):
        # Convert BGR to RGB
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process image
        self.result = self.pose.process(image)

        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        # Extract feature
        try:
            self.landmarks = self.result.pose_landmarks.landmark
        except:
            return None

    def set_frame_and_estimate(self, frame):
        self.set_frame(frame)
        self.estimate()
        return self.get_annotated_frame()

    def get_annotated_frame(self):
        # Render detection
        self.mp_drawing.draw_landmarks(self.frame, self.result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        return self.frame


if __name__ == "__main__":
    single_pose = SinglePoseEstimation()

    cap = cv2.VideoCapture(0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        single_pose.set_frame(frame)
        single_pose.estimate()

        result = single_pose.result

        # machine learning
        try:
            input_data = torch.Tensor([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten()

            # Load the model
            loaded_model = joblib.load("../training-area/ML/person1.joblib")

            # Make predictions on test data
            y_pred = loaded_model.predict(input_data)

            print(f"Prediction : {y_pred}")
        except TypeError as te:
            pass
        except Exception as e:
            print(f"Exception : {e}")

        frame = single_pose.get_annotated_frame()
        cv2.imshow("webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
