from pprint import pprint

import cv2
import mediapipe as mp
import torch
import torch.nn as nn


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


# Define the neural network model
class ThreeLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    # Initialize the model, loss function, and optimizer
    input_size = 132
    hidden_size1 = 50
    hidden_size2 = 25
    output_size = 1
    # Create an instance of your model
    model = ThreeLayerClassifier(input_size, hidden_size1, hidden_size2, output_size)
    # Load the model's state_dict from a .pth file
    model_path = '../training-area/person2.pth'  # replace with the path to your .pth file
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()

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
            predicted_output = model(input_data)
            predicted_class = 1 if predicted_output.item() > 0.5 else 0

            print(f'Predicted output: {predicted_output.item()}, Predicted class: {predicted_class}')
        except Exception as e:
            pass

        frame = single_pose.get_annotated_frame()
        cv2.imshow("webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
