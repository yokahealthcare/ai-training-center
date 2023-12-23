import cv2
import numpy as np
import joblib
from ultralytics import YOLO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class YoloPoseEstimation:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None

    def estimate(self, input):
        self.result = self.model(input, stream=True)
        return self.result

    def info(self):
        # Process results list
        for res in self.result:
            boxes = res.boxes  # Boxes object for bbox outputs
            masks = res.masks  # Masks object for segmentation masks outputs
            keypoints = res.keypoints  # Keypoints object for pose outputs
            probs = res.probs  # Probs object for classification outputs

            print(f"Boxes : {boxes}")
            print(f"Masks : {masks}")
            print(f"Keypoints : {keypoints}")
            print(f"Probs : {probs}")


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Define the neural network model
class ThreeLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size1, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    yolo = YoloPoseEstimation("yolo_model/yolov8n-pose_openvino_model")
    # loaded_model = joblib.load("training-area/MODEL/angel/lapas_ngaseman_logistic_regression.pkl")

    # Initialize the model, loss function, and optimizer
    input_size = 16
    hidden_size1 = 8
    output_size = 1
    model = ThreeLayerClassifier(input_size, hidden_size1, output_size)
    loaded_model = torch.load("training-area/MODEL/angel/lapas_ngaseman.pth")
    model.load_state_dict(loaded_model)
    model.eval()

    # variable for gathering angel
    # index keypoints number
    need = [
        [8, 6, 2],
        [11, 5, 7],
        [6, 8, 10],
        [5, 7, 9],
        [6, 12, 14],
        [5, 11, 13],
        [12, 14, 16],
        [11, 13, 15]
    ]

    THRESHOLD = 0.7
    for result in yolo.estimate("dataset/lapas ngaseman/CCTV FIGHT/NO_FIGHT_1010_1095.mp4"):
        # Wait for a key event and get the ASCII code
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        res = result.plot()
        try:
            boxes = result.boxes.xyxy.tolist()
            xyn = result.keypoints.xyn.tolist()
            confs = result.keypoints.conf
            if confs is None:
                confs = []
            else:
                confs = confs.tolist()

            temp_pred = []
            # Using a for loop with zip
            for conf_row, xyn_row, box in zip(confs, xyn, boxes):
                two = []  # this for angel
                # this is gathering angel data
                for n in need:
                    # index
                    first = n[0]
                    mid = n[1]
                    end = n[2]

                    # get data using the index before
                    # getting angel from three coordinate
                    two.append(calculate_angle(xyn_row[first], xyn_row[mid], xyn_row[end]))
                    two.append(torch.mean(torch.Tensor([conf_row[first], conf_row[mid], conf_row[end]])).item())

                # do prediction
                TEXT = "?"
                # pred = loaded_model.predict_proba(np.array(two).reshape(1, -1))
                pred = model(torch.Tensor(two))
                if pred.item() > THRESHOLD:
                    temp_pred.append(1)
                    TEXT = "FIGHT"
                else:
                    temp_pred.append(0)
                    TEXT = "NO FIGHT"

                # give some text
                # annotate the frame with text - for easier data capturing
                # Choose the font type and scale
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5

                # Choose the font color and thickness
                font_color = (255, 255, 255)  # White color in BGR
                font_thickness = 2

                # Choose the position to put the text
                text_position = (int(box[2]), int(box[3]))
                # Add text to the image
                cv2.putText(res, f"{TEXT}", text_position, font, font_scale, font_color, font_thickness)

            # print(f"PREDICTION : {temp_pred}")
            cv2.imshow("webcam", res)
        except TypeError as te:
            pass
