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


if __name__ == "__main__":
    yolo = YoloPoseEstimation("../yolo_model/yolov8n-pose_openvino_model")
    for result in yolo.estimate(0):
        # Wait for a key event and get the ASCII code
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        print(result.keypoints)
        res = result.plot()
        cv2.imshow("webcam", res)

    cv2.destroyAllWindows()
