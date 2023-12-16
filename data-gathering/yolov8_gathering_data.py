import time

import cv2
import torch
import pandas as pd
from ultralytics import YOLO


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
    yolo = YoloPoseEstimation("../yolo_model/yolov8n-pose.pt")

    target = 0
    dt = {
        "class": target
    }
    for i in range(1, 17):
        dt[f"x{i}"] = 0
        dt[f"y{i}"] = 0
        dt[f"v{i}"] = 0

    df_data = pd.DataFrame(dt, index=range(1, 1))

    for result in yolo.estimate("../dataset/video/v2_erwin.mkv"):
        # Wait for a key event and get the ASCII code
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            target = 1
            print("fighting key pressed!")
        elif key == ord('s'):
            target = 0
            print("stop key pressed!")
        elif key == ord('q'):
            break

        try:
            # one = [target, ]
            print(result.keypoints.data[0].flatten().tolist())

            # df_data.loc[-1] = one  # adding a row
            # df_data.index = df_data.index + 1  # shifting index
            # df_data = df_data.sort_index()  # sorting by index
        except TypeError as te:
            pass

    #     result = result.plot()
    #
    #     cv2.imshow("webcam", result)
    #
    # df_data.to_csv("yolov8_extracted/erwin.csv", index=False)
