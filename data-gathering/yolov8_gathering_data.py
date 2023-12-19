import time

import cv2
import torch
import numpy as np
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


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


if __name__ == "__main__":
    yolo = YoloPoseEstimation("../yolo_model/yolov8m-pose_openvino_model/")

    # variable for gathering pure coordinate
    target = 0
    dt = {
        "class": target
    }
    for i in range(1, 18):
        dt[f"x{i}"] = 0
        dt[f"y{i}"] = 0
        dt[f"v{i}"] = 0

    pure = pd.DataFrame(dt, index=range(1, 1))

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

    dt = {
        "class": target
    }
    for i in range(1, 9):
        dt[f"a{i}"] = 0

    angel = pd.DataFrame(dt, index=range(1, 1))

    for result in yolo.estimate("../dataset/lapas ngaseman/CCTV FIGHT MASJID/FIGHT_195_230.mp4"):
        # Wait for a key event and get the ASCII code
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            target = 1
            print("fight key pressed!")
        elif key == ord('s'):
            target = 0
            print("stop key pressed!")
        elif key == ord('q'):
            break

        try:
            one = [target]  # this for pure coordinate
            two = [target]  # this for angel

            # get the data
            xyn = result.keypoints.xyn
            confs = result.keypoints.conf

            print(f"xyn : {xyn}")
            print(f"conf : {confs}")


            # # this gathering pure coordinate data
            # for idx, keypoint in enumerate(xyn):
            #     x = keypoint[0]  # this is x coordinate
            #     one.append(x)
            #     y = keypoint[1]  # this is y coordinate
            #     one.append(y)
            #     conf = confs[idx]
            #     one.append(conf)
            #
            # pure.loc[-1] = one  # adding a row
            # pure.index = pure.index + 1  # shifting index
            # pure = pure.sort_index()  # sorting by index
            #
            # # this is gathering angel data
            # for n in need:
            #     # index
            #     first = n[0]
            #     mid = n[1]
            #     end = n[2]
            #
            #     # get data using the index before
            #     # getting angel from three coordinate
            #     two.append(calculate_angle(xyn[first], xyn[mid], xyn[end]))
            #
            # angel.loc[-1] = two  # adding a row
            # angel.index = angel.index + 1  # shifting index
            # angel = angel.sort_index()  # sorting by index

        except TypeError as te:
            pass

        result = result.plot()

        cv2.imshow("webcam", result)

    # # save the pure coordinate
    # pure.to_csv("yolov8_extracted/erwin.csv", index=False)
    # # save the angel
    # angel.to_csv("yolov8_extracted/erwin-angel.csv", index=False)
