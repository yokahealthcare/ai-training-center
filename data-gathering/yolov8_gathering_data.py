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
    yolo = YoloPoseEstimation("../yolo_model/yolov8n-pose_openvino_model/")
    FILENAME = "NO_FIGHT_1190_1275"

    # variable for gathering pure coordinate
    target = -1
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

    for result in yolo.estimate(f"../dataset/lapas ngaseman/CCTV FIGHT/{FILENAME}.mp4"):
        # Wait for a key event and get the ASCII code
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            target = 1
            print("fight key pressed!")
        elif key == ord('s'):
            target = 0
            print("no fight key pressed!")
        elif key == ord('p'):
            target = -1
            print("pause key pressed!")
        elif key == ord('q'):
            break

        try:
            # check if the pause key not pressed
            # pause mean not gathering any data
            if target != -1:
                # get the data
                xyn = result.keypoints.xyn.tolist()
                confs = result.keypoints.conf

                if confs is None:
                    confs = []
                else:
                    confs = confs.tolist()

                # Using a for loop with zip
                for conf_row, xyn_row in zip(confs, xyn):
                    one = [target]  # this for pure coordinate
                    two = [target]  # this for angel
                    # this gathering pure coordinate data
                    for idx, keypoint in enumerate(xyn_row):
                        x = keypoint[0]  # this is x coordinate
                        one.append(x)
                        y = keypoint[1]  # this is y coordinate
                        one.append(y)
                        conf = conf_row[idx]
                        one.append(conf)

                    pure.loc[-1] = one  # adding a row
                    pure.index = pure.index + 1  # shifting index
                    pure = pure.sort_index()  # sorting by index

                    # this is gathering angel data
                    for n in need:
                        # index
                        first = n[0]
                        mid = n[1]
                        end = n[2]

                        # get data using the index before
                        # getting angel from three coordinate
                        two.append(calculate_angle(xyn_row[first], xyn_row[mid], xyn_row[end]))

                    angel.loc[-1] = two  # adding a row
                    angel.index = angel.index + 1  # shifting index
                    angel = angel.sort_index()  # sorting by index

        except TypeError as te:
            pass

        frame = result.plot()

        # annotate the frame with text - for easier data capturing
        # Choose the font type and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0

        # Choose the font color and thickness
        font_color = (255, 255, 255)  # White color in BGR
        font_thickness = 2

        # Choose the position to put the text
        text_position = (25, 25)
        # Add text to the image
        cv2.putText(frame, f"TARGET: {target}", text_position, font, font_scale, font_color, font_thickness)

        cv2.imshow("webcam", frame)

    filename = f"lapas_ngaseman_{FILENAME}.csv"
    # save the pure coordinate
    pure.to_csv(f"yolov8_extracted/{filename}", index=False)
    # save the angel
    angel.to_csv(f"yolov8_extracted_angel/{filename}", index=False)
