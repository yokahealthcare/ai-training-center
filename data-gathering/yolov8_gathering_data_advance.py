import os
import sys
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
        self.result = self.model(input)
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
    yolo = YoloPoseEstimation("../yolo_model/yolov8x-pose.engine")

    # Path to the directory that stored all frame of the video
    directory_path = "../dataset/lapas ngaseman/CCTV FIGHT/FIGHT_595_640"
    FILENAME = directory_path.split("/")[-1]
    # Open the text file for reading
    file_path = f'{directory_path}/annotation_time'

    # Variable for gathering pure coordinate
    dt = {
        "class": 0
    }
    for i in range(1, 18):
        dt[f"x{i}"] = 0
        dt[f"y{i}"] = 0
        dt[f"v{i}"] = 0

    pure = pd.DataFrame(dt, index=range(1, 1))

    # variable for gathering angel
    # index key points number
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
        "class": 0
    }
    for i in range(1, 9):
        dt[f"a{i}"] = 0
        dt[f"v{i}"] = 0

    angel = pd.DataFrame(dt, index=range(1, 1))

    with open(file_path, 'r') as file:
        # Loop through each line in the file
        for line in file:
            # Split the values using commas
            values = line.strip().split(',')

            # Convert the values to integers or other data types as needed
            class_label = int(values[0])
            start_value = int(values[1])
            end_value = int(values[2])

            # Your processing code for each row goes here
            print("Starting new annotation process for...")
            print(f"Class: {class_label}, Start Frame: {start_value}, End Frame: {end_value}")

            target = None
            for num in range(start_value, end_value + 1):
                # Annotation process for each frame
                frame_path = f"{directory_path}/frame{str(num).zfill(4)}.jpg"
                result = yolo.estimate(frame_path)
                cv2.imshow("Webcam", result[0].plot())

                # Waiting for user to decide which one KEYPOINTS to write
                if target != 3:
                    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
                    if key == ord('a'):
                        print("LEFT skeletal keypoints saved")
                        target = 0
                    elif key == ord('d'):
                        print("RIGHT skeletal keypoints saved")
                        target = 1
                    elif key == ord('b'):
                        print("BOTH skeletal keypoints saved")
                        target = 2
                    elif key == ord('v'):
                        print("FAST ANNOTATE skeletal keypoints saved")
                        target = 3
                    elif key == ord('s'):
                        print("SKIPPED!")
                        continue
                    elif key == ord('q'):
                        print("Exited! Next Row of Annotation Time")
                        break

                try:
                    # Writing key points process
                    if target != 2 and target != 3:
                        xyn = [result[0].keypoints.xyn.tolist()[target]]
                        confs = [result[0].keypoints.conf.tolist()[target]]
                    else:
                        xyn = result[0].keypoints.xyn.tolist()
                        confs = result[0].keypoints.conf.tolist()

                    # Using a for loop with zip
                    for conf_row, xyn_row in zip(confs, xyn):
                        one = [class_label]  # this for pure coordinate
                        two = [class_label]  # this for angel
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
                            two.append(torch.mean(torch.Tensor([conf_row[first], conf_row[mid], conf_row[end]])).item())

                        angel.loc[-1] = two  # adding a row
                        angel.index = angel.index + 1  # shifting index
                        angel = angel.sort_index()  # sorting by index
                except:
                    pass

    filename = f"lapas_ngaseman_{FILENAME}.csv"
    # save the pure coordinate
    pure.to_csv(f"yolov8_extracted_advance/{filename}", index=False)
    # save the angel
    angel.to_csv(f"yolov8_extracted_angel_advance/{filename}", index=False)
    cv2.destroyAllWindows()
