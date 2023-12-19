import cv2
import numpy as np
import joblib
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
    yolo = YoloPoseEstimation("yolo_model/yolov8n-pose.pt")
    loaded_model = joblib.load("training-area/MODEL/angel/logistic_regression.pkl")

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

    for result in yolo.estimate("dataset/video/v2_erwin.mkv"):
        # Wait for a key event and get the ASCII code
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        try:
            # get the data
            xyn = result.keypoints.xyn[0].tolist()
            confs = result.keypoints.conf[0].tolist()

            # this is gathering angel data
            temp = []
            for n in need:
                # index
                first = n[0]
                mid = n[1]
                end = n[2]

                # get data using the index before
                # getting angel from three coordinate
                temp.append(calculate_angle(xyn[first], xyn[mid], xyn[end]))

            # do prediction
            pred = loaded_model.predict(np.array(temp).reshape(1, -1))
            print(f"Prediction : {pred}")

        except TypeError as te:
            pass

        cv2.imshow("webcam", result.plot())