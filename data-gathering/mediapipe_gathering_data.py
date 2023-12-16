from pprint import pprint

import cv2
import torch
import mediapipe as mp
import pandas as pd


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

    cap = cv2.VideoCapture("../dataset/video/v2_erwin.mkv")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target = 0
    dt = {
        "class": target
    }
    for i in range(1, 34):
        dt[f"x{i}"] = 0
        dt[f"y{i}"] = 0
        dt[f"z{i}"] = 0
        dt[f"v{i}"] = 0

    df_data = pd.DataFrame(dt, index=range(1, 1))

    while True:
        # Wait for a key event and get the ASCII code
        key = cv2.waitKey(1) & 0xFF

        ret, frame = cap.read()
        if not ret:
            break

        single_pose.set_frame(frame)
        single_pose.estimate()

        if key == ord('f'):
            target = 1
            print("fighting key pressed!")
        elif key == ord('s'):
            target = 0
            print("stop key pressed!")
        elif key == ord('q'):
            break

        try:
            one = [target]
            for i, entry in enumerate(single_pose.landmarks, start=1):
                one.append(entry.x)
                one.append(entry.y)
                one.append(entry.z)
                one.append(entry.visibility)

            df_data.loc[-1] = one  # adding a row
            df_data.index = df_data.index + 1  # shifting index
            df_data = df_data.sort_index()  # sorting by index
        except TypeError as te:
            pass

        result = single_pose.get_annotated_frame()

        cv2.imshow("webcam", result)

    df_data.to_csv("mediapipe_extracted/erwin.csv", index=False)

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
