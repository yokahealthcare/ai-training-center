import time
from itertools import count
from pprint import pprint

import cv2
import mediapipe as mp
import torch


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
    # Model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    person_detector = PersonDetection(yolo_model)
    single_pose = SinglePoseEstimation()

    cap = cv2.VideoCapture("dataset/video/person2.mkv")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # total_inference_time = 0
    for i in count():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        person_detector.set_frame(frame)
        person_detector.detect()

        person_coordinate = person_detector.get_coordinate()
        number_of_person = len(person_coordinate)

        for person in person_coordinate:
            x1, y1, x2, y2, confidence, _ = person.to(int)

            cropped_frame = frame[y1:y2, x1:x2]

            single_pose.set_frame(cropped_frame)
            single_pose.estimate()

            frm = single_pose.get_annotated_frame()
            frame[y1:y2, x1:x2] = frm

        cv2.imshow("webcam", frame)

        end = time.time()
        inference_time = end - start
        print(f"\rInference time : {round(inference_time, 3)} second | {round((end - start) * 1000, 3)} ms", end='',
              flush=True)

        # total_inference_time += inference_time
        # if i % 100 == 0 and i != 0:
        #     avg_inference_time = round(total_inference_time / 100, 3)
        #     print(f"Average of Inference per 100 frame : {avg_inference_time} second | {avg_inference_time * 1000} ms")
        #     total_inference_time = 0

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
