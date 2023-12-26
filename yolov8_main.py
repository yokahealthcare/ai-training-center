import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


class YoloPoseEstimation:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None

    def estimate(self, input):
        self.result = self.model(input, stream=True, persist=True)
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


def is_coordinate_zero(c1, c2, c3):
    if c1 == [0, 0] and c2 == [0, 0] and c3 == [0, 0]:
        return True
    else:
        return False


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


class FightDetection:
    def __init__(self, fight_model, fps):
        # Architect the deep learning structure
        self.input_size = 16
        self.hidden_size = 8
        self.output_size = 1
        self.model = ThreeLayerClassifier(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(torch.load(fight_model))
        self.model.eval()  # Set to evaluation mode

        # Coordinate for angel
        self.coordinate_for_angel = [
            [8, 6, 2],
            [11, 5, 7],
            [6, 8, 10],
            [5, 7, 9],
            [6, 12, 14],
            [5, 11, 13],
            [12, 14, 16],
            [11, 13, 15]
        ]

        # Set up the thresholds
        self.threshold = 0.8  # Dictate how deep learning is sure there is fight on that frame
        self.conclusion_threshold = 3  # Dictate how hard the program conclude if there is fight in the scene (1 - 3)
        self.FPS = fps

        # Event variables
        self.is_fight_occur = False

    def detect(self, conf, xyn):
        input_list = []
        keypoint_unseen = False
        fight_detected = 0
        for n in self.coordinate_for_angel:
            # Keypoint number that we want to make new angel
            first, mid, end = n[0], n[1], n[2]

            # Gather the coordinate with keypoint number
            c1, c2, c3 = xyn[first], xyn[mid], xyn[end]
            # Check if all three coordinate of one key points is all zeros
            if is_coordinate_zero(c1, c2, c2):
                keypoint_unseen = True
                break
            else:
                # Getting angel from three coordinate
                input_list.append(calculate_angle(c1, c2, c3))
                # Getting the confs mean of three of those coordinate
                conf1, conf2, conf3 = conf[first], conf[mid], conf[end]
                input_list.append(torch.mean(torch.Tensor([conf1, conf2, conf3])).item())

        if keypoint_unseen:
            return

        # Make a prediction
        prediction = self.model(torch.Tensor(input_list))
        if prediction.item() > self.threshold:
            # FIGHT
            # this will grow exponentially according to number of person fighting on scene
            # if there is two person, and this will be added 2 for each frame
            fight_detected += 1
        else:
            # NO FIGHT
            # this if statement is for fight_detected not exceed negative value
            if fight_detected > 0:
                fight_detected -= 3
                # this value will decide how hard the program will conclude there is a fight in the frame
                # the higher the value, the more hard program to conclude

        # Threshold for fight_detected value, when it concludes there is fight on the frame
        # THRESHOLD = FPS * NUMBER OF PERSON DETECTED
        if fight_detected > self.FPS * len(conf):
            self.is_fight_occur = True

    def annotate(self, frame, box):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 2

        # Choose the position to put the text
        text_position = (int(box[2]), int(box[3]))

        TEXT = "FIGHT" if self.is_fight_occur else "?"

        # Add text to the image
        cv2.putText(frame, f"{TEXT}", text_position, font, font_scale, font_color, font_thickness)

        return frame


YOLO_MODEL = "yolo_model/yolov8n-pose_openvino_model"
FIGHT_MODEL = "training-area/MODEL/angel/lapas_ngaseman.pth"
FPS = 20
if __name__ == "__main__":
    fdet = FightDetection(FIGHT_MODEL, FPS)
    yolo = YoloPoseEstimation(YOLO_MODEL)
    for result in yolo.estimate("dataset/lapas ngaseman/CCTV FIGHT/NO_FIGHT_775_825.mp4"):
        # Wait for a key event and get the ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Get the result image from YOLOv8
        result_frame = result.plot()

        try:
            boxes = result.boxes.xyxy.tolist()
            xyn = result.keypoints.xyn.tolist()
            confs = result.keypoints.conf
            if confs is None:
                confs = []
            else:
                confs = confs.tolist()

            # Prediction start here
            for conf, xyn, box in zip(confs, xyn, boxes):
                # Fight Detection
                fdet.detect(conf, xyn)

                # Plot
                result_frame = fdet.annotate(result_frame, box)

            cv2.imshow("webcam", result_frame)
        except TypeError as te:
            pass

    cv2.destroyAllWindows()
