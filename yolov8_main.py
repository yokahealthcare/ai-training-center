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
        self.result = self.model.track(input, stream=True, tracker="bytetrack.yaml", conf=0.4, persist=True)
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


class FightDetector:
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
        self.conclusion_threshold = 2  # Dictate how hard the program conclude if a person is in fight action (1 - 3)
        self.FPS = fps

        # Event variables
        self.fight_detected = 0

    def detect(self, conf, xyn):
        input_list = []
        keypoint_unseen = False
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
            self.fight_detected += 1
        else:
            # NO FIGHT
            # this if statement is for fight_detected not exceed negative value
            if self.fight_detected > 0:
                self.fight_detected -= self.conclusion_threshold
                # this value will decide how hard the program will conclude there is a fight in the frame
                # the higher the value, the more hard program to conclude

        # Threshold for fight_detected value, when it concludes there is fight on the frame
        # THRESHOLD = FPS * NUMBER OF PERSON DETECTED
        if self.fight_detected > self.FPS:
            return True
        else:
            return False


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate area of intersection
    area_inter = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate area of individual bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    area_union = area_box1 + area_box2 - area_inter

    # Calculate IoU
    iou = area_inter / area_union if area_union > 0 else 0.0
    return iou


def calculate_all_ious(bounding_boxes):
    num_boxes = len(bounding_boxes)
    ious = []

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            ious.append(calculate_iou(bounding_boxes[i], bounding_boxes[j]))

    return ious


YOLO_MODEL = "yolo_model/yolov8n-pose_openvino_model"
FIGHT_MODEL = "training-area/MODEL/angel/lapas_ngaseman.pth"
FPS = 20
FIGHT_ON = False
FIGHT_ON_TIMEOUT = 20  # second

if __name__ == "__main__":
    fdet = FightDetector(FIGHT_MODEL, FPS)
    yolo = YoloPoseEstimation(YOLO_MODEL)
    for result in yolo.estimate("dataset/video/CCTV_violent.mp4"):
        # Wait for a key event and get the ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Get the result image from YOLOv8
        result_frame = result.plot()

        try:
            boxes = result.boxes.xyxy.tolist()
            xyn = result.keypoints.xyn.tolist()
            confs = result.keypoints.conf
            ids = result.boxes.id
            dict_of_action = None

            confs = [] if confs is None else confs.tolist()
            ids = [] if ids is None else [str(int(ID)) for ID in ids]
            dict_of_action = {ID: {'ACTION': False, 'ENEMY': None} for ID in ids}

            # Processing interaction box
            interaction_boxes = []
            print("IoU between bounding boxes:")
            for i, iou in enumerate(calculate_all_ious(boxes)):
                print(f"Box {i + 1} and Box {i + 2}: {iou:.4f}")
                if iou > 0.05:
                    try:
                        # Create interaction box coordinate
                        interaction_coordinate = [
                            min(boxes[i][0], boxes[i + 1][0]),  # x1
                            min(boxes[i][1], boxes[i + 1][1]),  # y1
                            max(boxes[i][2], boxes[i + 1][2]),  # x2
                            max(boxes[i][3], boxes[i + 1][3])  # y2
                        ]
                        interaction_boxes.append(interaction_coordinate)
                    except IndexError:
                        pass

            # Interaction Box
            for inter_box in interaction_boxes:
                cv2.rectangle(result_frame, (int(inter_box[0]), int(inter_box[1])),
                              (int(inter_box[2]), int(inter_box[3])), (0, 255, 0), 2)

                # Prediction start here - per person - all person on the frame - including outside the interaction box
                both_fighting = []
                for conf, xyn, box, identity in zip(confs, xyn, boxes, ids):
                    # Check if the person is within the interaction box - filter only person inside interaction box
                    center_person_x, center_person_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
                    if inter_box[0] <= center_person_x <= inter_box[2] and inter_box[1] <= center_person_y <= inter_box[
                        3]:
                        # Fight Detection
                        is_person_fighting = fdet.detect(conf, xyn)
                        both_fighting.append(is_person_fighting)

                # Check if both fighting
                if all(both_fighting) or FIGHT_ON:
                    cv2.putText(result_frame, "FIGHTING", (int(inter_box[2]), int(inter_box[3])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    FIGHT_ON = True

        except TypeError as te:
            pass
        except IndexError as ie:
            pass

        cv2.imshow("webcam", result_frame)

        # RING THE ALARM
        if FIGHT_ON:
            print("RINGGGGGG")
            FIGHT_ON_TIMEOUT -= 1 / FPS

        if FIGHT_ON_TIMEOUT <= 0:
            FIGHT_ON = False
            FIGHT_ON_TIMEOUT = 20

    cv2.destroyAllWindows()
