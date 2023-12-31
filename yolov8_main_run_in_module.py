import cv2
import imutils

import fight_module

YOLO_MODEL = "yolo_model/yolov8n-pose_openvino_model"
FIGHT_MODEL = "training-area/MODEL/angel/lapas_ngaseman_v2.pth"
FPS = 20
FIGHT_ON = False
FIGHT_ON_TIMEOUT = 20  # second

if __name__ == "__main__":
    fdet = fight_module.FightDetector(FIGHT_MODEL, FPS)
    yolo = fight_module.YoloPoseEstimation(YOLO_MODEL)
    for result in yolo.estimate("dataset/lapas ngaseman/CCTV FIGHT/FIGHT_190_230.mp4"):
        # Wait for a key event and get the ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Get the result image from YOLOv8
        result_frame = result.plot()
        if result_frame.shape[0] > 720:
            result_frame = imutils.resize(result_frame, width=1280)

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
            interaction_boxes = fight_module.get_interaction_box(boxes)

            # Interaction Box
            for inter_box in interaction_boxes:
                cv2.rectangle(result_frame, (int(inter_box[0]), int(inter_box[1])), (int(inter_box[2]), int(inter_box[3])), (0, 255, 0), 2)

                # Prediction start here - per person - all person on the frame - including outside the interaction box
                both_fighting = []
                for conf, xyn, box, identity in zip(confs, xyn, boxes, ids):
                    # Check if the person is within the interaction box - filter only person inside interaction box
                    center_person_x, center_person_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
                    if inter_box[0] <= center_person_x <= inter_box[2] and inter_box[1] <= center_person_y <= inter_box[3]:
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
