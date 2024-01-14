import base64

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
    for result in yolo.estimate("dataset/lapas ngaseman/CCTV FIGHT/FIGHT_355_390.mp4"):
        # Wait for a key event and get the ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Get original image (without annotation, clean image) from YOLOv8
        orig_frame = result.orig_img
        # Get the result image from YOLOv8
        result_frame = result.plot()
        frame_height = result_frame.shape[0]
        frame_width = result_frame.shape[1]
        if frame_height > 720:
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
                cv2.rectangle(result_frame, (int(inter_box[0]), int(inter_box[1])),
                              (int(inter_box[2]), int(inter_box[3])), (0, 255, 0), 2)

                # Prediction start here
                both_fighting = []
                for conf, xyn, box, identity in zip(confs, xyn, boxes, ids):
                    # Check if the person is within the interaction box - filter only person inside interaction box
                    center_person_x, center_person_y = ((box[2] + box[0]) / 2), ((box[3] + box[1]) / 2)
                    if inter_box[0] <= center_person_x <= inter_box[2] and inter_box[1] <= center_person_y <= inter_box[
                        3]:
                        # Fight Detection
                        is_person_fighting = fdet.detect(conf, xyn)
                        both_fighting.append(is_person_fighting)

                    # If fight occur then send cropped face to VMS
                    # For Face Recognition Task
                    if FIGHT_ON:
                        # Left side
                        right_shoulder_x = xyn[6][0]
                        right_ear_x = xyn[4][0]

                        # Right side
                        left_shoulder_x = xyn[5][0]
                        left_ear_x = xyn[3][0]

                        # Take the average of left and right shoulder Y-value
                        left_shoulder_y = xyn[5][1]
                        right_shoulder_y = xyn[6][1]

                        # Take nose Y-value
                        nose_y = xyn[0][1]

                        if (right_shoulder_x != 0 and right_ear_x != 0)\
                            and (left_shoulder_x != 0 and left_ear_x != 0)\
                            and (left_shoulder_y != 0 and right_shoulder_y != 0)\
                            and (nose_y != 0):

                            # Decide which one is the most fartest - shoulder or ear
                            x1 = int(min(right_shoulder_x, right_ear_x) * frame_width)
                            x2 = int(max(left_shoulder_x, left_ear_x) * frame_width)

                            avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                            # Calculate the distance with previous average value and nose Y-value
                            distance_nose_shoulder = abs(avg_shoulder_y - nose_y)
                            # Setting up Y coordinate
                            y1 = int((avg_shoulder_y - (distance_nose_shoulder * 2)) * frame_height)
                            y2 = int(avg_shoulder_y * frame_height)

                            # Negative coordinates not allowed
                            if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
                                width_face = x2-x1
                                height_face = y2-y1
                                # Less than 96 not allowed
                                if width_face >= 96 and height_face >= 96:
                                    cropped_face = orig_frame[y1:y2, x1:x2]
                                    b64 = cv2.imencode('.jpg', cropped_face)[1]
                                    b64 = base64.b64encode(b64)
                                    b64 = str(b64)
                                    b64 = b64[2:-1]

                                    # errmsg, fr = get_fr_data_huawei(b64, x)
                                    # if errmsg == "Success":
                                    #     frs.append(fr)
                else:
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
