import cv2
import mediapipe as mp
import threading
import queue

def face_detection_thread(input_queue, output_queue):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    while True:
        frame = input_queue.get()

        if frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        output_queue.put(results)

    face_detection.close()

def main():
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    face_thread = threading.Thread(target=face_detection_thread, args=(input_queue, output_queue))
    face_thread.start()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        input_queue.put(frame)

        results = output_queue.get()

        if results.detections:
            for detection in results.detections:
                # Process the results as needed
                pass

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    input_queue.put(None)
    face_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
