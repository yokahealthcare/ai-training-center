import cv2
import mediapipe as mp
from multiprocessing import Pool


def process_frame(frame, pose, drawing):

    return frame


if __name__ == "__main__":
    # Read a video file or capture from a camera
    cap = cv2.VideoCapture("../sample/world720.mp4")

    # Initialize mediapipe outside the pool
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Create a multiprocessing pool
    with Pool() as pool:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Submit tasks to the pool, pass pose and drawing as arguments
            result = pool.apply(process_frame, (frame, mp_pose, mp_drawing))

            # Display the processed frame
            cv2.imshow("Processed Frame", result)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
