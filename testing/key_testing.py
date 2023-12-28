import cv2

# Open a video source (0 for default camera, or provide the video file path)
video_source = 0  # Use the default camera (change to a file path if reading from a video file)
cap = cv2.VideoCapture(video_source)

# Check if the video source is successfully opened
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Display the frame (or perform any other processing)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press

    # Check if the pressed key is a numeric key (0 to 9)
    if ord('0') <= key <= ord('9'):
        target = key - ord('0')
        print(f"Skeletal keypoints saved for target {target}")
    elif key == ord('a'):
        print("LEFT skeletal keypoints saved")
        target = 0
    elif key == ord('d'):
        print("RIGHT skeletal keypoints saved")
        target = 1
    elif key == ord('b'):
        print("BOTH skeletal keypoints saved")
        target = 200
    elif key == ord('v'):
        print("FAST ANNOTATE skeletal keypoints saved")
        target = 300
    elif key == ord('s'):
        print("SKIPPED!")
        continue
    elif key == ord('q'):
        print("Exited! Next Row of Annotation Time")
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
