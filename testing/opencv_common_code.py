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

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
