import sys
import os
import cv2
from tqdm import tqdm

# Get a list of all .mp4 files in the current directory
mp4_files = ["fi5.mkv"]

output_directory = "fi5"
count = 0
# Iterate through each .mp4 file
for NAME in mp4_files:
    video_name = os.path.splitext(NAME)[0]

    vidcap = cv2.VideoCapture(NAME)
    success, image = vidcap.read()
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use tqdm for the progress bar
    with tqdm(total=total_frames, desc=f"Extracting frames from {NAME}") as pbar:
        while success:
            cv2.imwrite(f"{output_directory}/frame{count:04d}.jpg", image)  # Save frame as JPEG file
            success, image = vidcap.read()
            count += 1
            pbar.update(1)  # Update progress bar

    print(f"Successfully extracted frames from {NAME} to {output_directory}")
