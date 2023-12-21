import sys

import cv2
from tqdm import tqdm

NAME = sys.argv[1]
vidcap = cv2.VideoCapture(f'{NAME}.mp4')
success, image = vidcap.read()
count = 0
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

# Use tqdm for the progress bar
with tqdm(total=total_frames, desc=f"Extracting frames from {NAME}.mp4") as pbar:
    while success:
        cv2.imwrite(f"{NAME}/frame{count:04d}.jpg", image)  # Save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        pbar.update(1)  # Update progress bar

print(f"Successfully extracted frames from {NAME}.mp4")
