import multiprocessing
import time
import cv2


def process_frame(args):
    frame, callback = args

    # Process the frame here
    processed_frame = frame * 2

    time.sleep(2)

    # Use the callback to yield the result
    callback(processed_frame)


def worker_callback(result):
    print("Received result:", len(result))


class OpenCVVideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self):
        """Enter the context (used with 'with' statement)."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context (used with 'with' statement)."""
        self.cap.release()

    def get_frame(self):
        """Generator function to yield frames from the input video."""
        while True:
            ret, frm = self.cap.read()
            if not ret:
                break
            yield frm


if __name__ == "__main__":
    # Create a multiprocessing-files pool
    pool = multiprocessing.Pool()

    with OpenCVVideoCapture() as video_capture:
        # Prepare iterable for frames and callback
        frame_iterator = video_capture.get_frame()
        args_list = [(next(frame_iterator), worker_callback) for _ in range(video_capture.total_frames)]

        # Use the pool to map the process_frame function to the iterable
        pool.map(process_frame, args_list)

    # Close the pool to free up resources
    pool.close()

    # Wait for all worker processes to finish
    pool.join()
