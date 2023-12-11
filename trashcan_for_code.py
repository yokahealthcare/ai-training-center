# MULTIPROCESS using multiprocessing.Process

from multiprocessing import Process, Queue

WAITING_TIMEOUT = 10


class SkeletalDetector(Process, SinglePoseEstimation):
    def __init__(self):
        Process.__init__(self)
        SinglePoseEstimation.__init__(self)
        self.qq = Queue()
        self.qq_out = Queue()
        self.waiting_timeout = WAITING_TIMEOUT

    def generator_function(self):
        for i in itertools.count():
            try:
                data = self.qq_out.get(timeout=1)
                yield data
            except Exception:
                if self.waiting_timeout <= 0:
                    return
                self.waiting_timeout -= 1
                print(f'\nWaiting for annotated frame, before exiting...{self.waiting_timeout}')

    def run(self):
        # Initiating loop for the process
        for i in itertools.count():
            try:
                data = self.qq.get(timeout=1)
                # If data equal to None then stop the multiprocess
                if data is None:
                    return

                start = time.time()

                self.set_frame(data)
                self.estimate()
                self.qq_out.put(self.get_annotated_frame())

                end = time.time()
                print(f"\rInference time inside multiprocessing : {round(end-start)}", end="", flush=True)

                # Reset the waiting timeout
                self.waiting_timeout = 10
            except Exception as e:
                # Exception will execute one second each time there is no data in Queue
                # Below if waiting timeout exceed then stop the multiprocess
                if self.waiting_timeout <= 0:
                    return

                # Decrease waiting timeout by one
                self.waiting_timeout -= 1
                print(f'\nWaiting for frame to estimate, before exiting...{self.waiting_timeout}')