import time
from multiprocessing import Process
from threading import Thread


class Task(Thread):
    def __init__(self):
        Thread.__init__(self)
        pass

    def run(self):
        p1 = Something()
        p1.start()

        p2 = Something()
        p2.start()

        p3 = Something()
        p3.start()

        p4 = Something()
        p4.start()


class Something(Process):
    def __init__(self):
        Process.__init__(self)
        pass

    def run(self):
        print("Prcess starting...")
        time.sleep(20)
        print("Process ending...")


t1 = Task()
t1.start()

t2 = Task()
t2.start()
