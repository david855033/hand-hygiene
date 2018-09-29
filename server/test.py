import threading
from time import sleep


def job():
    i = 0
    while True:
        print(i)
        i += 1
        sleep(1)


t = threading.Thread(target=job)
t.setDaemon(True)
t.start()
