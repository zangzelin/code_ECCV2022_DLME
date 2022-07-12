from pynvml import *
import time

if __name__ == "__main__":

    T = 1e7
    go = True

    while go:
        time.sleep(600)
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        print('device number ', deviceCount)
        go_list = []
        for i in range(deviceCount):
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(h)
            print('device {} used {} {}'.format(i, info.used, info.used < T))
            if info.used < T:
                go_list.append(0)
            else:
                go_list.append(1)

        if sum(go_list) == 0:
            go = False
            print('gogogo')
        else:
            go = True
            print('wait')
