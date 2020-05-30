import time
import math
from multiprocessing import Pool


def run(i, size):  # data 传入数据，index 数据分片索引，size进程数
    time.sleep(1)
    print('***************{}***************'.format(i))
    return i

if __name__ == '__main__':
    processor = 3
    res = []
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(run, args=(i, processor,)))
        print(str(i) + ' processor started !')
    p.close()
    print(res)
    p.join()
    print(res)
    print(res[0].get())
    print(res[1].get())
    print(res[2].get())




