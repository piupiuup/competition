import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
from pylab import *

#收集道路信息
def collect(n=10):

    x_max = int(np.floor(82202. / n))
    y_max = int(np.floor(128135. / n))

    arr = np.zeros((y_max,x_max))
    train_traffic = pd.read_csv(r'F:\data\traffic2\train\20140803_train.txt', header=None)
    train_traffic = train_traffic.values

    i = 0
    for tup in train_traffic:
        x = int(np.floor(tup[2]/n))
        y = int(np.floor(tup[1]/n))
        if(x>=x_max or x<0 or y>=y_max or y<0):
            continue
        arr[y,x] += 1
        i += 1
        if (i % 1000000 == 0):
            sys.stdout.flush()
            sys.stdout.write("#")

    return arr


#绘制地图
def draw(arr,strong=1,split=1,select=(1,1)):

    height,width = arr.shape
    height = height/split
    width = width/split
    arr = arr[(select[0]-1)*height:select[0]*height,(select[1]-1)*width:select[1]*width]

    arr = arr*strong
    im = Image.fromarray(arr)
    im.show()

#继续收集
def collect(arr,url,n=10):

    train_traffic = pd.read_csv(url, header=None)
    print url
    train_traffic = train_traffic.values

    i = 0
    for tup in train_traffic:
        x = int(np.floor(tup[2]/n))
        y = int(np.floor(tup[1]/n))
        if(x>=x_max or x<0 or y>=y_max or y<0):
            continue
        arr[y,x] += 1
        i += 1
        if (i % 1000000 == 0):
            sys.stdout.flush()
            sys.stdout.write("#")

    return arr

#道路收边
def clear_1(arr,n=5,coefficient=0.5):

    height, width = arr.shape
    clear_matrix = arr<0

    k = 0
    for i in xrange(0,(height-n),1):
        for j in xrange(0,(width-n),1):
            arr_temp = arr[i:(i+n),j:(j+n)]
            try:
                arr_eva = np.percentile(arr_temp[arr_temp>0],100*coefficient)
            except:
                arr_eva = 0
            if arr_eva>1:
                clear_matrix[i:(i+n),j:(j+n)] = clear_matrix[i:(i+n),j:(j+n)] | (arr_temp < arr_eva)

            k += 1
            if k%1000==0:
                sys.stdout.flush()
                sys.stdout.write("#")

    for i in xrange(height):
        for j in xrange(width):
            if clear_matrix[i,j]:
                arr[i,j]=0

    return arr


