import sys, logging
import numpy as np
import pandas as pd
import datetime, math
import os, random, gc, re
from collections import defaultdict

random.seed(2016)
np.random.seed(2016)

cache_path = 'F:/mayi_cache2/'
data_path = 'C:/Users/csw/Desktop/mayi/dta/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

# [row_id,user_id,shop_id,time_stamp,longitude,latitude,wifi_infos]

linedict = []
count = 0
for line in open(train_path):
    if not line:
        continue
    array = line.strip().split(',')
    time = datetime.datetime.strptime(array[3], '%Y-%m-%d %H:%M')
    linedict.append((time.strftime("%Y%m%d%H%M"), "-1," + line.strip()))
    count += 1
    if count % 100000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")

f = open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/train.csv', 'w')
for time, line in sorted(linedict, key=lambda x: x[0]):
    f.write(line)
    f.write('\n')
f.close()

for line in open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/evaluation_public.csv'):
    if not line:
        continue
    array = line.strip().split(',')
    time = datetime.datetime.strptime(array[3], '%Y-%m-%d %H:%M')
    linedict.append((time.strftime("%Y%m%d%H%M"), line.strip()))
    count += 1
    if count % 100000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")
'''
f = open('../data/test.csv','w')
for time,line in sorted(linedict,key = lambda x:x[0]):
    if time > '201709010000':
        print >> f,line
f.close()
'''
f = open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/wifi_whitelist.csv', 'w')
f2 = open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/wifi_blacklist.csv', 'w')
t = "20170800"
allwifiset = defaultdict(lambda: 0)
wifiset = defaultdict(lambda: 0)
lastset = defaultdict(lambda: 0)
lastset2 = defaultdict(lambda: 0)
blacklist = set()
for time, line in sorted(linedict, key=lambda x: x[0]):
    array = line.strip().split(',')
    if time[0:8] != t:
        l = []
        temp = []
        for wifiname in blacklist:
            if wifiset.get(wifiname, 0) >= 1:
                temp.append(wifiname)
        for wifiname in temp:
            blacklist.remove(wifiname)
        for wifiname, count in lastset.items():
            if wifiset.get(wifiname, 0) / (lastset.get(wifiname, 0) + 0.01) < 0.2 and lastset.get(wifiname,
                                                                                                  0) - wifiset.get(
                    wifiname, 0) > 20 and \
                                    wifiset.get(wifiname, 0) / (
                                lastset2.get(wifiname, 0) + 0.01) < 0.2 and lastset2.get(wifiname, 0) - wifiset.get(
                wifiname, 0) > 20 and wifiset.get(wifiname, 0) <= 1:
                l.append(wifiname + "," + str(lastset.get(wifiname, 0)) + "," + str(wifiset.get(wifiname, 0)))
                blacklist.add(wifiname)
            elif lastset.get(wifiname, 0) - wifiset.get(wifiname, 0) > 10 and \
                                    lastset2.get(wifiname, 0) - wifiset.get(wifiname, 0) > 10 and wifiset.get(wifiname,
                                                                                                              0) <= 0:
                l.append(wifiname + "," + str(lastset.get(wifiname, 0)) + "," + str(wifiset.get(wifiname, 0)))
                blacklist.add(wifiname)
            elif allwifiset.get(wifiname, 0) - wifiset.get(wifiname, 0) > 50 and wifiset.get(wifiname, 0) <= 0:
                l.append(wifiname + "," + str(lastset.get(wifiname, 0)) + "," + str(wifiset.get(wifiname, 0)))
                blacklist.add(wifiname)
        if len(set(wifiset.keys()) - set(allwifiset.keys())) > 0:
            f.write(t[0:8] + "|" + ";".join(set(wifiset) - set(allwifiset)))
            f.write('\n')
        for wifiname, count in wifiset.items():
            allwifiset[wifiname] = allwifiset.get(wifiname, 0) + count
        if len(blacklist) > 0:
            f2.write(t[0:8] + "|" + ";".join(blacklist))
            f2.write('\n')
        print(time[:8], len(blacklist))
        lastset2 = lastset
        lastset = wifiset
        wifiset = {}
        t = time[0:8]
    for wifi in array[6].split(';'):
        wifiname = wifi.split('|')[0]
        wifiset[wifiname] += 1

print(len(allwifiset))
print(len(list(filter(lambda x: x[1] > 500, allwifiset))))























