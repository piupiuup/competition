import datetime
import numpy as np
import sys, logging
import pandas as pd
import os, random, gc, re
from collections import OrderedDict,defaultdict

np.random.seed(2016)
random.seed(2016)

shop_gps = {}
mall_shop_dict = defaultdict(lambda : [])
shop_wifi_day_count = {}
shop_wifi_signal = {}
shop_day_count = {}
shop_wifi_day_signal = {}


#shop_id,category_id,longitude,latitude,price,mall_id
count = 0
for line in open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/ccf_first_round_shop_info.csv'):
    if not line:
        continue
    array = line.strip().split(',')
    mall_shop_dict[array[5]].append(array[0])
    shop_wifi_signal[array[0]] = {}
    shop_wifi_day_signal[array[0]] = {}
    shop_gps[array[0]] = (float(0.0), float(0.0))
    shop_wifi_day_count[array[0]] = {}
    shop_day_count[array[0]] = {}

f = open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/mall_info', 'w')
for k, v in sorted(mall_shop_dict.items(), key=lambda x: x[0]):
    f.write(k + "|" + ";".join(sorted(v, key=lambda x: x)))
    f.write('\n')
f.close()

shop_set = set()
# -1,40501,u_9048305,s_305692,2017-08-01 00:00,112.32588100000001,32.688764,b_9789379|-46|false;b_15171116|-40|true;b_30560209|-72|false;b_6658117|-74|false
count = 0
dis = []
md5 = set()
for line in open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/train.csv'):
    if not line:
        continue
    array = line.strip().split(',')[1:]
    time = datetime.datetime.strptime(array[3], '%Y-%m-%d %H:%M')
    shop_set.add(array[3])
    per = 1

    count += 1
    matric = {}
    wifiset = set()
    for wifi in array[6].split(';'):
        wifiname = wifi.split('|')[0]
        if wifiname not in wifiset:
            wifiset.add(wifiname)
            shop_wifi_signal[array[1]][wifiname] = shop_wifi_signal[array[1]].get(wifiname, 0) + 80 + int(wifi.split('|')[1])
            shop_wifi_day_count[array[1]][wifiname] = shop_wifi_day_count[array[1]].get(wifiname, {})
            shop_wifi_day_count[array[1]][wifiname][time.strftime("%Y%m%d")] = shop_wifi_day_count[array[1]][wifiname].get(
                time.strftime("%Y%m%d"), 0) + 1
            shop_wifi_day_signal[array[1]][wifiname] = shop_wifi_day_signal[array[1]].get(wifiname, OrderedDict())
            shop_wifi_day_signal[array[1]][wifiname][time.strftime("%Y%m%d")] = shop_wifi_day_signal[array[1]][wifiname].get(
                time.strftime("%Y%m%d"), [])
            shop_wifi_day_signal[array[1]][wifiname][time.strftime("%Y%m%d")].append(80 + int(wifi.split('|')[1]))
    shop_gps[array[1]] = (shop_gps[array[1]][0] + float(array[3]), shop_gps[array[1]][1] + float(array[4]))

    shop_day_count[array[1]][time.strftime("%Y%m%d")] = shop_day_count[array[1]].get(time.strftime("%Y%m%d"), 0) + 1
    if count % 100000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")

print(count)

f = open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/shop_loc2', 'w')
for k, v in sorted(shop_wifi_day_signal.items(), key=lambda x: x[0]):
    res2 = []
    if k not in shop_set:
        print(k)
    shop_count_alla = sum(shop_day_count.get(k, {}).values())
    for wifi, value in sorted(v.items(), key=lambda x: -sum(map(lambda y: sum(y[1]), x[1].items()))):
        temp = []
        for k2, v2 in value.items():
            temp.extend(v2)
        res = temp
        m = sum(res) / len(res)
        wificount_all = sum(map(lambda x: len(x[1]), filter(lambda y: y[0] >= "20170826", value.items())))
        shop_count_all = sum(map(lambda x: x[1], filter(lambda y: y[0] >= "20170826", shop_day_count[k].items()))) + 0.01
        if (len(res) / shop_count_alla > 0.05 or len(res) > 20 or (wificount_all / shop_count_all > 0.13 and wificount_all > 2 and shop_count_all > 8)) and\
                (m * len(res) / (shop_count_alla) > -4.0 ):
            res2.append(wifi + "," + str(m) + "," + str(len(res)))
    f.write(k + "|" + ";".join(res2) + "|" + str(shop_count_alla) + "|" + str(
        shop_gps[k][0] / (shop_count_alla + 0.001)) + ',' + str(shop_gps[k][1] / (shop_count_alla + 0.001)))
    f.write('\n')
f.close()


wifilist = set()
f = open('C:/Users/csw/Desktop/mayi/tf_bilstm_atten_v2/data/wifi_whitelist_dis', 'w')
for k, v in sorted(shop_wifi_signal.items(), key=lambda x: x[0]):
    res = []
    if k not in shop_set:
        print(k)
    shop_count_alla = sum(shop_day_count.get(k, {}).values())
    for wifi, value in sorted(v.items(), key=lambda x: -x[1]):
        shop_count_wifi = sum(map(lambda x: x[1], shop_wifi_day_count.get(k).get(wifi, {}).items())) + 0.01
        if shop_count_wifi > 1 and shop_count_wifi / shop_count_alla > 0.02 or (shop_count_alla < 5 and shop_count_wifi == 1):
            wifilist.add(wifi)
print(len(wifilist))
f.write(",".join(wifilist))
f.write('\n')
f.close()














