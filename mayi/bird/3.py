import datetime
import sys,logging
import numpy as np
import pandas as pd
import os,random,gc, re

random.seed(2016)
np.random.seed(2016)


blacklist = {}
for line in open('../data/wifi_blacklist.csv'):
    if not line:
        continue
    array = line.strip().split('|')
    blacklist[array[0]] = set(array[1].split(';'))

whitelist2 = set()
for line in open('./wifi_whitelist_dis'):
    if not line:
        continue
    whitelist2 = set(line.strip().split(','))

shopmatric = {}
shopgps = {}
#u_376,s_2871718,2017-08-06 21:20,122.308291,32.08804,b_6396480|-67|false;b_41124514|-86|false;b_28723327|-90|false;b_6396479|-55|false;b_8764723|-90|false;b_32053319|-74|false;b_5857370|-68|false;b_56326644|-89|false;b_56328155|-77|false;b_5857369|-55|false
count = 0
dis = []
for line in open('../data/train.csv'):
    if not line:
        continue
    array = line.strip().split(',')[1:]
    time = datetime.datetime.strptime(array[2],'%Y-%m-%d %H:%M')

    #if time.strftime("%Y%m%d") > '20170824':
    #    continue
    shopmatric[array[1]] = shopmatric.get(array[1], [])
    shopgps[array[1]] = shopgps.get(array[1], [])
    count+=1
    res = []
    for wifi in array[5].split(';'):
        wifiname = wifi.split('|')[0]
        if wifi.split('|')[2] == 'true':
            res.append(wifiname + "," + str(80 + int(wifi.split('|')[1])))
        else:
            res.append(wifiname + "," + str(80 + int(wifi.split('|')[1])))
    shopmatric[array[1]].append(time.strftime("%Y%m%d")+"|"+";".join(sorted(res,key=lambda x : x.split(',')[0])))
    shopgps[array[1]].append(time.strftime("%Y%m%d")+"|gps1,"+array[3] + ';gps2,' + array[4])
    if count % 100000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")
        #break

print count
f = open('./shop_detail','w')
for k,v in sorted(shopmatric.items(),key=lambda x:x[0]):
    res = []
    resgps = []
    #print sorted(v,key=lambda x:x)
    num = 0
    for value in sorted(v,key=lambda x:x,reverse=True):
        if num == 150:
            break
        num += 1
        res.append(value.split("|")[1])
    num = 0
    for value in sorted(shopgps.get(k), key=lambda x: x, reverse=True):
        if num == 100:
            break
        num += 1
        resgps.append(value.split("|")[1])
    print >> f, k + "\t" + "|".join(res)+ "\t" + "|".join(resgps)+"\t"+str(num)
f.close()
