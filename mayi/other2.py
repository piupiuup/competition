import csv
import shelve
import os
from math import sqrt, pow


def merge_data():
    merge = open('merge.csv', 'w+', encoding='utf8', newline='')
    fd = open('../ccf_first_round_user_shop_behavior.csv','r')
    fd2 = open('../ccf_first_round_shop_info.csv','r')
    data_orgin2 = csv.DictReader(fd2)
    load_file = csv.writer(merge)
    map_data = {}
    for n, data in enumerate(data_orgin2):
        map_data[data['shop_id']] = [data['category_id'],data['longitude'],data['latitude'],data['price'],data['mall_id']]

    data_orgin = csv.DictReader(fd)
    load_file.writerow(['category_id', 'shop_longitude', 'shop_latitude', 'shop_price', 'mall_id', 'shop_id', 'longitude', 'latitude', 'wifi_infos'])
    for data in data_orgin:
        li = map_data[data['shop_id']]+[data['shop_id'],data['longitude'],data['latitude'],data['wifi_infos']]
        load_file.writerow(li)

def clsfy():
    if os.path.exists('../data/cls.pkl.dat'):
        cache = shelve.open('../data/cls.pkl', 'c')
        return cache
    origin = open('merge.csv', 'r', encoding='utf8', newline='')
    cache = shelve.open('../data/cls.pkl', 'c')
    data = csv.DictReader(origin)
    for d in data:
        if cache.get(d['shop_id']) is None:
            cache[d['shop_id']] = [[d['category_id'], d['shop_longitude'], d['shop_latitude'], d['shop_price'], d['mall_id'], d['longitude'], d['latitude'], d['wifi_infos']]]
        else:
            cache[d['shop_id']]+=[[d['category_id'], d['shop_longitude'], d['shop_latitude'], d['shop_price'], d['mall_id'], d['longitude'], d['latitude'], d['wifi_infos']]]
    cache.close()
    return cache

def shop(cache):
    if os.path.exists('../data/shop.pkl.dat'):
        sign = shelve.open('../data/shop.pkl', 'c')
        return sign
    wifi = {}
    sign = shelve.open('../data/shop.pkl', 'c')
    for k, v in cache.items():
        wifi[k] = {}
        for n, c in enumerate(v):
            tmp = list(map(lambda x:x.split('|')[:-1],c[-1].split(';')))
            tmp.sort(key=lambda x:x[1],reverse=False)
            del c[-1]
            c += tmp
            for n, t in enumerate(tmp):
                wifi[k][t[0]] = wifi[k].get(t[0],[0,0])
                wifi[k][t[0]][0] += 1
                wifi[k][t[0]][1] += int(t[1])
        result = sorted(wifi[k].items(),key=lambda x:x[1][0],reverse=True)
        # 某个商店前10的wifi信号，个数，比重
        wifi_tmp = {}
        for r in result[:15]:
            wifi_tmp[r[0]] = r[1]
        sign[k] = wifi_tmp
    s_sign = sign
    sign.close()
    return s_sign

def addr(cache):
    if os.path.exists('../data/address.pkl.dat'):
        address = shelve.open('../data/address.pkl', 'c')
        return address
    address = shelve.open('../data/address.pkl', 'c')
    for k, v in cache.items():
        for n, c in enumerate(v):
            address[k] = address.get(k,[0,float(c[:-1][5]),float(c[:-1][5]),float(c[:-1][6]),float(c[:-1][6]),float(c[:-1][1]),float(c[:-1][2])])
            r = 10000*sqrt(pow((float(c[:-1][1])-float(c[:-1][5])),2)+pow((float(c[:-1][2])-float(c[:-1][6])),2))
            tmp = address[k]
            if tmp[0] < r:
                tmp[0] = r
            if tmp[1] < float(c[:-1][5]):
                tmp[1] = float(c[:-1][5])
            if tmp[2] > float(c[:-1][5]):
                tmp[2] = float(c[:-1][5])
            if tmp[3] < float(c[:-1][6]):
                tmp[3] = float(c[:-1][6])
            if tmp[4] > float(c[:-1][6]):
                tmp[4] = float(c[:-1][6])
            if tmp[1] < float(c[:-1][1]):
                tmp[1] = float(c[:-1][1])
            if tmp[2] > float(c[:-1][1]):
                tmp[2] = float(c[:-1][1])
            if tmp[3] < float(c[:-1][2]):
                tmp[3] = float(c[:-1][2])
            if tmp[4] > float(c[:-1][2]):
                tmp[4] = float(c[:-1][2])
            address[k] = tmp
    adr = address
    address.close()
    return adr

if __name__ == '__main__':
    # 1、合并文件
    # 字段顺序
    # 'category_id','longitude','latitude','price','mall_id','shop_id','longitude','latitude','wifi_infos'
    # merge_data()
    # 2、分类
    cache = clsfy()
    # for k,v in cache.items():
    #     print(i)
    #     exit()
    # 3、预处理计算items wifi
    item = shop(cache)
    # for i in item.items():
    #     print(i)
    #     exit()
    # 4、处理地理位置信息
    # [最长半径，上，下，左，右,源,源]
    adr = addr(cache)
    # for i in adr.items():
    #     print(i)
    #     exit()
    # 5 、预测函数
    # ret = csv.writer(open('../result/result1.csv', 'w+', encoding='utf8', newline=''))
    # merge = open('evaluation_public1.csv', 'r+', encoding='utf8', newline='')
    ret = csv.writer(open('../result/errorresult.csv', 'a+', encoding='utf8', newline=''))
    ret.writerow(['row_id', 'shop_id'])
    merge = open('error.csv', 'r+', encoding='utf8', newline='')
    prt = csv.reader(merge)
    prt.__next__()
    def sigmod(avg, fi,sig):
        tmp = 1 / 15 * (1 - abs(abs((avg - float(fi))) / 90))
        #
        # tmp = ((sig*(2/3)) / 105) * (1 - abs(abs((avg - float(fi))) / 90))
        return tmp
    # 测试数据
    uu = 0
    for m in prt:

        # print("需要预测的数据",m)
        gailv = {}
        uu+=1
        # print(m[:-1])
        wifi = list(map(lambda x: x.split('|')[:-1], m[-1].split(';')))
        adrx, adry = float(m[4]),float(m[5])
        # 所有wifi信息

            # exit()
        for k,v in item.items():
            t = {}
            n = 0
            for g in v.items():
                # print(g)
                t[g[0]] = g[1]
                n+=1
                if n ==7:
                    break
            v = t
            # print("某个商铺的前8信息",k,v)
            for prate, w in enumerate(wifi):
                sig = (14-prate)
                # print('一个wifi信息', w)
                pos =gailv.get(k,0)
                tmp = v.get(w[0])
                rate = 0
                if tmp is not None:
                    avg = float(tmp[1])/tmp[0]
                    # print(w,"平均信号",avg)
                    rate = sigmod(avg,float(w[1]),sig)
                    if rate < 0:
                        rate = 0
                    # print("信号有效率",rate)
                pos += rate
                gailv[k] = pos
                # print(pos)
            # exit()
        for k,v in adr.items():
            pos = gailv.get(k, 0)
            rate = 0
            if (adrx <= v[1]) and (adrx >= v[2]) and (adry<=v[3]) and (adry>=v[4]):
                rate = 1/3
                pos += rate
            else:
                rate = 0
            gailv[k] = pos
        result = sorted(gailv.items(), key=lambda x: x[1], reverse=True)
        ret.writerow([m[0],result[:10]])
        print(uu)
