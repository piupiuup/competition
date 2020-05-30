import numpy as np
import pandas as pd

#将r=1的球面坐标系转换为欧式空间
def change_1(latitude,longitude):
    a = 180/np.pi
    latitude = latitude/a
    longitude = longitude/a
    x_y = np.cos(latitude)
    x = x_y*np.sin(longitude)
    y = x_y*np.cos(longitude)
    z = np.sin(latitude)
    #print '欧式空间：',(x,y,z)

    return (x,y,z)

#平面坐标系旋转公式
def rotate(coordinate,angle):
    angle = angle/180.0*np.pi
    x,y = coordinate
    x_result = x*np.cos(angle)+y*np.sin(angle)
    y_result = y*np.cos(angle)-x*np.sin(angle)

    return (x_result,y_result)

#选定坐标轴将欧式坐标系映射为平面坐标系,(lat,lon)为中心点
def change_2(coordinate, lat_lon=(30.661203,103.940613)):
    x,y,z = coordinate
    lat,lon = lat_lon
    x,y = rotate((x,y),-lon)
    #print '第一次绕z轴旋转后的坐标：', (x, y, z)
    y,z = rotate((y,z),lat)
    #print '第二次绕x轴旋转后的坐标：', (x, y, z)

    return x,z

#将球面坐标映射为平面坐标
def mapping(latitude,longitude,r=6371004.0):
    x,y,z = chang_1(latitude, longitude)
    x_mapping,y_mapping = chang_2((x,y,z))
    x_mapping = np.arcsin(x_mapping) * r
    y_mapping = np.arcsin(y_mapping) * r

    return round(x_mapping,2),round(y_mapping,2)

#读取文件进行转换
def change_csv(path_source,path_target):
    data = pd.read_csv(path_source,header=None)
    data.sort_values([0,4],inplace=True)
    data = data.values
    for x in data:
        x[1],x[2] = mapping(x[1],x[2])
    data = pd.DataFrame(data)
    data.to_csv(path_target,index=False,header=False)

#计算平面距离
def calu_distance2(coordinate1,coordinate2):
    x1,y1 = coordinate1
    x2,y2 = coordinate2

    return ((x1-x2)**2+(y1-y2)**2)**0.5

#计算大圆距离
def calcu_distance(lon1, lat1, lon2, lat2):
    dx = lon1 - lon2  # 经度差
    dy = lat1 - lat2  # 维度差
    b = (lat1 + lat2) / 2.0;
    Lx = (dx / 57.2958) 0 * np.cos(b / 57.2958)
    Ly = (dy / 57.2958) * 6371004.0

    return (Lx * Lx + Ly * Ly)**0.5


#计算相邻两个点的距离
def distance_list(data):
    pl = list(data[['lon','lat']].values)
    dist_list = []
    for i in range(len(pl)-1):
        dist_list.append(((pl[i][0]-pl[i+1][0])**2+(pl[i][1]-pl[i+1][1])**2)**2)

    return dist_list
