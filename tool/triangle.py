import numpy as np
def get_distances(angle21,angle22,distance,angle52):
    '''
    :param angle21: 二车俯仰角
    :param angle22: 二车方位角
    :param distance: 二车五车距离
    :param angle52: 五车方位角
    :param angle2_5: 五车对二车的方位角
    :return: 目标距离二车距离 和 目标距离五车距离
    '''
    angle21 = angle21 / 180 * np.pi
    angle22 = angle22 / 180 * np.pi
    angle52 = angle52 / 180 * np.pi

    L2 = distance * np.cos(angle52) / np.sin(angle22 - angle52) / np.cos(angle21)

    return L2

if __name__ == "__main__":
    angle21 = float(input("请输入二车俯仰角："))
    angle22 = float(input("请输入二车方位角："))
    distance = float(input("请输入二车距离五车的距离："))
    angle52 = float(input("请输入五车方位角："))
    L2, L5 = get_distances(angle21, angle22, distance,  angle52)
    print('目标距离二车%f米。' % L2)
    print('目标距离五车%f米。' % L5)