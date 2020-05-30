import pandas as pd
import numpy as np
import time

# 时间差（分钟）
def diff_of_minutes(time1,time2):
    try:
        return (time1-time2).seconds/60
    except:
        return np.nan


# 判断是否在某个时间区间
def if_in_period(time, periods):
    if periods is None:
        return False
    else:
        for period in periods:
            if (time>period[0]) & (time<period[1]):
                return True
        return False
# 判断在此区间是否停机
def if_in_stopping(time1, time2, periods):
    if periods is None:
        return False
    else:
        for period in periods:
            if (time1<period[1]) & (time2>period[0]):
                return True
        return False

# 判断是机场是否关闭，判断台风能否起飞，判断台风能否降落，判断能否降落
def if_in_restrict(raw, end_time, ON_time, takeoff_time, landing_time, stopping_time):
    result = []
    result.append(1 if if_in_period(raw['起飞时间'], ON_time[raw['起飞机场']])          else 0)
    result.append(1 if if_in_period(raw['降落时间'], ON_time[raw['降落机场']])          else 0)
    result.append(1 if if_in_period(raw['起飞时间'], takeoff_time[raw['起飞机场']])     else 0)
    result.append(1 if if_in_period(raw['降落时间'], landing_time[raw['降落机场']])     else 0)
    result.append(1 if if_in_stopping(raw['降落时间'], end_time, stopping_time[raw['降落机场']])else 0)
    return result

data_path = r'C:\Users\csw\Desktop\python\aviation\data.xlsx'

schedule = pd.read_excel(data_path,sheetname=0)
schedule['飞行时间'] = schedule.apply(lambda x:diff_of_minutes(x['降落时间'],x['起飞时间']),axis=1)
airline_restrict = pd.read_excel(data_path,sheetname=1)
ON_restrict = pd.read_excel(data_path,sheetname=2)
typhoon = pd.read_excel(data_path,sheetname=3)
time = pd.read_excel(data_path,sheetname=4)
time.rename(columns={'飞机机型':'机型'},inplace=True)
start_time = schedule['起飞时间'].min()
end_time = schedule['降落时间'].max()
airport_list = list(set(list(schedule['起飞机场'])+list(schedule['降落机场'])))
airplane_list = list(set(schedule['飞机ID'].unique()))

# 开始时机场飞机的分布情况
schedule.sort_values(['飞机ID','起飞时间'],inplace=True,ascending=True)
start_airport = schedule.drop_duplicates('飞机ID',keep='first')
grouped = start_airport.groupby('起飞机场')
start_airport = {}
for apt, group in grouped:
    start_airport[apt] = group['飞机ID'].unique().tolist()
for apt in airport_list:
    if apt not in start_airport:
        start_airport[apt] = []

# 最后时机场飞机的分布情况
schedule.sort_values(['飞机ID', '降落时间'], inplace=True, ascending=True)
end_airport = schedule.drop_duplicates('飞机ID', keep='last')
grouped = end_airport.groupby('降落机场')
end_airport = {}
for apt, group in grouped:
    end_airport[apt] = group['飞机ID'].unique().tolist()
for apt in airport_list:
    if apt not in end_airport:
        end_airport[apt] = []

# 开始时飞机在机场的分布情况
schedule.sort_values(['飞机ID', '起飞时间'], inplace=True, ascending=True)
start_airplane = schedule.drop_duplicates('飞机ID', keep='first')
start_airplane = dict(zip(start_airplane['飞机ID'],start_airplane['起飞机场']))

# 结束时飞机在机场的分布情况
schedule.sort_values(['飞机ID', '降落时间'], inplace=True, ascending=True)
end_airplane = schedule.drop_duplicates('飞机ID', keep='last')
end_airplane = dict(zip(end_airplane['飞机ID'],end_airplane['降落机场']))

########## 机场关闭限制 ###########
# 获取机场关闭区间
def get_ON_time(time1,time2,data1,date2,start_time,end_time):
    date_range = pd.date_range(pd.Timestamp(data1),pd.Timestamp(date2)-pd.Timedelta(days=1))
    ON_time = []
    for date in date_range:
        start_time_date = pd.Timestamp(str(date)[:11] + str(time1))
        end_time_date = pd.Timestamp(str(date)[:11] + str(time2))
        if (end_time_date>start_time) & (start_time_date<end_time):
            ON_time.append([max(start_time_date,start_time),min(end_time_date,end_time)])
    return ON_time
ON_dict = {}
for index,row in ON_restrict.iterrows():
    ON_dict[row['机场']] = get_ON_time(row['关闭时间'],row['开放时间'],row['生效日期'],row['失效日期'],start_time,end_time)
for apt in airport_list:
    if apt not in ON_dict:
        ON_dict[apt] = None

##########  台风起飞限制 ###########
def get_typhoon_time(time1,time2,start_time,end_time):
    start_time_date = pd.Timestamp(time1)
    end_time_date = pd.Timestamp(time2)
    if (end_time_date>start_time) & (start_time_date<end_time):
        typhoon_time = [max(start_time_date,start_time),min(end_time_date,end_time)]
    return typhoon_time
# 获取机场关闭起飞区间
takeoff_dict = {}
for index,row in typhoon.iterrows():
    if row['影响类型'] == '起飞':
        if row['机场'] in takeoff_dict:
            takeoff_dict[row['机场']].append(get_typhoon_time(row['开始时间'], row['结束时间'], start_time, end_time))
        else:
            takeoff_dict[row['机场']] = [get_typhoon_time(row['开始时间'],row['结束时间'],start_time,end_time)]
for apt in airport_list:
    if apt not in takeoff_dict:
        takeoff_dict[apt] = None
# 获取机场关闭降落区间
landing_dict = {}
for index,row in typhoon.iterrows():
    if row['影响类型'] == '降落':
        if row['机场'] in landing_dict:
            landing_dict[row['机场']].append(get_typhoon_time(row['开始时间'], row['结束时间'], start_time, end_time))
        else:
            landing_dict[row['机场']] = [get_typhoon_time(row['开始时间'],row['结束时间'],start_time,end_time)]
for apt in airport_list:
    if apt not in landing_dict:
        landing_dict[apt] = None
# 获取机场停机区间
stopping_dict = {}
for index,row in typhoon.iterrows():
    if row['影响类型'] == '停机':
        if row['机场'] in stopping_dict:
            stopping_dict[row['机场']].append(get_typhoon_time(row['开始时间'], row['结束时间'], start_time, end_time))
        else:
            stopping_dict[row['机场']] = [get_typhoon_time(row['开始时间'],row['结束时间'],start_time,end_time)]
for apt in airport_list:
    if apt not in stopping_dict:
        stopping_dict[apt] = None
# 创建机场类
class airport:
    # 初始化
    def __init__(self, name, start_airplanes, end_airplanes, ON_time, takeoff_time, landing_time, stopping_time):
        self.name = name
        self.start_airplanes = start_airplanes
        self.end_airplanes = end_airplanes
        self.ON_time = ON_time
        self.takeoff_time = takeoff_time
        self.landing_time = landing_time
        self.stopping_time = stopping_time
        self.event = pd.DataFrame(columns=['航班ID','时间','飞机ID','起降','是否关闭机场'])

    # 为机场添加事件
    def add_event(self,row):
        if row['起飞机场'] == self.name:
            self.event.loc[self.event.shape[0]] = {'航班ID': row['航班ID'], '时间': row['起飞时间'],
                                             '飞机ID': row['飞机ID'], '起降': 0,
                                             '是否关闭机场': if_in_period(row['起飞时间'], self.ON_time)}
        if row['降落机场'] == self.name:
            self.event.loc[self.event.shape[0]] = {'航班ID': row['航班ID'], '时间': row['降落时间'],
                                             '飞机ID': row['飞机ID'], '起降': 1,
                                             '是否关闭机场': if_in_period(row['起飞时间'], self.ON_time)}

# 创建飞机类
class airplane:
    def __init__(self, name, start_airplane, end_airplane):
        self.name = name
        self.start_airplane = start_airplane
        self.end_airplane = end_airplane
        self.event = pd.DataFrame(columns=['航班ID', '起飞时间', '起飞机场', '降落时间', '降落机场'])

    # 为机场添加事件
    def add_event(self, row):
        self.event.loc[self.event.shape[0]] = {'航班ID': row['航班ID'], '起飞时间': row['起飞时间'],
                                               '起飞机场': row['起飞机场'], '降落时间': row['降落时间'],
                                               '降落机场': row['降落机场'], '重要系数':row['重要系数']}

    # 判断时间是有限制
    def if_district(self, ON_dict, takeoff_dict, landing_dict, stopping_dict):
        district = []
        for i in range(self.event.shape[0]):
            district.append(
                if_in_restrict(self.event.iloc[i], end_time, ON_dict, takeoff_dict, landing_dict,
                               stopping_dict))
        district = np.array(district)
        self.event['机场关闭起飞'] = district[:, 0]
        self.event['机场关闭降落'] = district[:, 1]
        self.event['台风关闭起飞'] = district[:, 2]
        self.event['台风关闭降落'] = district[:, 3]
        self.event['台风停机'] = district[:, 4]

    # 判断限制类型，做出修改
    def adjust(self):





# 为每个机场创建对象
airport_dict = {}
for apt in airport_list:
    airport_dict[apt] = airport(apt,start_airport[apt],end_airport[apt],ON_dict[apt],
                                takeoff_dict[apt],landing_dict[apt],stopping_dict[apt])

# 为每个飞机创建对象
airplane_dict = {}
for apl in airplane_list:
    airplane_dict[apl] = airplane(apl,start_airplane[apl], end_airplane[apl])
# 给每个飞机添加事件
for i,row in schedule.iterrows():
    airplane_dict[row['飞机ID']].add_event(row)
# 判断每个飞机是否有硬限制
for apl in airplane_dict:
    airplane_dict[apl].if_district(ON_dict, takeoff_dict, landing_dict, stopping_dict)
# 对航班修改解决硬限制



# 判断是否 满足航线限制
def if_airline_restrict(takeoff_airport,landing_airport,airplaneID,airline_restrict):
    exist = ((airline_restrict['起飞机场']==takeoff_airport) & (airline_restrict['降落机场']==landing_airport) & (airline_restrict['飞机ID']==airplaneID))
    return False if sum(exist) == 1 else True
schedule['if_airline_restrict'] = schedule.apply(lambda x:if_airline_restrict(x['起飞机场'],x['降落机场'],x['飞机ID'],airline_restrict),aixs=1)

# 判断是否满足机场关闭限制
for i,row in schedule.iterrows():
    airport_dict[row['起飞机场']].add_event(row)
    airport_dict[row['降落机场']].add_event(row)