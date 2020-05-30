from multiprocessing import Pool
from time import sleep
from recruit.feat2 import *
import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

def main():
    pool = Pool(processes=8)    # set the processes max number 3
    train_feat = pd.DataFrame()
    start_date = '2017-03-12'
    periods = [[14,28,56,1000],
               [3,15,40,200],
               [5,17,45,230],
                [7,19,50,260],
                [9,21,55,290],
                [11,23,60,320],
                [13,25,65,350],
                [20,50,150,1000],
                [10,40,100,1000],
                [10,25,80,300],
              ]
    preds = []
    for period in periods:
        for i in range(58):
            pool.apply_async(make_feats, (date_add_days(start_date, i*(-7)),39,period,))
        for i in range(1,6):
            pool.apply_async(make_feats, (date_add_days(start_date,i*(7)),42-(i*7),period,))
        pool.apply_async(make_feats, (date_add_days(start_date, 42),39,period,))

    pool.close()
    pool.join()

# def main():
#     pool = Pool(processes=8)    # set the processes max number 3
#     start_date = '2017-01-29'
#     for i in range(52):
#         result = pool.apply_async(make_feats, (date_add_days(start_date, i * (-7)), 39,))
#     for i in range(1, 6):
#         result = pool.apply_async(make_feats, (date_add_days(start_date, i * (7)), 42 - (i * 7),))
#     pool.close()
#     pool.join()
#     if result.successful():
#         print('successful')


if __name__ == "__main__":
    main()

