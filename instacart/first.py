import numpy as np
import pandas as pd
import lightgbm as lgb

order_products_train_path = r'C:\Users\csw\Desktop\python\instacart\data\order_products__train.csv'
order_products_prior_path = r'C:\Users\csw\Desktop\python\instacart\data\order_products__prior.csv'
orders_path = r'C:\Users\csw\Desktop\python\instacart\data\orders.csv'
products_path = r'C:\Users\csw\Desktop\python\instacart\data\products.csv'
aisles_path = r'C:\Users\csw\Desktop\python\instacart\data\aisles.csv'
departments_path = r'C:\Users\csw\Desktop\python\instacart\data\departments.csv'



order_products_train_df = pd.read_csv(order_products_train_path)
order_products_prior_df = pd.read_csv(order_products_prior_path)
orders_df = pd.read_csv(orders_path)
products_df = pd.read_csv(products_path)
aisles_df = pd.read_csv(aisles_path)
departments_df = pd.read_csv(departments_path)