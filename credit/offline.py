from tool.tool import *

data_path = 'C:/Users/csw/Desktop/python/credit/data/'

train = pd.read_csv(data_path + 'application_train.csv')
bureau = pd.read_csv(data_path + 'bureau.csv')
bureau_balance = pd.read_csv(data_path + 'bureau_balance.csv')
credit_card_balance = pd.read_csv(data_path + 'credit_card_balance.csv')
#HomeCredit_columns_description = pd.read_csv(data_path + 'HomeCredit_columns_description.csv')
installments_payments = pd.read_csv(data_path + 'installments_payments.csv')
POS_CASH_balance = pd.read_csv(data_path + 'POS_CASH_balance.csv')
previous_application = pd.read_csv(data_path + 'previous_application.csv')
sample_submission = pd.read_csv(data_path + 'sample_submission.csv')


test = train.sample(frac=0.5,random_state=66)
train = train[~train['SK_ID_CURR'].isin(test['SK_ID_CURR'].values)]


offline_path = 'C:/Users/csw/Desktop/python/credit/offline/'

test.to_csv(offline_path + 'application_test.csv',index=False)
train.to_csv(offline_path + 'application_train.csv',index=False)
bureau.to_csv(offline_path + 'bureau.csv',index=False)
bureau_balance.to_csv(offline_path + 'bureau_balance.csv',index=False)
credit_card_balance.to_csv(offline_path + 'credit_card_balance.csv',index=False)
installments_payments.to_csv(offline_path + 'installments_payments.csv',index=False)
POS_CASH_balance.to_csv(offline_path + 'POS_CASH_balance.csv',index=False)
previous_application.to_csv(offline_path + 'previous_application.csv',index=False)
sample_submission.to_csv(offline_path + 'sample_submission.csv',index=False)