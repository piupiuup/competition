from tool.tool import *


def replace_white_user_id(submission):
    white_user_id = pd.read_csv(r'white_user_id.csv')
    white_user_id = white_user_id[white_user_id['user_id']!='4VNcD6kE0sjnAvFX']
    white_dict = dict(zip(white_user_id['user_id'],white_user_id['current_service'].astype(int)))
    # submission = pd.read_csv(r'C:\Users\csw\Desktop\python\dianxin\submission\.csv')
    temp = submission['current_service'].values.copy()
    submission['current_service'] = list(map(lambda x,y: y if x not in white_dict else white_dict[x],submission['user_id'],submission['current_service']))
    print('修改了{}个'.format(sum(temp!=submission['current_service'].values)))
    # submission.to_csv(r'C:\Users\csw\Desktop\python\dianxin\submission\.csv',index=False)
    return submission



