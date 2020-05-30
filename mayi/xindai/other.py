

def initFeatureList():

    feature_list.extend([

    Feature(prefix='age', startid=1, name='age',drop=True),

    Feature(prefix='sex', startid=1, name='sex', drop=True),
    Feature(prefix='age_sex', startid=1, name='age_sex',drop=True),


    #active_date:激活日期
    #加上drop=False 反而下降了
    Feature(prefix='active_date', startid=1, name='active_date',drop=False),

    #实测有效
    Feature(prefix='days_to_now', startid=1, name='days_to_now', drop=False),
    Feature(prefix='limit', startid=1, name='limit',drop=False),


    #品类购买金额
    Feature(prefix='buy_cate', startid= 1, name = 'buy_cate') ,
    Feature(prefix='buy_sum', startid=1, name='buy_sum'),

    Feature(prefix='buy_cate_discount', startid=1, name='buy_cate_discount'),
    Feature(prefix='buy_discount_sum', startid=1, name='buy_discount_sum'),


    Feature(prefix='click', startid=1,name='click') , #历史点击某个商品的次数
    Feature(prefix='click_param', startid=1, name='click_param'),  # 历史点击某个商品+param的次数
    # Feature(prefix='order_click', startid=1,name='order_click'),
    ])
def initStatFeatureList():
    """
    最近一个月的统计贷款数
    :return:
    """
    stat_list.extend([
        #上一个月的借贷均值
        StatFeature(prefix='age',startid=1,  expand=True,name='age' , idfile='user.id', drop=True),
        StatFeature(prefix='sex',startid=1,  expand=True, name='sex', idfile='user.id', drop=True),
        StatFeature(prefix='active_date', startid=1,  expand=True, name ='active_date',  idfile='user.id' ,drop=True),

        #limit stat特征 重要度低  这个特征也基本没用，后续废弃
        StatFeature(prefix='limit', startid=1, expand=True, name='limit', idfile='user.id',drop=True),
        StatFeature(prefix='age_sex',startid=1, expand=True , name= 'age_sex', idfile='user.id', drop=True),
        # 类别购买的平均贷款额度
        StatFeature(prefix='buy_cate_avg_loan', startid=1,  expand=True, name='buy_cate_avg_loan', idfile='buy_cate.id', drop=True),

        #历史贷款总额
        StatFeature(prefix='loan_sum_before',startid=1,name='loan_sum_before', expand=False) ,
        # 对数值 求和
        # StatFeature(prefix='loan_sumlog_before', startid=1, name='loan_sumlog_before', expand=False),
    #    StatFeature(prefix='loan_weight_sum_before', startid=1, name='loan_weight_sum_before', expand=False), #加权重的贷款总额

#        StatFeature(prefix='loan_weight_sum_before_0', startid=1, name='loan_weight_sum_before_0', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_1', startid=1, name='loan_weight_sum_before_1', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_2', startid=1, name='loan_weight_sum_before_2', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_3', startid=1, name='loan_weight_sum_before_3', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_4', startid=1, name='loan_weight_sum_before_4', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_5', startid=1, name='loan_weight_sum_before_5', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_6', startid=1, name='loan_weight_sum_before_6', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_7', startid=1, name='loan_weight_sum_before_7', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_8', startid=1, name='loan_weight_sum_before_8', expand=False), #加权重的贷款总额
#        StatFeature(prefix='loan_weight_sum_before_9', startid=1, name='loan_weight_sum_before_9', expand=False), #加权重的贷款总额
        # 加权重的贷款总额
        #历史贷款均值
        StatFeature(prefix='loan_avg_before', startid=1, name='loan_avg_before', expand=False),
        # StatFeature(prefix='loan_avglog_before', startid=1, name='loan_avglog_before', expand=False),
        StatFeature(prefix='loan_median_before', startid=1, name='loan_median_before', expand=False),
        StatFeature(prefix='loan_avg_std_before', startid=1, name='loan_avg_std_before', expand=False),

        ##NOTE: todo  half_std  and double_std 与  avg_std 高度相关，加入特征后 并没有明显提升  ,后续可以考虑放弃 plan
        StatFeature(prefix='loan_avg_half_std_before', startid=1, name='loan_avg_half_std_before', expand=False),
        StatFeature(prefix='loan_avg_double_std_before', startid=1, name='loan_avg_double_std_before', expand=False),

        StatFeature(prefix='loan_skew_before', startid=1, name='loan_skew_before', expand=False),
        StatFeature(prefix='loan_kurt_before', startid=1, name='loan_kurt_before', expand=False),

        # StatFeature(prefix='loan_skewlog_before', startid=1, name='loan_skewlog_before', expand=False),
        # StatFeature(prefix='loan_kurtlog_before', startid=1, name='loan_kurtlog_before', expand=False),


        StatFeature(prefix='loan_max_before', startid=1, name='loan_max_before', expand=False),
        StatFeature(prefix='loan_min_before', startid=1, name='loan_min_before', expand=False),

        StatFeature(prefix='loan_mad_before', startid=1, name='loan_mad_before', expand=False),
        # StatFeature(prefix='loan_madlog_before', startid=1, name='loan_madlog_before', expand=False),

        #StatFeature(prefix='loan_diff_before', startid=1, name='loan_diff_before', expand=False), # sum 1st diff will be last - first .
        #
        #上个月贷款总额
        StatFeature(prefix='loan_sum_previous',startid=1,name='loan_sum_previous', expand=False) ,
        StatFeature(prefix='loan_avg_previous', startid=1, name='loan_avg_previous', expand=False),
        StatFeature(prefix='loan_mad_previous', startid=1, name='loan_mad_previous', expand=False),

        StatFeature(prefix='loan_min_previous', startid=1, name='loan_min_previous', expand=False),
        StatFeature(prefix='loan_max_previous', startid=1, name='loan_max_previous', expand=False),
        StatFeature(prefix='loan_median_previous', startid=1, name='loan_median_previous', expand=False),

        StatFeature(prefix='loan_skew_previous', startid=1, name='loan_skew_previous', expand=False),
        StatFeature(prefix='loan_kurt_previous', startid=1, name='loan_kurt_previous', expand=False),



        # StatFeature(prefix='loan_sumlog_previous', startid=1, name='loan_sumlog_previous', expand=False),
        StatFeature(prefix='loan_sum_previous2', startid=1, name='loan_sum_previous2', expand=False),
        # StatFeature(prefix='loan_sumlog_previous2', startid=1, name='loan_sumlog_previous2', expand=False),
        #截止当前的贷款余额 ，通过 贷款额度 + 分期数 推算
        StatFeature(prefix='loan_balance', startid=1, name='loan_balance', expand=False),

        #balance vs initial limit.
        StatFeature(prefix='loan_left_limit', startid=1, name='loan_left_limit', expand=False),
        #  step1:  用最大值近似  近似以后 效果降低了 .
        #  step2:  采用时间和贷款次数近似 .
        #  step3:  采用  left_balance/limit , 效果稳定了，挺好.
    #    StatFeature(prefix='loan_left_magic_limit', startid=1, name='loan_left_magic_limit', expand=False),

        # max/limit
        StatFeature(prefix='loan_max_rate', startid=1, name='loan_max_rate', expand=False),

        # 上个月的 left_limit情况
        # StatFeature(prefix='loan_previous_left_limit',startid=1, name='loan_previous_left_limit', expand=False),




        #每期的平均额度
        # NOTE:没有提升成绩，还是1.79 ， 后期放弃
        StatFeature(prefix='loan_perplannum_avg_before', startid=1, name='loan_perplannum_avg_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_max_before', startid=1, name='loan_perplannum_max_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_min_before', startid=1, name='loan_perplannum_min_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_mad_before', startid=1, name='loan_perplannum_mad_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_skew_before', startid=1, name='loan_perplannum_skew_before', expand=False,drop=False),
        StatFeature(prefix='loan_perplannum_kurt_before', startid=1, name='loan_perplannum_kurt_before', expand=False,drop=False),

        #剩余期数
        # NOTE:新增的planum_avg max,min提升成绩

        StatFeature(prefix='loan_plannum_sum_before', startid=1, name='loan_plannum_sum_before', expand=False,drop=False),
        StatFeature(prefix='loan_plannum_avg_before',startid=1,name='loan_plannum_avg_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_max_before', startid=1, name='loan_plannum_max_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_min_before', startid=1, name='loan_plannum_min_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_mad_before', startid=1, name='loan_plannum_mad_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_skew_before', startid=1, name='loan_plannum_skew_before', expand=False, drop=False),
        StatFeature(prefix='loan_plannum_kurt_before', startid=1, name='loan_plannum_kurt_before', expand=False, drop=False),


        StatFeature(prefix='loan_balance_plannum', startid=1, name='loan_balance_plannum', expand=False),
        #总的贷款次数
        StatFeature(prefix='loan_count_before', startid=1, name='loan_count_before', expand=False),
        StatFeature(prefix='loan_count_previous', startid=1, name='loan_count_previous', expand=False),



        #点击某商品对应的平均贷款额度
        # 特征重要度低
        StatFeature(prefix='click_avg_loan', startid=1,  expand=True,  name='click_avg_loan',idfile='click_pid.id', drop=True) ,
        StatFeature(prefix='click_param_avg_loan', startid=1,  expand=True, name='click_param_avg_loan', idfile='click_pid_param.id' ,drop=True),

    ])

================================================================================================================
以下是 Feature 和 StatFeature的定义 ，可以方便的添加特征； 也可以按照类似结构扩展出自己的Feature类
import re,logging
import numpy as  np

class Feature(object):
    def __init__(self, prefix, startid, **kwargs ):
        """
        这里的Feature 代指 特征类别
        特征空间要素： 覆盖范围 特征含义

        统计要素单独处理

        处理场景: 普通特征+ 交叉特征
        """
        self.start_feaid , self.end_feaid = startid, startid
        self.idmap = {}
        self.valmap = {}  #value map
        self.prefix = prefix

        self.kwargs = kwargs
        if 'drop' in kwargs:
            self.drop = kwargs['drop']
        else:
            self.drop = False
        pass
    def name(self):
        """
        特征含义
        :return:
        """
        return self.kwargs['name']

    def getIdMap(self):
        return self.idmap
    def coverRange(self):
        """
        feature id 的覆盖范围
        [s,e)
        :return:
        """
        return "[{0} , {1})  {2}".format( self.start_feaid,self.end_feaid , self.kwargs['name'])
    def alignFeatureID(self,start):

        self.start_feaid = start
        self.end_feaid  = start + len(self.idmap)
        return self.end_feaid
    def transform(self, prefix, feastr,sep =':', val = 1 ):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}
        :param feastr:
        :return  -1 skip the feastr.
        """

        # prefix, feaval = re.split(sep, feastr, maxsplit=1)
        if prefix != self.prefix or self.drop == True:
            return -1


        if feastr in self.idmap:
            return "{0}:{1}".format(self.start_feaid +  self.idmap[feastr] - 1 , val )
        else:
            return -1

    def tryAdd(self, prefix,  feastr, sep=':'):
        """

        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix , feaval = re.split(sep, feastr, maxsplit=2)
        if self.drop: return False
        if prefix == self.prefix:
            if feastr in self.idmap:
                pass
            else:
                self.idmap[feastr] =len(self.idmap) + 1

            return True
        else:
            return False


class StatFeature(Feature):

    def __init__(self, prefix , startid,expand =False, **kwargs  ):
        """
        统计特征
        单维度统计特征+ 交叉统计特征 ;

        只会是一种情况的统计特征  , 多维度统计的另外考虑

        NOTE: 保证测试集 和 训练集中的ID 一致
        :param prefix:
        :param start_id:
        :param expand: True - 在特征级别 统计以后，再编码成 统计特征
        """


        start_id = startid
        self.start_feaid = start_id
        self.end_feaid = start_id +1

        self.idmap = {}
        self.expand = expand
        self.valmap = {}  # value map
        self.prefix = prefix
        self.kwargs= kwargs
        if 'name' in kwargs:
            self.kwargs['name'] = 'stat_{0}'.format(self.kwargs['name'])
        if 'drop' in kwargs:
            self.drop = kwargs['drop']
        else:
            self.drop = False

        if 'default' in kwargs:
            self.default = kwargs['default']
        else:
            self.default = None

        if not self.drop and self.expand  and  'idfile' in self.kwargs:

            self.loadIdMap_(self.kwargs['idfile'])


        pass

    def name(self):
        """
        特征含义
        :return:
        """
        return self.kwargs['name']
    def coverFeaId(self,feaid):
        if not self.drop and feaid < self.end_feaid and feaid >= self.start_feaid:
            return True
        return False
    def alignFeatureID(self, start):
        self.start_feaid = start
        if self.expand:
            self.end_feaid = start + len(self.idmap)
        else:
            self.end_feaid = start +1
        return self.end_feaid
    def transform(self, prefix, feastr ,sep=":"):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}
        :param feastr:
        :return  -1 skip the feastr.
        """
        # prefix, feaval = re.split(sep, feastr, maxsplit=1)
        if prefix != self.prefix or self.drop:
            return -1

        if feastr in self.valmap:
            if self.expand:
                month , feaval = re.split(sep,feastr,maxsplit=1)
                if feaval in self.idmap:
                    return "{0}:{1}".format(self.idmap[feaval] + self.start_feaid -1 , self.valmap[feastr])
                else:
                    return -1

            else:
                return "{0}:{1}".format(self.start_feaid ,  self.valmap[feastr])
        else:
            if self.default is not None and self.expand == False:
                return "{0}:{1}".format(self.start_feaid, self.default)
            else:
                return -1
    def loadIdMap_(self,file):
        """
        加载一个idmap，保证训练集和测试集是一致的 feature id
        :return:
        """
        with open(file ,'r') as f:
            for L in f:
                feaval = L.strip()
                prefix, val = feaval.split(':')
                if prefix == self.prefix and  feaval not in self.idmap:
                    self.idmap[feaval] = len(self.idmap) + 1
        logging.info('done loadidmap '+ file )



    def tryAdd(self, prefix, feastr, val , sep=':'):
        """

        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix, feaval = re.split(sep, feastr, maxsplit=2)
        if self.drop: return False
        if prefix == self.prefix:
            if feastr in self.valmap:
                pass
            else:
                self.valmap[feastr] = val
                if self.expand:
                    month, feaval = re.split(sep, feastr, maxsplit=1)

                    if feaval not in self.idmap:
                        self.idmap[feaval] = len(self.idmap) +1
            return True
        else:
            return False


class SuperStatFeature(Feature):
    def __init__(self, prefix, startid, cnt , **kwargs):
        """
        统计特征
        其中某个维度是 数组的情况

        Top-n Feature
        """

        start_id = startid
        self.start_feaid = start_id
        self.end_feaid = start_id + cnt

        self.valmap = {}  # value map
        self.prefix = prefix
        self.cnt =  cnt
        self.kwargs = kwargs

        pass

    def name(self):
        """
        特征含义
        :return:
        """
        return "name"


    def alignFeatureID(self, start):

        self.start_feaid = start
        self.end_feaid = start + self.cnt
        return self.end_feaid


    def transform(self, prefix, feastr_list, sep=":"):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}

        选统计值Top-cnt的作为特征 ，保持特征空间可控


        :param feastr:
        :return  -1 skip the feastr.
        """

        if type(feastr_list) != list:
            return -1
        if prefix != self.prefix:
            return -1
        buf = []
        for feastr in feastr_list:
            if feastr in self.valmap:
                buf.append( float(self.valmap[feastr]))

        buf = sorted(buf, reverse=True)

        strbuf = []

        i = 0
        for v in buf[:self.cnt]:
            strbuf.append("{0}:{1}".format( self.start_feaid + i , v  ))
            i +=1

        if len(strbuf) == 0:
            return -1
        else:
            return ' '.join(strbuf)


    def tryAdd(self, prefix,  feastr, val, sep=':'):
        """
        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix, feaval = re.split(sep, feastr, maxsplit=2)

        if prefix == self.prefix:
            if feastr in self.valmap:
                pass
            else:
                self.valmap[feastr] = val
            return True
        else:
            return False


