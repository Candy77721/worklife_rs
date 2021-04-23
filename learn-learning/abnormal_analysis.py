# coding:utf-8
# In[]
from __future__ import division
import pandas as pd

import os
import numpy as np
import scipy.stats as st
def abnormal_detection_gaussian(X_train, x_test, threshold):
    '''
    X_train: M * N
    x_test: 1 * N
    '''
    M = len(X_train)
    N = len(X_train.iloc[0])

    mu = []
    sigma2 = []
    # 每一维特征均值和方差的估计值
    mu = np.mean(X_train, axis = 0)
    _sigma2 = (X_train - mu) ** 2
    sigma2 = _sigma2.sum() / M
    
    p = 1
    for j in range(N):
        p_j = st.norm.pdf(x_test.loc[:,j], loc = mu[j], scale = sigma2 ** 0.5)
        p = p * p_j 
    
    if p < threshold:
        return True
    return False

# In[]
import shutil
def init_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

class WMA(object):
    """
    加权移动平均实现
    """

    @staticmethod
    def get_wma_weights(span, flag=True):
        """
        计算每个数值点的wma权重值
        """
        paras = range(1, span + 1)
        count = sum(paras)
        if flag:
            return [float(para) / count for para in paras]
        else:
            return [float(para) / count for para in paras][::-1]

    def get_wma_values(self, datas):
        """
        计算wma数值
        """
        wma_values = []
        wma_keys = datas.index
        for length in range(1, len(datas) + 1):
            wma_value = 0
            weights = self.get_wma_weights(length)
            for index, weight in zip(datas.index, weights):
                wma_value += datas[index] * weight
            wma_values.append(wma_value)
        return pd.Series(wma_values, wma_keys)


def calculate_variance(dps, moving_average):
    variance = 0
    flag_list = moving_average.isnull()
    count = 0
    for index in range(len(dps)):
        if flag_list[index]:
            count += 1
            continue
        variance += (dps[index] - moving_average[index]) ** 2
    variance /= (len(dps) - count)
    return variance

if __name__ == '__main__':
    '''
    1. 首先构造测试数据
    '''
    # TODO 以后改成读取数据,直接生成 pandas Series 对象
    data_dict_series = {
        '1490707920': 19.8219660272,
        '1490707980': 20.0681534509,
        '1490708040': 20.1842385903,
        '1490708100': 19.7650368611,
        '1490708160': 19.9269200861,
        '1490708220': 18.470530654,
        '1490708280': 18.2211077462,
        '1490708340': 19.3140179366,
        '1490708400': 20.5632611228,
        '1490708460': 20.6341463476,
        '1490708520 ': 8.9789650937,
        '1490708580': 19.3494873891,
        '1490708640': 20.0160623292,
        '1490708700': 19.6702364186,
        '1490708760': 19.624609674,
        '1490708820': 20.456564799,
        '1490708880': 17.6035744548,
        '1490708940': 17.1818237007,
        '1490709000': 20.600406309,
        '1490709060': 19.9717502842,
        '1490709120': 20.0490164061,
        '1490709180': 20.1280070096,
        '1490709240': 20.579920421,
        '1490709300': 19.0682847972,
        '1490709360': 20.6150588592,
        '1490709420': 19.8059894244,
        '1490709480': 19.7459333887,
        '1490709540': 20.1206678947,
        '1490709600': 21.0094336718,
        '1490709660': 19.8375952393
    }
    dps = pd.Series(data_dict_series)

    '''
    2. ewma进行拟合
    '''
    ewma_line = pd.ewma(dps, span=4)

    '''
    3. 简单移动平均
    '''
    sma_line = pd.rolling_mean(dps, window=4)

    '''
    4. wma加权移动平均
    '''
    wma_line = WMA().get_wma_values(dps)

    '''
    5. 计算告警
    '''
    # 计算每种平滑下原始数据的标准差

    sma_var = calculate_variance(dps, sma_line)
    wma_var = calculate_variance(dps, wma_line)
    ewma_var = calculate_variance(dps, ewma_line)

    flag_list = sma_line.isnull()

    for index in sma_line.index:
        if not (sma_line[index] - sma_var <= dps[index] <= sma_line[index] + sma_var):
            if not flag_list[index]:
                print("异常点", dps[index])
    print("==================================")
    for index in wma_line.index:
        if not (wma_line[index] - wma_var <= dps[index] <= wma_line[index] + wma_var):
            print ("异常点", dps[index])
    print("==================================")
    for index in ewma_line.index:
        if not (ewma_line[index] - ewma_var <= dps[index] <= ewma_line[index] + ewma_var):
            print ("异常点", dps[index])
