import numpy as np
import torch
from scipy.ndimage import maximum_filter
from torch import Tensor
import torch.nn.functional as F

def prep_clf(obs, pre, threshold=0.1):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    if isinstance(obs, Tensor):
        obs = torch.where(obs >= threshold, 1, 0)
        pre = torch.where(pre >= threshold, 1, 0)

        # True positive (TP)
        hits = torch.sum((obs == 1) & (pre == 1))

        # False negative (FN)
        misses = torch.sum((obs == 1) & (pre == 0))

        # False positive (FP)
        falsealarms = torch.sum((obs == 0) & (pre == 1))

        # True negative (TN)
        correctnegatives = torch.sum((obs == 0) & (pre == 0))
    else:
        obs = np.where(obs >= threshold, 1, 0)
        pre = np.where(pre >= threshold, 1, 0)

        # True positive (TP)
        hits = np.sum((obs == 1) & (pre == 1))

        # False negative (FN)
        misses = np.sum((obs == 1) & (pre == 0))

        # False positive (FP)
        falsealarms = np.sum((obs == 0) & (pre == 1))

        # True negative (TN)
        correctnegatives = np.sum((obs == 0) & (pre == 0))


    return hits, misses, falsealarms, correctnegatives


def precision(obs, pre, threshold=0.1):
    '''
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FP + 1e-10)  # 预测对的降水在所有预测为降水样本的比例

def recall(obs, pre, threshold=0.1):
    '''
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FN + 1e-10)  # POD  预测对的降水在所有本就为降水样本的比例


def ACC(obs, pre, threshold=0.1):
    '''
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN + 1e-10)


def FSC(obs, pre, threshold=0.1):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score + 1e-10))

def POD(obs, pre, threshold=0.1):
    '''
    TP/(TP+FP)
    func : 计算命中率 hits / (hits + misses)
    pod - Probability of Detection
    Args:
        obs  : observations
        pre  : prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: PDO value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return hits / (hits + misses + 1e-10)

def FAR(obs, pre, threshold=0.1):
    '''
    FN / （TP+FN）
    func: 计算误警率。falsealarms / (hits + falsealarms)
    FAR - false alarm rate
    Args:
        obs  : observations
        pre  : prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms + 1e-10)



def CSI(obs, pre, threshold=0.1):
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses)
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return hits / (hits + falsealarms + misses + 1e-10)


def csi_neighborhood(obs, pre, kappa, threshold=0.1):
    """
    计算CSI-Neighborhood指标。

    参数:
    predictions: 预测结果的二值数组（0或1），形状为 (height, width)
    labels: 真实标签的二值数组（0或1），形状为 (height, width)
    kappa: 最大池化的核大小（窗口大小）

    返回:
    CSI-Neighborhood指标
    """
    # 应用最大池化
    pooled_predictions = F.max_pool2d(pre, kernel_size=kappa)
    pooled_labels = F.max_pool2d(obs, kernel_size=kappa)

    pooled_labels = torch.where(pooled_labels >= threshold, 1, 0)
    pooled_predictions = torch.where(pooled_predictions >= threshold, 1, 0)

    # 计算命中、漏报和误报
    hits = np.sum((pooled_predictions == 1) & (pooled_labels == 1))
    misses = np.sum((pooled_predictions == 0) & (pooled_labels == 1))
    false_alarms = np.sum((pooled_predictions == 1) & (pooled_labels == 0))

    # 计算CSI
    csi = hits / (hits + misses + false_alarms + 1e-10)  # 添加小数以防除零错误

    return csi