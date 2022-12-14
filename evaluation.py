import numpy as np
import math
import heapq
import pandas as pd
from sklearn.metrics import *

def erase(score,train_dict,user_index_dict,item_index_dict):
    score = np.array(score)
    for user in train_dict:
        userno = user_index_dict[user]
        for item in train_dict[user]:
            itemno = item_index_dict[item]
            score[userno, itemno] = -100
    return score


def averauc(score,label):
    auc = 0
    counter = 0
    for i in range(score.shape[0]):
        user_score = score[i].tolist()
        user_label = label[i].tolist()
        try:
            auc += roc_auc_score(user_label, user_score)
        except ValueError:
            counter += 1
            pass
    return auc / (score.shape[0] - counter)


def mapk(score,label,k):
    counter = 0
    apk = 0
    for user_no in range(score.shape[0]):
        user_score = score[user_no].tolist()
        user_label = label[user_no].tolist()
        label_count = sum(user_label)
        if label_count == 0:
            counter += 1
            continue
        else:
            topk_score = sorted(user_score, reverse=True)[:k]  # top-k 分数
            index = [user_score.index(i) for i in topk_score]
            topk_label = [user_label[i] for i in index]  # top-k 推荐所对应的标签 [0,1,1,0]
            presicion_k = [sum(topk_label[:i]) / i for i in range(1, len(topk_label) + 1) if
                           topk_label[i - 1] == 1]  # 每个位置对应的prcision值
            pre = [max(presicion_k[i:]) for i in range(len(presicion_k))]
            try:
                apk += sum(pre) / sum(topk_label)
            except ZeroDivisionError:
                apk += 0
    return apk / (score.shape[0] - counter)


def topk(score,label, k):
    '''
    :param score: prediction
    :param k: number of top-k
    '''
    evaluation = [0, 0, 0, 0]
    counter = 0
    discountlist = [1 / math.log(i + 1, 2) for i in range(1, k + 1)]

    for user_no in range(score.shape[0]):
        user_score = score[user_no].tolist()
        user_label = label[user_no].tolist()
        label_count = int(sum(user_label))
        topn_recommend_score = heapq.nlargest(k, user_score)
        topn_recommend_index = [user_score.index(i) for i in
                                topn_recommend_score]
        topn_recommend_label = [user_label[i] for i in topn_recommend_index]
        idcg = discountlist[0:label_count]

        if label_count == 0:
            counter += 1
            continue
        else:
            topk_label = topn_recommend_label[0:k]
            true_positive = sum(topk_label)
            evaluation[0] += true_positive / k  # precision
            evaluation[1] += true_positive / label_count  # recall
            evaluation[2] += 2 * true_positive / (k + label_count)  # f1
            evaluation[3] += np.dot(topk_label, discountlist[0:]) / sum(idcg)  # ndcg
    return [i / (score.shape[0] - counter) for i in evaluation]
