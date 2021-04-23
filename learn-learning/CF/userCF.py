'''
两种方法，第二种方法去掉对结果为0的计算过程，时间复杂度较低

先计算物品和用户的倒排表
再计算两两用户之间共同感兴趣的物品个数有多少个
再计算两两用户之间的相似度
最后计算用户u对物品i的感兴趣程度
'''

import math
from collections import defaultdict
from operator import itemgetter

# In[]
def UserSimilarity_0(train):
    # W = [ [0 for i in range(len(train))] for i in range(len(train))]
    W = dict()
    for u in train.keys():
        W[u] = {}
        for v in train.keys():
            if u == v:
                continue
            W[u][v] = len(train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

#计算用户u对物品i的感兴趣程度
def Recommend_0(user, train, W, K, N):
    rank = dict()
    rvi = 1 # 用户v对物品i的感兴趣程度，因为使用的是单一行为的隐反馈数据，所以rvi取1
    interated_items = train[user]
    #0 按Key排序，1按value的第一项排序, reverse = True 由大到小
    for v, wuv in sorted(W[user].items(), key = itemgetter(1), reverse = True)[0:K]:
        for item in train[v]:
            if item in interated_items:
                continue
            rank.setdefault(item, 0)
            rank[item] += wuv * rvi
     # 根据得分最后为所有的物品排序
    # return rank
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

def UserSimilarity(train):
    #计算物品和用户的倒排表
    item_users = dict()
    for u, items in train.items():
        for item in items:
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(u)
    
    #计算两两用户之间共同感兴趣的物品个数有多少个，C矩阵,方法：扫描倒排表
    C = dict()
    for i, user in item_users.items():
        for u in user:
            ## 当需要将某个键映射到一个集合类型时,有时候需要初始化集合类型
            C.setdefault(u, defaultdict(int))
            for v in user:
                if u == v:
                    continue
                C[u][v] += 1

    #计算两两用户之间的相似度
    W = dict()
    for u, related_users in C.items():
        W.setdefault(u, defaultdict(float))
        for v, count in related_users.items():
            W[u][v] = count/math.sqrt(len(train[u])*len(train[v]))

    return W

#计算用户u对物品i的感兴趣程度
def Recommend(user, train, W, K, N):
    rank = dict()
    rvi = 1 # 用户v对物品i的感兴趣程度，因为使用的是单一行为的隐反馈数据，所以rvi取1
    interated_items = train[user]
    for v, wuv in sorted(W[user].items(), key = itemgetter(1), reverse = True)[0:K]:
        for item in train[v]:
            if item in interated_items:
                continue
            rank.setdefault(item, 0)
            rank[item] += wuv * rvi
    # 根据得分最后为所有的物品排序
    # return rank
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

if __name__ == "__main__":
    data = {'A':{'a','b','d'}, 'B':{'a','c'}, 'C':{'b','e'}, 'D':{'c','d','e'}}
    # l = len(data[0] & data[1])
    W = UserSimilarity(data)
    res = Recommend('A', data, W, 3, 2)
    print(res)   
