'''
@wangtt20200804
建立用户物品的倒排表,主要为了统计同一物品有多少用户感兴趣
针对每一个用户,得到两两物品的共现矩阵
累加所有共现矩阵,并进行归一化
然后计算物品相似度
'''
# In[]
import math
from collections import defaultdict
from operator import itemgetter
def ItemSimilarity(train):
    # 计算共现矩阵和倒排表
    C = dict()
    N = dict()
    for user, items in train.items():
        for u in items:
            if u not in N:
                N[u] = set()
            N[u].add(user)
            C.setdefault(u, defaultdict(int))
            for v in items:
                N[u].add(user)
                if u == v:
                    continue
                C[u][v] += 1
        
    #计算物品相似度
    W = dict()
    for i, related_items in C.items():
        W.setdefault(i, defaultdict(float))
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(len(N[i])*len(N[j]))
    return W

def Recommend(user, train, W, K, N):
    rank = dict()
    rui = 1
    nu = train[user]
    for i in nu:
        for j, wij in sorted(W[i].items(), key = itemgetter(1), reverse = True)[0:K]:
            if j in nu: #已经给用户推荐过该物品了
                continue
            rank.setdefault(j, 0)
            rank[j] += wij * rui

    return sorted(rank.items(), key = itemgetter(1), reverse = True)[0:N]

if __name__ == '__main__':
    data = {'A':{'a', 'b', 'd'}, 'B':{'b','c','e'}, 'C':{'c','d'}, 'D':{'b','c','d'},'E':{'a','d'}}
    W = ItemSimilarity(data)
    res = Recommend('A', data, W, 3, 2)
    print(W)
