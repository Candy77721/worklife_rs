# In[]
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split

data = pd.read_csv('')
data = data.sample(frac = 0.1, random_state = 10)   #从dataframe中随机选取行 axis: 0或”行”和1或”列”

data.shape # data.sample查看数据的行和列
data.dropna(inplace = True)

# %%xgb
import xgboost as xgb   
from sklearn.model_selection import GridSearchCV
from sklearn import metrics   

def auc(model, train, test):
    return (metrics.roc_auc_score(y_train, model.predict_proba(train)[:, 1]),
           metrics.roc_auc_score(y_test, model.predict_proba(test)[:, 1]))

model = xgb.XGBClassifier()
param_dist = {
    'max_depth' : [10, 30, 50],
    'min_child_weight': [1],
    'n_estimators': [200],
    'learning_rate': [0.16]
}

grid_search = GridSearchCV(model, param_grid = param_dist, cv = 3, verbose = 10, n_jobs = -1)
grid_search.fit(train, y_train)

grid_search.best_estimator_

model = xgb.XGBClassifier(
    max_depth = 10, 
    min_child_weight = 1,
    n_estimators = 200,
    n_jobs = -1,
    verbose = 1,
    learning_rate = 0.16
)

model.fit(train, y_train)  # grid_search.fit()
auc(model, train, test)

# In[] lgb
import lightgbm as lgb        
from sklearn import metrics  

def auc2(model, train, test):
    return (metrics.roc_auc_score(y_train, model.predict(train)),
        metrics.roc_auc_score(y_test, model.predict(test)))

lg = lgb.LGBMClassifier(silent = False) # 训练过程是否打印日志信息
param_dist = {
    'max_depth': [25, 50, 75],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [300, 900, 1200],
    'n_estimators':[200]
}

grid_search = GridSearchCV(lg, n_jobs = -1, param_grid = param_dist, cv = 3, scoring = 'roc_auc', verbose = 5)
grid_search.fit(train, y_train)
grid_search.best_estimator_

d_train = lgb.Dataset(train, label = y_train)
params = {
    'max_depth': 50,
    'learning_rate': 0.1,
    'num_leaves': 900,
    'n_estimators': 300
}

model2 = lgb.train(params, d_train)
# model2 = lgb.train(params, d_train, categorical_feature = cate_features_name)
auc2(model2, train, test)


# In[] catboost
import catboost as cb  
cat_features_index = [0,1,2,3,4,5,6]

def auc(model, train, test):
    return (metrics.roc_auc_score(y_train, model.predict_proba(train)[:, 1]),
             metrics.roc_auc_score(y_test, mdodel.predict_proba(test)[:, 1]))

params = {
    'depth':[4, 7, 10],
    'learning_rate': [0.03, 0.1, 0.15],
    'l2_leaf_reg':[1, 4, 9],
    'iterations': [300]
}
cb = cb.CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring = 'roc_auc', cv = 3)
cb_model.fit(train, y_train)

clf = cb.CatBoostClassifier(eval_metrics = 'AUC', depth = 10, iterations = 500, l2_leaf_reg = 9, learning_rate = 0.15)
# %timeit clf.fit(train, y_train)