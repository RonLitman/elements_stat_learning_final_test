import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from catboost import CatBoostRegressor
import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
import xgboost as xgb
from sklearn import ensemble


def lin_model(x_train, y_train,alpha=0.006, method='Lasso'):
    '''
    fit a LinearRegression
    :param x_train:
    :param y_train:
    :return:
    '''

    if method == 'normal':
        clf_reg = linear_model.LinearRegression()
    if method == 'Lasso':
        clf_reg = linear_model.Lasso(alpha=alpha)
    if method == 'Ridge':
        clf_reg = linear_model.Ridge(alpha=alpha)

    clf_reg.fit(x_train, y_train)
    return clf_reg

def run_catbost(x_train, x_dev, y_train, y_dev):
    '''

    :param x_train:
    :param x_dev:
    :param y_train:
    :param y_dev:
    :return:
    '''
    print('Training a CatBoost Regressor model')
    cb_model = CatBoostRegressor(iterations=500,
                                 learning_rate=0.05,
                                 depth=10,
                                 eval_metric='RMSE',
                                 random_seed=42,
                                 bagging_temperature=0.2,
                                 od_type='Iter',
                                 metric_period=50,
                                 od_wait=20)

    cb_model.fit(x_train, y_train,
                 c=(x_dev, y_dev),
                 use_best_model=True,
                 verbose=True)
    # pred_test_cat = np.expm1(cb_model.predict(df_test))

    return cb_model

def run_lgb(x_train, x_dev, y_train, y_dev):
    '''

    :param x_train:
    :param x_dev:
    :param y_train:
    :param y_dev:
    :return:
    '''
    print('Training a lightgbm model')
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 40,
        "learning_rate": 0.005,
        "bagging_fraction": 0.6,
        "feature_fraction": 0.6,
        "bagging_frequency": 1,
        "verbosity": -1
    }
    print('Params that are being used for training are: {}'.format(params))
    lgtrain = lgb.Dataset(x_train, label=y_train.values.ravel())
    lgval = lgb.Dataset(x_dev, label=y_dev.values.ravel())
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000,
                      valid_sets=[lgtrain, lgval],
                      early_stopping_rounds=500,
                      verbose_eval=150,
                      evals_result=evals_result)

    # pred_test_y = np.expm1(model.predict(df_test, num_iteration=model.best_iteration))
    return model

def run_xgb(x_train, x_dev, y_train, y_dev):
    print('Training a xgb model')
    params = {'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'eta': 0.001,
              'max_depth': 40,
              'subsample': 0.6,
              'colsample_bytree': 0.6,
              'alpha': 0.001,
              'silent': True}

    tr_data = xgb.DMatrix(x_train, y_train)
    va_data = xgb.DMatrix(x_dev, y_dev)

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds=500, verbose_eval=1)

    return model_xgb

def run_gb(x_train, y_train):
    print('Training a GradientBoostingRegressor model')

    params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls', 'max_features': 'sqrt'}

    print('Params that are being used for training are: {}'.format(params))
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(x_train, y_train)

    return clf

def knn_cosine(x_train, x_test, y_train, y_test, k=50):
    x_train_temp = x_train.reset_index(drop=True)
    x_dev_temp = x_test.reset_index(drop=True)
    y_train_temp = y_train.reset_index(drop=True)
    y_dev_temp = y_test.reset_index(drop=True)

    preds_train = np.zeros((y_train_temp.shape[0], 1))
    similarity_dist = 1 - cosine_similarity(x_train_temp, x_train_temp)
    for i in range(similarity_dist.shape[1]):
        min_ind_dev = similarity_dist[:, i].argsort()[:int(k)]
        preds_train[i] = np.mean(y_train_temp[y_train_temp.index.isin(min_ind_dev)])
    mse_train_reg = math.sqrt(mean_squared_error(y_train_temp, preds_train))

    preds_dev = np.zeros((y_dev_temp.shape[0],1))
    similarity_dist = 1 - cosine_similarity(x_train_temp, x_dev_temp)
    for i in range(similarity_dist.shape[1]):
        min_ind_dev = similarity_dist[:,i].argsort()[:int(k)]
        preds_dev[i] = np.mean(y_train_temp[y_train_temp.index.isin(min_ind_dev)])
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev_temp, preds_dev))
    return mse_train_reg, mse_dev_reg

def knn_mahalanobis(x_train, x_test, y_train, y_test, k=50):
    dist = DistanceMetric.get_metric('manhattan')
    x_train_temp = x_train.reset_index(drop=True)
    x_dev_temp = x_test.reset_index(drop=True)
    y_train_temp = y_train.reset_index(drop=True)
    y_dev_temp = y_test.reset_index(drop=True)

    preds_train = np.zeros((y_train_temp.shape[0], 1))
    similarity_dist = dist.pairwise(x_train_temp, x_train_temp)
    for i in range(similarity_dist.shape[1]):
        min_ind_dev = similarity_dist[:, i].argsort()[:int(k)]
        preds_train[i] = np.mean(y_train_temp[y_train_temp.index.isin(min_ind_dev)])

    mse_train_reg = math.sqrt(mean_squared_error(y_train_temp, preds_train))

    preds_dev = np.zeros((y_dev_temp.shape[0],1))
    similarity_dist = dist.pairwise(x_train_temp, x_dev_temp)
    for i in range(similarity_dist.shape[1]):
        min_ind_dev = similarity_dist[:,i].argsort()[:int(k)]
        preds_dev[i] = np.mean(y_train_temp[y_train_temp.index.isin(min_ind_dev)])

    mse_dev_reg = math.sqrt(mean_squared_error(y_dev_temp, preds_dev))
    return mse_train_reg, mse_dev_reg


def knn(x_train, x_test, y_train, y_test, k=500, metric='cosine'):
    clf = KNeighborsRegressor(n_neighbors=k, metric=metric)
    clf.fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    pred_dev = clf.predict(x_test)
    rmse_train = math.sqrt(mean_squared_error(y_train, pred_train))
    rmse_dev = math.sqrt(mean_squared_error(y_test, pred_dev))
    return rmse_train, rmse_dev