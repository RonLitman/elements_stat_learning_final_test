import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
from sklearn.model_selection import train_test_split
import models as models
from sklearn.metrics import mean_squared_error
import math
from sklearn.neighbors import KNeighborsClassifier
import more_itertools
from sklearn.cross_validation import KFold
from sklearn import linear_model


def load_and_set_data():
    '''
    loads and set the data
    :return:
    '''
    train = {}
    test = {}
    path = '/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Final Test/data/'
    movie_titles = pd.read_csv(path + 'movie_titles.txt', sep=",", header=None)

    train['train_y_rating'] = pd.read_csv(path + 'train_y_rating.txt', delimiter=r"\s+", header=None)
    train['train_y_date'] = pd.read_csv(path + 'train_y_date.txt', delimiter=r"\s+", header=None)
    train['train_ratings_all'] = pd.read_csv(path + 'train_ratings_all.txt', delimiter=r"\s+", header=None)
    train['train_dates_all'] = pd.read_csv(path + 'train_dates_all.txt', delimiter=r"\s+", header=None)

    test['test_y_date'] = pd.read_csv(path + 'test_y_date.txt', delimiter=r"\s+", header=None)
    test['test_ratings_all'] = pd.read_csv(path + 'test_ratings_all.txt', delimiter=r"\s+", header=None)
    test['test_dates_all'] = pd.read_csv(path + 'test_dates_all.txt', delimiter=r"\s+", header=None)

    train['train_ratings_all'].columns = movie_titles.iloc[:, ]
    test['test_ratings_all'].columns = movie_titles.iloc[:, ]

    return movie_titles, train, test


def set_df(train, test):
    """
    remove users with variance lower then the threshold (in ratings)
    :param test:
    :param train:
    :return:
    """
    ind_missing_train = ((train['train_ratings_all'] == 0) * 1).astype('int64', copy=False)
    ind_missing_train.columns = [str(col[1]) + '_indicator' for col in ind_missing_train.columns]
    ind_missing_test = ((test['test_ratings_all'] == 0) * 1).astype('int64', copy=False)
    ind_missing_test.columns = [str(col) + '_indicator' for col in ind_missing_test.columns]

    avg_dict_train = get_avg_by_year(train['train_ratings_all'])
    avg_dict_test = get_avg_by_year(test['test_ratings_all'])

    train['train_ratings_all'] = fill_missing(train['train_ratings_all'])
    test['test_ratings_all'] = fill_missing(test['test_ratings_all'])

    train['train_ratings_all'] = set_dates_feat(train, kind='train')
    test['test_ratings_all'] = set_dates_feat(test, kind='test')

    for year, value in avg_dict_train.items():
        train['train_ratings_all']['avg_{}_{}'.format(value[1], year)] = value[0]

    for year, value in avg_dict_test.items():
        test['test_ratings_all']['avg_{}_{}'.format(value[1], year)] = value[0]

    train['train_ratings_all'] = train['train_ratings_all'].join(ind_missing_train.iloc[:, 14:])
    test['test_ratings_all'] = test['test_ratings_all'].join(ind_missing_test.iloc[:, 14:])

    return train, test

def fill_missing(df_ratings_all):
    '''

    :param df_ratings_all:
    :return:
    '''

    for i in range(14, df_ratings_all.shape[1]):
        # print('filling column {}'.format(i))
        column_name = df_ratings_all.columns[i]
        train_temp = df_ratings_all[df_ratings_all[column_name] > 0].iloc[:, list(range(0, i))]
        test_temp = df_ratings_all[df_ratings_all[column_name] == 0].iloc[:, list(range(0, i))]
        y_temp = df_ratings_all[df_ratings_all[column_name] > 0].iloc[:, [i]]

        if (train_temp.shape[0] == 0) | (test_temp.shape[0] == 0):
            print('No missing values in {} column, number: {}'.format(column_name, i))
            continue
        fill_na = lin_model(train_temp, test_temp, y_temp, method='normal')

        df_ratings_all[column_name][df_ratings_all[column_name] == 0] = fill_na
        df_ratings_all[column_name][df_ratings_all[column_name] > 5] = 5
        df_ratings_all[column_name][df_ratings_all[column_name] < 1] = 1

    return df_ratings_all

def lin_model(x_train, x_test, y_train, method='normal'):
    clf = models.lin_model(x_train, y_train, method=method)
    preds = clf.predict(x_test)

    if method == 'Lasso':
        return list(more_itertools.flatten([preds]))

    return list(more_itertools.flatten(preds))

def set_dates_feat(dict, kind='train'):
    '''

    :param dict:
    :param kind:
    :return:
    '''

    df_ratings_all = dict['{}_ratings_all'.format(kind)].copy()
    df_y_date = dict['{}_y_date'.format(kind)].copy()
    df_dates_all = dict['{}_dates_all'.format(kind)].copy()

    for i in range(df_dates_all.shape[1]):
        df_y_date[i] = df_y_date[0]

    df_dates_all = df_dates_all.replace(0, np.nan)

    seen_on_same_day = ((df_dates_all - df_y_date) == 0)

    rating_on_same_day = pd.DataFrame(np.array(seen_on_same_day) * np.array(df_ratings_all))
    rating_on_same_day = rating_on_same_day.replace(0, np.nan)

    df_ratings_all['day average'] = rating_on_same_day.mean(axis=1)
    df_ratings_all['day average'] = df_ratings_all['day average'].replace(np.nan, 0)

    df_ratings_all['day std'] = rating_on_same_day.std(axis=1)
    df_ratings_all['day std'] = df_ratings_all['day std'].replace(np.nan, 0)

    df_ratings_all['day 1'] = (rating_on_same_day == 1).sum(axis=1)
    df_ratings_all['day 1'] = df_ratings_all['day 1'].replace(np.nan, 0)

    df_ratings_all['day 5'] = (rating_on_same_day == 5).sum(axis=1)
    df_ratings_all['day 5'] = df_ratings_all['day 5'].replace(np.nan, 0)

    df_ratings_all['movies_on_day'] = seen_on_same_day.sum(axis=1)
    df_ratings_all['date'] = df_y_date[0]

    return df_ratings_all


def get_avg_by_year(df_ratings_all):
    '''

    :param df_ratings_all:
    :return:
    '''
    avg_yesr_dict = {2005: 'before', 1990: 'before', 1995: 'before', 2000: 'before', 2001: 'after'}
    df = df_ratings_all.replace(0, np.NaN).copy()

    for year in avg_yesr_dict.keys():
        if avg_yesr_dict[year] == 'before':
            columns_avg_list = [i for i in df.columns[:99] if int(i[0]) <= year]
        else:
            columns_avg_list = [i for i in df.columns[:99] if int(i[0]) >= year]

        avg_yesr_dict[year] = [df[columns_avg_list].mean(axis=1), avg_yesr_dict[year]]

    return avg_yesr_dict


def split_train_dev(data, labels, test_size=0.2):
    x_train, x_dev, y_train, y_dev = train_test_split(data, labels, test_size=test_size)
    return x_train, x_dev, y_train, y_dev


def run_lin_model(x_train, x_test, y_train, y_test, alpha=0.006, method='Lasso'):
    print('\n')
    print('Training a Linear model - {} with alpha = {}'.format(method,alpha))
    clf = models.lin_model(x_train, y_train, alpha=alpha, method=method)
    preds_train = clf.predict(x_train)
    preds_dev = clf.predict(x_test)
    mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
    mse_test_reg = math.sqrt(mean_squared_error(y_test, preds_dev))
    print('RMSE on Train is: {}'.format(mse_train_reg))
    print('RMSE on Test is: {}'.format(mse_test_reg))
    return mse_train_reg, mse_test_reg


def run_gb(x_train, x_test, y_train, y_test):
    print('\n')
    clf = models.run_gb(x_train, y_train)
    preds_train = clf.predict(x_train)
    preds_dev = clf.predict(x_test)
    mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
    mse_test_reg = math.sqrt(mean_squared_error(y_test, preds_dev))
    print('RMSE on Train is: {}'.format(mse_train_reg))
    print('RMSE on Test is: {}'.format(mse_test_reg))
    return mse_train_reg, mse_test_reg

def run_lgb_model(x_train_full, x_test, y_train_full, y_test):
    print('\n')
    x_train, x_dev, y_train, y_dev = split_train_dev(x_train_full, y_train_full, test_size=0.2)
    clf = models.run_lgb(x_train, x_dev, y_train, y_dev)
    preds_train = clf.predict(x_train, num_iteration=clf.best_iteration)
    preds_dev = clf.predict(x_dev, num_iteration=clf.best_iteration)
    preds_test = clf.predict(x_test, num_iteration=clf.best_iteration)
    mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
    mse_test_reg = math.sqrt(mean_squared_error(y_test, preds_test))
    print('RMSE on Train is: {}'.format(mse_train_reg))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))
    print('RMSE on Test is: {}'.format(mse_test_reg))
    return mse_train_reg, mse_dev_reg ,mse_test_reg

def run_knn_cosine(x_train, x_test, y_train, y_test, k=100):
    print('\n')
    print('Training a KNN model with cosine_similarity for k = {}'.format(k))
    mse_train_reg, mse_dev_reg = models.knn(x_train, x_test, y_train, y_test, k=500, metric='cosine')
    print('RMSE on Train is: {}'.format(mse_train_reg))
    print('RMSE on Test is: {}'.format(mse_dev_reg))
    return mse_dev_reg

def run_knn_euclidean(x_train, x_test, y_train, y_test, k=100):
    print('\n')
    print('Training a KNN model with euclidean for k = {}'.format(k))
    mse_train_reg, mse_dev_reg = models.knn(x_train, x_test, y_train, y_test, k=500, metric='euclidean')
    print('RMSE on Train is: {}'.format(mse_train_reg))
    print('RMSE on Test is: {}'.format(mse_dev_reg))
    return mse_dev_reg

def run_cv_to_find_lammda(x_train, y_train, alphas, method='Lasso'):
    kf = KFold(x_train.shape[0], n_folds=10, )

    e_alphas = list()
    e_alphas_t = list()  # holds average r2 error
    for alpha in alphas:
        if method == 'Lasso':
            clf = linear_model.Lasso(alpha=alpha)
        if method == 'Ridge':
            clf = linear_model.Ridge(alpha=alpha)
        err = list()
        err_t = list()
        for tr_idx, tt_idx in kf:
            X_tr, X_tt = x_train.iloc[tr_idx], x_train.iloc[tt_idx]
            y_tr, y_tt = y_train.iloc[tr_idx], y_train.iloc[tt_idx]
            clf.fit(X_tr, y_tr)
            y_hat_tr = clf.predict(X_tr)
            y_hat = clf.predict(X_tt)
            err_t.append(math.sqrt(mean_squared_error(y_tr.values, y_hat_tr)))
            err.append(math.sqrt(mean_squared_error(y_tt.values, y_hat)))
        e_alphas.append(np.average(err))
        e_alphas_t.append(np.average(err_t))

    ind_alpha_min = np.argmin(e_alphas)
    alpha_min = alphas[ind_alpha_min]
    plt.figsize = (28, 15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alphas, e_alphas, 'b-', label='Dev')
    ax.plot(alphas, e_alphas_t, 'g--', label='Train')
    ax.set_xlabel("alpha")
    ax.set_ylabel("RMSE")
    plt.legend()
    plt.show()
    return alpha_min

def run_cv_to_find_k(x_train, y_train, ks, metric='cosine'):
    kf = KFold(x_train.shape[0], n_folds=10, )

    e_ks = list()
    e_ks_t = list()
    for k in ks:
        err = list()
        err_t = list()
        for tr_idx, tt_idx in kf:
            X_tr, X_tt = x_train.iloc[tr_idx], x_train.iloc[tt_idx]
            y_tr, y_tt = y_train.iloc[tr_idx], y_train.iloc[tt_idx]
            rmse_train, rmse_dev = models.knn(X_tr, X_tt, y_tr, y_tt, k=int(k), metric=metric)
            err.append(rmse_dev)
            err_t.append(rmse_train)
        e_ks.append(np.average(err))
        e_ks_t.append(np.average(err_t))

    ind_k_min = np.argmin(e_ks)
    k_min = ks[ind_k_min]
    plt.figsize = (28, 15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ks, e_ks, 'b-', label='Dev')
    ax.plot(ks, e_ks_t, 'g--', label='Train')
    ax.set_xlabel("alpha")
    ax.set_ylabel("RMSE")
    plt.legend()
    plt.show()
    return k_min