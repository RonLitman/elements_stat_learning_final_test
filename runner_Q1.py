import numpy as np
import pandas as pd
from utilities import *


movie_titles, train, test = load_and_set_data()

train, test = set_df(train, test)

x_train, x_test, y_train, y_test = split_train_dev(train['train_ratings_all'], train['train_y_rating'], test_size=0.2)

# ks = np.linspace(1, 100, num=100)
# run_cv_to_find_k(x_train, y_train, ks, metric='cosin')

# alphas = np.linspace(0.0, 0.01, num=100)
# run_cv_to_find_lammda(x_train, y_train, alphas, method='Lasso')

mse_train_reg_lin_l1, mse_test_reg_lin_l1 = run_lin_model(x_train, x_test, y_train, y_test, method='Lasso')

# alphas = np.linspace(100, 1000, num=100)
# run_cv_to_find_lammda(x_train, y_train, alphas, method='Ridge')

mse_train_reg_lin_l2, mse_test_reg_lin_l2 = run_lin_model(x_train, x_test, y_train, y_test, alpha=400.0, method='Ridge')

mse_train_reg_lgb, mse_dev_reg_lgb, mse_test_reg_lgb = run_lgb_model(x_train, x_test, y_train, y_test)

mse_test_reg_cosine = run_knn_cosine(x_train, x_test, y_train, y_test, k=100)

mse_test_reg_manhattan = run_knn_euclidean(x_train, x_test, y_train, y_test, k=100)

mse_train_reg_gb, mse_test_reg_gb = run_gb(x_train, x_test, y_train, y_test)

