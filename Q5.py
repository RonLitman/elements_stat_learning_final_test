import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from utilities import *

def run_model(x_train, y_train, alphas, method='Lasso'):
    e_alphas_t = list()
    sum_coef_abs = list()
    sum_coef_sqr = list()
    for alpha in alphas:
        if method == 'Lasso':
            clf = linear_model.Lasso(alpha=alpha)
        if method == 'Ridge':
            clf = linear_model.Ridge(alpha=alpha)
        clf.fit(x_train, y_train)
        sum_coef_abs.append(np.sum(np.abs(clf.coef_)))
        sum_coef_sqr.append(np.sum(np.power(clf.coef_, 2)))
        y_hat_tr = clf.predict(x_train)
        e_alphas_t.append(math.sqrt(mean_squared_error(y_train.values, y_hat_tr)))

    return e_alphas_t, sum_coef_abs, sum_coef_sqr


def plot_values(alphas, e_alphas, title_x, title):
    plt.figsize = (28, 15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alphas, e_alphas, 'b-')
    ax.set_xlabel(title_x)
    ax.set_ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.show()


movie_titles, train, test = load_and_set_data()

x_train = train['train_ratings_all'].iloc[:100,]
y_train = train['train_y_rating'].iloc[:100,]
exp_point = np.asarray([5 for i in range(99)])

gamma = 5
ker_dist_5 = rbf_kernel(x_train, exp_point.reshape(1, -1), gamma=gamma)
print(pd.DataFrame(ker_dist_5).describe())

gamma = 0.0001
ker_dist_1 = rbf_kernel(x_train, exp_point.reshape(1, -1), gamma=gamma)
print(pd.DataFrame(ker_dist_1).describe())

print(y_train.describe())
