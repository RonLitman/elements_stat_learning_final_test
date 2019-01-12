import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import itertools

def load_and_set_data():
    '''
    loads and sets to data to have only 2&3
    :return:
    '''
    digit_for_class = [2,3,5]
    train = pd.read_csv('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Final Test/data/test.txt', header=None, delimiter=r"\s+")
    test = pd.read_csv('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Final Test/data/train.txt', header=None, delimiter=r"\s+")
    train = train[train[0].isin(digit_for_class)].reset_index(drop=True)
    test = test[test[0].isin(digit_for_class)].reset_index(drop=True)
    return train, test

def set_train_test(train, test):
    '''
    set the vectors X & Y for train and test
    :param train:
    :param test:
    :return:
    '''
    x_train = train.drop(train.columns[0], axis=1)
    x_test = test.drop(test.columns[0], axis=1)

    y_train = train[0]
    y_test = test[0]

    print('X Train data Shape: {}'.format(x_train.shape))
    print('X Test data Shape: {}'.format(x_test.shape))
    print('Y Train data Shape: {}'.format(y_train.shape))
    print('Y Test data Shape: {}'.format(y_test.shape))

    x_train = np.asarray(x_train).reshape(-1, 16, 16, 1)
    x_test = np.asarray(x_test).reshape(-1, 16, 16, 1)
    x_train = np.asarray(x_train).astype("float32")
    x_test = np.asarray(x_test).astype("float32")

    y_train = to_categorical(y_train)[:, [2, 3, 5]]
    y_test = to_categorical(y_test)[:, [2, 3, 5]]

    return x_train, x_test, y_train, y_test

def plot_some_number(x_train, y_train, ind_list=[1,2,500]):
    '''
    plot some numbers
    :param ind_list:
    :return:
    '''
    for i in ind_list:
        img = x_train.iloc[i].as_matrix()
        img = img.reshape((16, 16))
        plt.imshow(img, cmap='gray')
        plt.title(y_train.iloc[i])
        plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((16,16)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1