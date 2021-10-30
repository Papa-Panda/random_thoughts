# factorization machine implementation
# https://www.jianshu.com/p/610dff83f709


import numpy as np
from random import normalvariate
# from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler as MM
import pandas as pd
data_train = pd.read_csv('diabetes_train.txt', header=None)
data_test = pd.read_csv('diabetes_test.txt', header=None)


def preprocessing(data_input):
    standardopt = MM()
    data_input.iloc[:, -1].replace(0, -1, inplace=True)
    feature = data_input.iloc[:, :-1]
    feature = standardopt.fit_transform(feature)
    feature = np.mat(feature)#传回来的是array，如果要dataframe那用dataframe
    label = np.array(data_input.iloc[:, -1])
    return feature, label


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sgd_fm(datamatrix, label, k, iter, alpha):
    '''
    k：分解矩阵的长度
    '''
    m, n = np.shape(datamatrix)
    w0 = 0.0
    w = np.zeros((n, 1))
    v = normalvariate(0, 0.2) * np.ones((n, k))
    for it in range(iter):
        for i in range(m):
            # inner1 = datamatrix[i] * w
            inner1 = datamatrix[i] * v
            inner2 = np.multiply(datamatrix[i], datamatrix[i]) * np.multiply(v, v)
            jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
            ypredict = w0 + datamatrix[i] * w + jiaocha
            # print(np.shape(ypredict))
            # print(ypredict[0, 0])
            yp = sigmoid(label[i]*ypredict[0, 0])
            loss = 1 - (-(np.log(yp)))
            w0 = w0 - alpha * (yp - 1) * label[i] * 1
            for j in range(n):
                if datamatrix[i, j] != 0:
                    w[j] = w[j] - alpha * (yp - 1) * label[i] * datamatrix[i, j]
                    for k in range(k):
                        v[j, k] = v[j, k] - alpha * ((yp - 1) * label[i] * \
                                  (datamatrix[i, j] * inner1[0, k] - v[j, k] * \
                                  datamatrix[i, j] * datamatrix[i, j]))
        print('第%s次训练的误差为：%f' % (it, loss))
    return w0, w, v


def predict(w0, w, v, x, thold):
    inner1 = x * v
    inner2 = np.multiply(x, x) * np.multiply(v, v)
    jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
    ypredict = w0 + x * w + jiaocha
    y0 = sigmoid(ypredict[0,0])
    if y0 > thold:
        yp = 1
    else:
        yp = -1
    return yp


def calaccuracy(datamatrix, label, w0, w, v, thold):
    error = 0
    for i in range(np.shape(datamatrix)[0]):
        yp = predict(w0, w, v, datamatrix[i], thold)
        if yp != label[i]:
            error += 1
    accuray = 1.0 - error/np.shape(datamatrix)[0]
    return accuray

datamattrain, labeltrain = preprocessing(data_train)
datamattest, labeltest = preprocessing(data_test)
w0, w, v = sgd_fm(datamattrain, labeltrain, 20, 300, 0.01)
maxaccuracy = 0.0
tmpthold = 0.0
for i in np.linspace(0.4, 0.6, 201):
    print(i)
    accuracy_test = calaccuracy(datamattest, labeltest, w0, w, v, i)
    if accuracy_test > maxaccuracy:
        maxaccuracy = accuracy_test
        tmpthold = i
print(accuracy_test, tmpthold)
