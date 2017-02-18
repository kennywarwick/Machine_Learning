#coding:UTF-8
from sklearn import datasets
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
iris = datasets.load_iris()
# print iris
# print type(iris.data) ##<type 'numpy.ndarray'>
# print iris.data[1:5]
'''[[ 4.9  3.   1.4  0.2]
    [ 4.7  3.2  1.3  0.2]
    [ 4.6  3.1  1.5  0.2]
    [ 5.   3.6  1.4  0.2]]'''
# print type(iris.target) ##<type 'numpy.ndarray'>
# print iris.target[1:5]
X = np.array()
y = np.array()

x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

