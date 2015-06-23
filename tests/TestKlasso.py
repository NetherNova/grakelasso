__author__ = 'martin'

from learning.deklasso import KLasso
import numpy as np

X = np.array([[1, 10, 3],
              [1, 2, 5],
              [-3, 22, 23],
              [2, 3, 1]])

y = np.array([1, 2, 3, 4])
kernel = np.matrix([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])
lasso = KLasso(kernel)
print(lasso.enhanced_data_matrix(X))

# theta = lasso.train(X, y, 0.9, 2)
# print(theta)