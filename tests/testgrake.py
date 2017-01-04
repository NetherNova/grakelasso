__author__ = 'martin'

from learning.grakelasso import GraKeLasso, ModelManager
import numpy as np

import unittest

lambd = 0.1
alpha = 1
num_examples = 1000
response = "TestingProduct"


class GraphKernelTest(unittest.TestCase):
    def test(self):
        """ Load test kernel laplacian and features """
        mm = ModelManager()
        mm.load_data(["../data/test.txt"])
        kernel_lap = mm.load_kernel_laplacian("../data/laplacian.csv")
        data = mm.get_data()

        index_sparse = np.ones(num_examples, dtype=bool)
        index_sparse = np.concatenate((index_sparse, np.zeros(mm.num_examples() - num_examples - 1, dtype=bool)))
        np.random.shuffle(index_sparse)

        X_sparse = mm.get_all_features_except_response(response, index_sparse)
        y_sparse = data.ix[index_sparse, response]

        # Evaluate GraKeLasso
        klasso = GraKeLasso(kernel_lap.as_matrix(), alpha)
        rmse, avg_theta = klasso.cross_val(X_sparse, y_sparse, 10, 10000, lambd)
        print("MSE and Coefficient Reduction ", rmse, avg_theta)
        self.assertTrue(rmse > 0)
        self.assertTrue(avg_theta > 0)
