__author__ = 'martin'

from learning.grakelasso import KLasso, ModelManager
import numpy as np

lambd = 0.1
alpha = 1
num_examples = 1000
response = "TestingProduct"

# *************** Load Data ************** #
mm = ModelManager()
mm.load_data(["../data/test.txt"])
kernel_lap = mm.load_kernel_laplacian("../data/laplacian.csv")
data = mm.get_data()

index_sparse = np.ones(num_examples, dtype=bool)
index_sparse = np.concatenate((index_sparse, np.zeros(mm.num_examples() - num_examples - 1, dtype=bool)))
np.random.shuffle(index_sparse)

X_sparse = mm.get_all_features_except_response(response).ix[index_sparse, :]
y_sparse = data.ix[index_sparse, response]

# Evaluate GraKeLasso
klasso = KLasso(kernel_lap.as_matrix(), alpha)
mse = klasso.cross_val(X_sparse, y_sparse, 10, 10000, lambd)
print("MSE: ", mse)
