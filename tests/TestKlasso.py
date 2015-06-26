__author__ = 'martin'

from learning.deklasso import KLasso, ModelManager
from sklearn.linear_model import Lasso
import numpy as np

lambd = 0.5
alpha = 0.9
num_examples = 80

# *************** Load Data ************** #
mm = ModelManager()
file_list = ["D:/Dissertation/Data Sets/foerdern.txt", "D:/Dissertation/Data Sets/foerdern_ind.txt",
                 "D:/Dissertation/Data Sets/testen.txt", "D:/Dissertation/Data Sets/beladen.txt",
                 "D:/Dissertation/Data Sets/verpacken.txt"]
mm.load_data(file_list)
k_sem_reduced = mm.load_kernel_laplacian("D:/Dissertation/Data Sets/kernel.csv")
k_full = mm.load_kernel_laplacian("D:/Dissertation/Data Sets/full_kernel.csv")
data = mm.get_data()

index_sparse = np.ones(num_examples, dtype=bool)
index_sparse = np.concatenate((index_sparse, np.zeros(mm.num_examples() - num_examples - 1, dtype=bool)))
np.random.shuffle(index_sparse)
X_sparse = mm.get_all_features_except_response("PackagingCycleTime").ix[index_sparse, :]
y_sparse = data.ix[index_sparse, 'PackagingCycleTime']
mean_y_sparse = np.mean(y_sparse)
k_full = mm.load_kernel_laplacian("D:/Dissertation/Data Sets/full_kernel.csv")
klasso = KLasso(k_full.as_matrix(), alpha)
# Evaluate Grake
klasso.cross_val(X_sparse, y_sparse, 10, 10000, lambd)
reg_denom = np.sqrt(1 + 0.5 * ((alpha - alpha * lambd)))
lasso = Lasso(alpha= (alpha*lambd) / reg_denom, fit_intercept=False, normalize=False, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-8, positive=False)
X = klasso.enhanced_data_matrix(X_sparse, lambd, reg_denom)
y = np.append(y_sparse, np.zeros(X_sparse.shape[1]))
# Evaluate modified Lasso
klasso.cross_val(X, y, 10, 1000, lambd, model=lasso)
lasso = Lasso(alpha=alpha*lambd, fit_intercept=False, normalize=False, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-8, positive=False)
# Evaluate standard Lasso
klasso.cross_val(X_sparse, y_sparse, 10, 1000, lambd, model=lasso)