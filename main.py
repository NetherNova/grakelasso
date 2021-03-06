__author__ = 'martin'

from learning.grakelasso import GraKeLasso, ModelManager
import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
import csv

"""Main script for processing time estimation and feature selection - Lambda Grid Search"""
if __name__ == '__main__':
    # *************** Parameters ************* #
    BASE_PATH = "D:/Dissertation/Data Sets/CycleTime-ISWC/"
    np.random.seed(1)
    writer = csv.writer(open(BASE_PATH + "results_sparse.csv", "wb"), delimiter=",")
    writer_full = csv.writer(open(BASE_PATH + "results_full.csv", "wb"), delimiter=",")
    csv_header = ['Lambda', 'Lasso', 'Lasso Red', 'ElasticNet', 'ElasticNet Red', 'Graph', 'Graph Red', 'GraKe', 'GraKe Red']
    writer_full.writerow(csv_header)
    writer.writerow(csv_header)

    lambda_range = [0.1, 1.1]   # inclusive 1.0
    n_fold = 10
    n_iter = 1000
    p_val = 0.05
    num_examples_sparse = 40
    num_examples_big = 2000
    response = "PackagingCycleTime"

    # *************** Load Data ************** #
    mm = ModelManager()
    file_list = [BASE_PATH + "foerdern.txt", BASE_PATH + "foerdern_ind.txt",
                 BASE_PATH + "testen.txt", BASE_PATH + "beladen.txt",
                 BASE_PATH + "verpacken.txt"]
    mm.load_data(file_list)
    k_sem_reduced = mm.load_kernel_laplacian(BASE_PATH + "kernel.csv")
    k_full = mm.load_kernel_laplacian(BASE_PATH + "full_kernel.csv")
    k_reg_reduced = mm.load_kernel_laplacian(BASE_PATH + "p_value_kernel.csv")

    dependency_graph_full = mm.load_kernel_laplacian(BASE_PATH + "dependency_full.csv")
    dependency_graph_sem_reduced = mm.load_kernel_laplacian(BASE_PATH + "dependency.csv")

    index_sparse = np.ones(num_examples_sparse, dtype=bool)
    index_sparse = np.concatenate((index_sparse, np.zeros(mm.num_examples() - num_examples_sparse - 1, dtype=bool)))
    np.random.shuffle(index_sparse)

    index_big = np.ones(num_examples_big, dtype=bool)
    index_big = np.concatenate((index_big, np.zeros(mm.num_examples() - num_examples_big - 1, dtype=bool)))
    np.random.shuffle(index_big)

    # ****************** Semantic FS and Standard Regression FS ****************** #
    features = list(k_sem_reduced.columns.values)
    k_sem_reduced.columns = [f.replace("http://www.i40.com/ontology#", "") for f in features]
    print("Getting features: ", k_sem_reduced.columns.values)
    X_sem = mm.get_all_features_except_response(response, index_big, k_sem_reduced)
    num_features_sem = X_sem.shape[1]
    print("Semantic reduced features: ", k_sem_reduced.columns.values)

    X_all = mm.get_all_features_except_response(response, index_big)
    num_features_all = X_all.shape[1]
    y_all = mm.get_data().ix[index_big, response]
    mean_y_all = np.mean(y_all)
    F, p_vals = f_regression(X_all, y_all)
    index_reg_reduced = p_vals <= p_val
    X_reg = X_all.ix[:, index_reg_reduced]
    num_features_reg = X_reg.shape[1]

    print("P-value reduced features: ", k_full.columns.values[index_reg_reduced])

    for alpha in np.arange(0.1, 2.1, 0.1):
        # **************************** Full Data Set *************************** #

        grake_lasso = GraKeLasso(k_full.as_matrix(), alpha)
        glasso = GraKeLasso(dependency_graph_full.as_matrix(), alpha)
        lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                      max_iter=n_iter, tol=1e-4, positive=False)
        elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                                max_iter=n_iter, copy_X=True, tol=1e-4, positive=False)
        lambda_glasso = glasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=None)
        lambda_grake_lasso = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=None)
        lambda_elastic_net = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=elasticNet)
        print("Full Data Set:")
        print("Evaluating Lasso:")
        lasso_full, lasso_full_red = grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, 0, model=lasso)
        print("Evaluating Elastic Net:")
        eNet_full, eNet_full_red = grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
        print("Evaluating GraKeLasso:")
        grake_full, grake_full_red = grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_grake_lasso, model=None)
        print("Evaluating Glasso:")
        glasso_full, glasso_full_red = glasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_glasso, model=None)
        print("---------------------------------------------------------------------------")

        # **************************** Reg reduced performance *************************** #



        # ******************************Sparse Data Set************************************ #
        X_sparse = mm.get_all_features_except_response(response, index_sparse)
        y_sparse = mm.get_data().ix[index_sparse, response]
        mean_y_sparse = np.mean(y_sparse)
        grake_lasso = GraKeLasso(k_full.as_matrix(), alpha)
        glasso = GraKeLasso(dependency_graph_full.as_matrix(), alpha)
        lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                      max_iter=n_iter, tol=1e-4, positive=False)
        elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                                max_iter=n_iter, copy_X=True, tol=1e-4, positive=False)
        lambda_glasso = glasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=None)
        lambda_grake_lasso = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=None)
        lambda_elastic_net = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=elasticNet)

        lasso_sparse, lasso_sparse_red = grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, 0, model=lasso)
        eNet_sparse, eNet_sparse_red = grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
        grake_sparse, grake_sparse_red = grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_grake_lasso, model=None)
        glasso_sparse, glasso_sparse_red = glasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_glasso, model=None)

        print("Lambda: ", alpha)
        print("---------------------------------------------------------------------------")
        print("Full:")
        print("Lasso: ", lasso_full / mean_y_all, lasso_full_red)
        print("ENet: ", eNet_full / mean_y_all, eNet_full_red)
        print("Glasso, ", glasso_full / mean_y_all, glasso_full_red)
        print("Grake, ", grake_full / mean_y_all, grake_full_red)

        writer_full.writerow([alpha, lasso_full / mean_y_all, lasso_full_red, eNet_full / mean_y_all, eNet_full_red,
                              glasso_full / mean_y_all, glasso_full_red, grake_full / mean_y_all, grake_full_red])

        print("---------------------------------------------------------------------------")
        print("Sparse:")
        print("Lasso: ", lasso_sparse / mean_y_sparse, lasso_sparse_red)
        print("ENet: ", eNet_sparse / mean_y_sparse, eNet_sparse_red)
        print("Glasso, ", glasso_sparse / mean_y_sparse, glasso_sparse_red)
        print("Grake, ", grake_sparse / mean_y_sparse, grake_sparse_red)
        print("---------------------------------------------------------------------------")

        writer.writerow([alpha, lasso_sparse / mean_y_sparse, lasso_sparse_red, eNet_sparse / mean_y_sparse, eNet_sparse_red,
                         glasso_sparse / mean_y_sparse, glasso_sparse_red, grake_sparse / mean_y_sparse, grake_sparse_red])