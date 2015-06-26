__author__ = 'martin'

from learning.deklasso import KLasso, ModelManager
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression

"""Main script for processing time estimation and feature selection"""
if __name__ == '__main__':
    # *************** Parameters ************* #
    np.random.seed(24)
    #np.random.seed(25)
    lambda_range = [0.0, 1.0]
    n_fold = 10
    n_iter = 100
    alpha = 0.5
    p_val = 0.05
    num_examples_sparse = 40
    num_examples_big = 2000
    # *************** Load Data ************** #
    mm = ModelManager()
    file_list = ["D:/Dissertation/Data Sets/foerdern.txt", "D:/Dissertation/Data Sets/foerdern_ind.txt",
                 "D:/Dissertation/Data Sets/testen.txt", "D:/Dissertation/Data Sets/beladen.txt",
                 "D:/Dissertation/Data Sets/verpacken.txt"]
    mm.load_data(file_list)
    k_sem_reduced = mm.load_kernel_laplacian("D:/Dissertation/Data Sets/kernel.csv")
    k_full = mm.load_kernel_laplacian("D:/Dissertation/Data Sets/full_kernel.csv")
    data = mm.get_data()

    index_sparse = np.ones(num_examples_sparse, dtype=bool)
    index_sparse = np.concatenate((index_sparse, np.zeros(mm.num_examples() - num_examples_sparse - 1, dtype=bool)))
    np.random.shuffle(index_sparse)

    index_big = np.ones(num_examples_big, dtype=bool)
    index_big = np.concatenate((index_big, np.zeros(mm.num_examples() - num_examples_big - 1, dtype=bool)))
    np.random.shuffle(index_big)

    # ******************Big Data Set OLS on Semantic FS and Standard Regression FS****************** #
    features = list(k_sem_reduced.columns.values)
    features_rep = [f.replace("http://www.i40.com/ontology#", "") for f in features]
    print("Getting features: ", features_rep)
    X_sem = data.ix[index_big, features_rep[0]]
    for f in features_rep[1:len(features_rep)]:
        X_sem = pd.concat([X_sem, data.ix[index_big, f]], axis=1)
    num_features_sem = X_sem.shape[1]
    print("Semantic reduced features: ", num_features_sem)

    X_all = mm.get_all_features_except_response("PackagingCycleTime").ix[index_big, :]
    num_features_all = X_all.shape[1]
    y_all = data.ix[index_big, 'PackagingCycleTime']
    mean_y_all = np.mean(y_all)
    F, p_vals = f_regression(X_all, y_all)
    index_reg_reduced = p_vals <= p_val
    X_reg = X_all.ix[:, index_reg_reduced]
    num_features_reg = X_reg.shape[1]
    print("P-value reduced features: ", num_features_reg)

    # **************************** Full Data Set *************************** #
    k_reg_reduced = k_full.ix[index_reg_reduced, index_reg_reduced].as_matrix()

    grake_lasso = KLasso(k_full.as_matrix(), alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=lasso)
    print("Full Data Set:")
    print("Evaluating Lasso:")
    lasso_full = grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Evaluating Elastic Net:")
    eNet_full = grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("Evaluating OLS:")
    ols_full = grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, 0.5, model=ols)
    print("Evaluating GraKeLasso:")
    grake_full = grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")

    # **************************** Reg reduced performance *************************** #
    k_reg_reduced = k_full.ix[index_reg_reduced, index_reg_reduced].as_matrix()

    # TODO: replace with actual Laplacian of reduced and full kernel
    # TODO: Vergleich machen zwischen dependency network und kernel approach

    grake_lasso = KLasso(k_reg_reduced, alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_reg, y_all, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_reg, y_all, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_reg, y_all, n_fold, n_iter, lambda_range, model=lasso)
    print("p-value reduction:")
    print("Evaluating Lasso:")
    lasso_p = grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Evaluating Elastic Net:")
    eNet_p = grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("Evaluating OLS:")
    ols_p = grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, 0.5, model=ols)
    print("Evaluating GraKeLasso:")
    grake_p = grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")

    # **************************** Sem reduced performance *************************** #
    grake_lasso = KLasso(k_sem_reduced.as_matrix(), alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_sem, y_all, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_sem, y_all, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_sem, y_all, n_fold, n_iter, lambda_range, model=lasso)
    print("Semantic reduction:")
    print("Evaluating Lasso:")
    lasso_sem = grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Evaluating Elastic Net:")
    eNet_sem = grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("Evaluating OLS:")
    ols_sem = grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, 0.5, model=ols)
    print("Evaluating GraKeLasso:")
    grake_sem = grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")

    # ******************************Sparse Data Set************************************ #
    X_sparse = mm.get_all_features_except_response("PackagingCycleTime").ix[index_sparse, :]
    y_sparse = data.ix[index_sparse, 'PackagingCycleTime']
    mean_y_sparse = np.mean(y_sparse)
    grake_lasso = KLasso(k_full.as_matrix(), alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=lasso)
    print("Sparse data set (full features):")
    print("Lasso:")
    lasso_sparse = grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Elastic Net:")
    eNet_sparse = grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("OLS:")
    ols_sparse = grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, 0.5, model=ols)
    print("GraKeLasso:")
    grake_sparse = grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")
    print("Full:")
    print("Lasso: ", lasso_full)
    print("ENet: ", eNet_full)
    print("OLS:, ", ols_full)
    print("Grake, ", grake_full)
    print("---------------------------------------------------------------------------")
    print("P-Value:")
    print("Lasso: ", lasso_p)
    print("ENet: ", eNet_p)
    print("OLS:, ", ols_p)
    print("Grake, ", grake_p)
    print("---------------------------------------------------------------------------")
    print("Semantic:")
    print("Lasso: ", lasso_sem)
    print("ENet: ", eNet_sem)
    print("OLS:, ", ols_sem)
    print("Grake, ", grake_sem)
    print("---------------------------------------------------------------------------")
    print("Sparse:")
    print("Lasso: ", lasso_sparse)
    print("ENet: ", eNet_sparse)
    print("OLS:, ", ols_sparse)
    print("Grake, ", grake_sparse)
    print("---------------------------------------------------------------------------")
    print("Mean y_all: ", mean_y_all)
    print("Mean y_sparse: ", mean_y_sparse)