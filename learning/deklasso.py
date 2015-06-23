__author__ = 'martin'
# -*- coding: UTF-8 -*-

import numpy as np
import logging
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.linear_model.base import center_data
from sklearn.feature_selection import f_regression
_logger = logging.getLogger(__name__)


class KLasso(object):
    """A Lasso regression with kernel regularization"""
    def __init__(self, kernel):
        self.kernel = kernel
        self.n = 0
        self.alpha = 0.1

    def cross_val_lambda(self, X, y, n_fold, n_iter, lambda_range, model=None):
        """
        Evaluate lambda parameter of Lasso Models
        :param X: Feature matrix
        :param y: Response
        :param n_fold: number of cross-vals for each lambda
        :param n_iter: training iterations
        :param lambda_range: range of lambda values to test
        :param model: the learning model
        :return: best lambda value
        """
        best_lambda = 0
        error = np.inf
        for lambda_cur in np.arange(lambda_range[0], lambda_range[1], 0.1):
                avg_error = self.cross_val(X, y, n_fold, n_iter, lambda_cur, model=model)
                if avg_error < error:
                    error = avg_error
                    best_lambda = lambda_cur
        print("Best lambda %s for model %s: ", best_lambda, model)
        return best_lambda

    def cross_val(self, X, y, n_fold, n_iter, lambd, model=None):
        """
        Perform general cross-validation
        :param X: Feature matrix
        :param y: Response
        :param n_fold: how many cross-val runs
        :param n_iter: training iterations
        :param lambd: reguralization parameter
        :param model: learning model
        :return:
        """
        X, y, X_mean, y_mean, X_std = center_data(X, y, fit_intercept=True, normalize=True)
        train_prct = 1 - (n_fold / 100.0)
        n_rows = np.floor(X.shape[0] * train_prct)
        index = np.ones(n_rows, dtype=bool)
        index = np.concatenate((index, np.zeros(X.shape[0] - n_rows - 1, dtype=bool)))
        avg_error = 0
        for i in xrange(n_fold):
            np.random.shuffle(index)
            new_index = 1-index
            new_index = np.array(new_index, dtype=bool)
            num_test_examples = sum(new_index)
            if model:
                model.l1_ratio_ = lambd
                model.fit(X[index, :], y[index])
                theta = model.coef_
                y_temp = np.array(y[new_index])
                y_temp.shape = num_test_examples
            else:
                theta = self.train(X[index, :], y[index], lambd, n_iter)
                y_temp = np.array(y[new_index])
                y_temp.shape = (num_test_examples, 1)
                y_temp.shape = num_test_examples
            predict = np.dot(X[new_index, :], theta)
            errors = y_temp - predict
            error = np.sqrt(1/(1.0*num_test_examples)*sum(np.square(errors)))
            avg_error += error
        avg_error = avg_error / (1.0 * n_fold)
        print("Average Error: ", avg_error)
        return avg_error

    def train(self, X, y, lambd, n_iter):
        """
        Coordinate-descent algorithm - extended data set
        :param X: Feature Matrix
        :param y: response
        :param lambd: regularization parameter
        :param n_iter: max training iterations
        :return: fitted coefficients
        """
        X, y, X_mean, y_mean, X_std = center_data(X, y, fit_intercept=True, normalize=True)
        self.n_rows = X.shape[0]
        X = self.enhanced_data_matrix(X)    # augmented data set
        y = np.append(y, np.zeros(self.kernel.shape[0]))    # add zeros to response
        # y.shape = (self.n_rows, 1)
        if lambd == 0:
            lambd = 0.01
        self.n = X.shape[1]
        if self.kernel is None:
            k_matrix = np.eye(self.n)
        else:
            k_matrix = self.kernel
        theta = np.array([np.random.normal() for j in xrange(0, self.n)])
        # theta.shape = (self.n, 1)
        prev_error = 0
        k_sums = [sum(k_matrix[k]) for k in xrange(0, len(theta))]
        for i in xrange(1, n_iter):
            ind = np.ones(len(theta), dtype=bool)
            for k in xrange(0, len(theta)):
                ind[k] = False
                r = y - np.dot(X[:, ind], theta[ind])
                num = np.dot(np.transpose(X[:, k]), r)
                denom = np.dot(np.transpose(X[:, k]), X[:, k])
                temp = num / ((2*lambd*(1-self.alpha)) * k_sums[k] + denom) # TODO: k_sums replaced with shrinkage
                norm = np.linalg.norm(X[:, k], 2)**2
                if norm == 0:
                    norm = 0.1
                theta[k] = self.soft_threshold(temp, 1/(2*self.alpha*lambd)) * (1.0/norm)
                ind[k] = True
            errors1 = y - np.dot(X, theta)
            train_error = np.sqrt(1/(1.0*len(errors1))*sum(np.square(errors1)))
            if abs(prev_error - train_error) < 0.01 or sum([theta[k] != 0 for k in xrange(0, len(theta))]) == 0:
                break
            else:
                prev_error = train_error
        return theta

    def enhanced_data_matrix(self, X):
        """Concatenate the graph-constrained regularization to the data matrix"""
        S = np.linalg.cholesky(self.kernel)
        X_star = np.append(X, S, axis=0)
        return X_star

    def set_alpha(self, alpha):
        self.alpha = alpha

    def soft_threshold_vec(self, weights, lambd):
        """Derivative of l-1 regularization, i.e. *shrinkage* operator"""
        result = np.zeros((self.n, 1))
        for i in xrange(1, self.n):
            if weights[i] > lambd:
                result[i] = weights[i] - lambd
            elif abs(weights[i]) <= lambd:
                result[i] = 0
            else:
                result[i] = weights[i] + lambd
        return result

    def soft_threshold(self, weight, lambd):
        """Derivative of l-1 regularization, i.e. *shrinkage* operator"""
        result = 0
        if weight > lambd:
            result = weight - lambd
        elif abs(weight) <= lambd:
            result = 0
        else:
            result = weight + lambd
        return result


class ModelManager(object):
    """Handles semantic data models for the automation system / MOM-layer """

    def __init__(self):
        self.featureIndex = dict()
        self.data = None
        self.kernel = None

    def __del__(self):
        pass

    def load_data(self, files):
        dfs = []
        for f in files:
            fp = open(f, "rb")
            data = pd.read_csv(fp, sep='\t')
            dfs.append(data.loc[:2100, :])
        self.data = pd.concat(dfs, axis=1)
        self.data.rename(columns=lambda x: x.strip())
        headers = list(self.data.columns.values)
        for i, h in enumerate(headers):
            self.featureIndex[h] = i
        _logger.debug("Loaded data ", self.data)

    def load_kernel(self, file):
        fp = open(file, "rb")
        self.kernel = pd.read_csv(fp, sep=',')
        return self.kernel

    def get_all_features_except_response(self, response):
        print("Getting all features, except: ", response)
        X = None
        for f in self.data.columns.values:
            if f == response:
                continue
            elif X is None:
                X = self.data.ix[:, f]
            else:
                X = pd.concat([X, self.data.ix[:, f]], axis=1)
        return X

    def get_data(self):
        return self.data

    def num_examples(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # *************** Parameters ************* #
    np.random.seed(24)
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
    k_sem_reduced = mm.load_kernel("D:/Dissertation/Data Sets/kernel.csv")
    k_full = mm.load_kernel("D:/Dissertation/Data Sets/full_kernel.csv")
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

    grake_lasso = KLasso(k_full.as_matrix())
    grake_lasso.set_alpha(alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_all, y_all, n_fold, n_iter, lambda_range, model=lasso)
    print("Performance Full Data Set:")
    print("Lasso:")
    grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Elastic Net:")
    grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("OLS:")
    grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, 0.5, model=ols)
    print("GraKeLasso:")
    grake_lasso.cross_val(X_all, y_all, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")

    # **************************** Reg reduced performance *************************** #
    k_reg_reduced = k_full.ix[index_reg_reduced, index_reg_reduced].as_matrix()

    grake_lasso = KLasso(k_reg_reduced)
    grake_lasso.set_alpha(alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_reg, y_all, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_reg, y_all, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_reg, y_all, n_fold, n_iter, lambda_range, model=lasso)
    print("Performance under p-value reduction:")
    print("Lasso:")
    grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Elastic Net:")
    grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("OLS:")
    grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, 0.5, model=ols)
    print("GraKeLasso:")
    grake_lasso.cross_val(X_reg, y_all, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")

    # **************************** Sem reduced performance *************************** #
    grake_lasso = KLasso(k_sem_reduced.as_matrix())
    grake_lasso.set_alpha(alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_sem, y_all, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_sem, y_all, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_sem, y_all, n_fold, n_iter, lambda_range, model=lasso)
    print("Performance under semantic reduction:")
    print("Lasso:")
    grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Elastic Net:")
    grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("OLS:")
    grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, 0.5, model=ols)
    print("GraKeLasso:")
    grake_lasso.cross_val(X_sem, y_all, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")

    # ******************************Sparse Data Set************************************ #
    X_sparse = mm.get_all_features_except_response("PackagingCycleTime").ix[index_sparse, :]
    y_sparse = data.ix[index_sparse, 'PackagingCycleTime']
    mean_y_sparse = np.mean(y_sparse)
    grake_lasso = KLasso(k_full.as_matrix())
    grake_lasso.set_alpha(alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, precompute='auto', copy_X=True,
                   max_iter=1000, tol=1e-4, positive=False)
    elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=1000, copy_X=True, tol=1e-4, positive=False)
    ols = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lambda_grake_lasso = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=None)
    lambda_elastic_net = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=elasticNet)
    lambda_lasso = grake_lasso.cross_val_lambda(X_sparse, y_sparse, n_fold, n_iter, lambda_range, model=lasso)
    print("Performance on sparse data set (full features):")
    print("Lasso:")
    grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_lasso, model=lasso)
    print("Elastic Net:")
    grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_elastic_net, model=elasticNet)
    print("OLS:")
    grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, 0.5, model=ols)
    print("GraKeLasso:")
    grake_lasso.cross_val(X_sparse, y_sparse, n_fold, n_iter, lambda_grake_lasso, model=None)
    print("---------------------------------------------------------------------------")
    print("Mean y_all: ", mean_y_all)
    print("Mean y_sparse: ", mean_y_sparse)