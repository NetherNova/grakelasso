__author__ = 'martin'
# -*- coding: UTF-8 -*-

import numpy as np
import logging
import pandas as pd
from sklearn.linear_model.base import center_data

logging.basicConfig(level=logging.INFO)

TOLERANCE = 0.001


class KLasso(object):
    """
    A Lasso regression with laplacian regularization
    :param kernel: symmetric positive semi-definite
    :param alpha: overall regularization parameter
    """
    def __init__(self, kernel, alpha):
        self.kernel = kernel
        self.n = 0
        self.alpha = alpha

    def cross_val_lambda(self, X, y, n_fold, n_iter, lambda_range, model=None):
        """
        Evaluate lambda parameter of Lasso Models
        :param X: Feature matrix
        :param y: Response
        :param n_fold: number of cross-vals for each lambda
        :param n_iter: training iterations
        :param lambda_range: range of lambda values to test
        :param model: the learning model (None for *THIS*)
        :return: best lambda value
        """
        best_lambda = 0
        error = np.inf
        for lambda_cur in np.arange(lambda_range[0], lambda_range[1], 0.1):
                avg_error = self.cross_val(X, y, n_fold, n_iter, lambda_cur, model=model)
                if avg_error < error:
                    error = avg_error
                    best_lambda = lambda_cur
        logging.debug("Best lambda= %s for model: %s", best_lambda, model)
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
                model.l1_ratio_ = lambd # if model has this property, i.e. ElasticNet
                model.fit(X[index, :], y[index])
                theta = model.coef_
                y_temp = np.array(y[new_index])
                y_temp.shape = num_test_examples
            else:
                theta = self.train(X[index, :], y[index], lambd, n_iter)
                y_temp = np.array(y[new_index])
                y_temp.shape = (num_test_examples, 1)
                y_temp.shape = num_test_examples
            logging.info("Theta: %s", theta)
            predict = np.dot(X[new_index, :], theta)
            errors = y_temp - predict
            error = np.sqrt(1/(1.0*num_test_examples)*sum(np.square(errors)))
            avg_error += error
        avg_error = avg_error / (1.0 * n_fold)
        return avg_error

    def train_standard(self, X, y, lambd, n_iter):
        """
        Coordinate-descent algorithm - extended data set
        :param X: Feature Matrix
        :param y: response
        :param lambd: regularization parameter
        :param n_iter: max training iterations
        :return: fitted coefficients
        """
        self.n_rows = X.shape[0]
        if lambd == 0:
            # lambd = TOLERANCE
            logging.warning("calling regularization with zero lambda")
        self.n = X.shape[1]
        theta = np.array([np.random.normal() for j in xrange(0, self.n)])
        prev_error = 0
        denom = [np.linalg.norm(X[:, k], 2)**2 for k in xrange(0, len(theta))]
        for i in xrange(1, n_iter):
            ind = np.ones(len(theta), dtype=bool)
            for k in xrange(0, len(theta)):
                ind[k] = False
                r = y - np.dot(X[:, ind], theta[ind])
                num = np.dot(np.transpose(X[:, k]), r)
                if denom[k] == 0:
                    theta[k] = 0
                    continue
                temp = num / denom[k]
                theta[k] = self.soft_threshold(temp, (2.0*self.alpha*lambd)/denom[k])
                ind[k] = True
            errors1 = y - np.dot(X, theta)
            train_error = np.sqrt(1/(1.0*len(errors1))*sum(np.square(errors1)))
            if abs(prev_error - train_error) < TOLERANCE:
                logging.info("converged at iteration %s", i)
                break
            else:
                prev_error = train_error
        return theta

    def train(self, X, y, lambd, n_iter):
        """
        Coordinate-descent algorithm - extended data set
        :param X: Feature Matrix
        :param y: response
        :param lambd: regularization parameter
        :param n_iter: max training iterations
        :return: fitted coefficients
        """
        self.n_rows = X.shape[0]
        reg_denom = np.sqrt(1 + (0.5 * (self.alpha - self.alpha * lambd)))
        X = self.enhanced_data_matrix(X, lambd, reg_denom)    # augmented data set
        y = np.append(y, np.zeros(self.kernel.shape[0]), axis=0)    # add zeros to response
        if lambd == 0:
            logging.warning("calling regularization with zero lambda")
        self.n = X.shape[1]
        theta = np.array([np.random.normal() for j in xrange(0, self.n)])
        prev_error = 0
        denom = [np.linalg.norm(X[:, k], 2)**2 for k in xrange(0, len(theta))]
        for i in xrange(1, n_iter):
            ind = np.ones(len(theta), dtype=bool)
            for k in xrange(0, len(theta)):
                ind[k] = False
                r = y - np.dot(X[:, ind], theta[ind])
                num = np.dot(np.transpose(X[:, k]), r)
                if denom[k] == 0:
                    theta[k] = 0
                    continue
                temp = num / denom[k]
                theta[k] = self.soft_threshold(temp, ((2.0*self.alpha*lambd)/reg_denom)/denom[k])
                theta[k] = theta[k] * reg_denom
                ind[k] = True
            errors1 = y - np.dot(X, theta)
            train_error = np.sqrt(1/(1.0*len(errors1))*sum(np.square(errors1)))
            if abs(prev_error - train_error) < TOLERANCE:
                logging.info("converged at iteration %s", i)
                break
            else:
                prev_error = train_error
        return theta / reg_denom

    def enhanced_data_matrix(self, X, lambd, denom):
        """Concatenate the graph-constrained regularization to the data matrix"""
        U, s, V = np.linalg.svd(self.kernel, full_matrices=True)
        S = np.dot(U, np.sqrt(np.diagflat(s)))
        if lambd > TOLERANCE:
            S = S * np.sqrt(0.5*(self.alpha - self.alpha * lambd))
        S = np.nan_to_num(S.astype(float))
        X_star = np.append(X, S, axis=0)
        if denom > TOLERANCE:
            X_star = X_star / denom
        else:
            logging.warning("regularization parameter below threshold!")
        return X_star

    def set_alpha(self, alpha):
        self.alpha = alpha

    def soft_threshold(self, weight, lambd):
        """Derivative of l-1 regularization, i.e. *shrinkage* operator"""
        if weight > lambd:
            result = weight - lambd
        elif abs(weight) <= lambd:
            result = 0
        else:
            result = weight + lambd
        return result


class ModelManager(object):
    """Loads different data sources and kernel matrix """

    def __init__(self):
        self.featureIndex = dict()
        self.data = None
        self.kernel_laplacian = None

    def __del__(self):
        self.data = None
        self.kernel_laplacian = None

    def load_data(self, files, sep='\t'):
        dfs = []
        for f in files:
            fp = open(f, "rb")
            data = pd.read_csv(fp, sep=sep)
            dfs.append(data.loc[:2100, :])
        self.data = pd.concat(dfs, axis=1)
        self.data.rename(columns=lambda x: x.strip())
        headers = list(self.data.columns.values)
        for i, h in enumerate(headers):
            self.featureIndex[h] = i
        logging.info("Loaded data with headers %s", headers)

    def load_kernel_laplacian(self, file):
        """
        read csv file of symmetric matrix with column headers
        :param: file to be opened
        """
        fp = open(file, "rb")
        self.kernel_laplacian = pd.read_csv(fp, sep=',')
        return self.kernel_laplacian

    def get_all_features_except_response(self, response):
        logging.info("Getting all features, except: %s", response)
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

    def index_of_feature(self, feature_name):
        f = None
        try:
            f = self.featureIndex[feature_name]
        except IndexError:
            logging.error("Feature index not found.")
            pass
        return f