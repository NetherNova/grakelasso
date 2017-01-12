__author__ = 'martin'

import operator
import fileio
import gspan
from datetime import datetime
import csv
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from rdflib import URIRef, RDF, ConjunctiveGraph
from sklearn.multiclass import OneVsRestClassifier
from movieetl import MovieEtl
from simulation import SimulationEtl
from constraints import label_ml_cons_new, label_cl_cons, label_cl_cons_new
import pickle
import multiprocessing as mp


def merge_dicts(d1, d2, op=operator.concat):
    return dict(d1.items() + d2.items() + [(k, op(d1[k], d2[k])) for k in set(d1) & set(d2)])


def result_write_listener(path, q):
    """
    listens for messages on the q, writes to file
    :param path:
    :param q:
    :return:
    """
    print "Writer process started..."
    now = datetime.now()
    with open(path + "\\classifiers_movies" + now.strftime("%Y-%m-%d") + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_ML_constraints", "num_CL_constraints", "num_features", "f1-score", "model", "runtime", "classifier"])
        while True:
            m = q.get()
            if m == 'kill':
                # kill the writer process
                break
            writer.writerow(m)


def run_model_experiment(etl, model, cons, k_fold, min_sup, clfs, names, max_pattern_num, q):
    """
    Run a full *k-fold* experiment with *model* and a list of classifiers *clfs*
    :param etl:
    :param model:
    :param cons:
    :param k_fold:
    :param min_sup:
    :param clfs:
    :param names:
    :param num_classes:
    :param max_pattern_num:
    :param q:
    :return:
    """
    print "Running experiment %s..." % (model)
    mapper, train_labels_all, test_labels_all, labels_mapping, num_classes = \
        etl.load_training_files()
    scores_combined = dict()
    times = []
    pattern_lengths = []
    average_method = "weighted"
    for k in xrange(k_fold):
        print "%s K-Fold: %s" % (model, k)
        train_file = etl.path + "\\train" + str(k) + ".txt"
        test_file = etl.path + "\\test" + str(k) + ".txt"
        database_train = fileio.read_file(train_file)
        print "%s Number Graphs Read: %s" % (model, len(database_train))
        abs_minsup = int((float(min_sup)*len(database_train)))
        print "%s Minsup: %s" %(model, abs_minsup)
        database_train, freq, trimmed, flabels = gspan.trim_infrequent_nodes(database_train, abs_minsup)
        database_train = fileio.read_file(train_file, frequent=freq)
        train_labels = np.array(train_labels_all[k])
        if model != "top-k":
            results = dict()
            for class_index in xrange(num_classes):
                H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = \
                    fileio.preprocessing(database_train, class_index, labels_mapping, model)
                tik = datetime.utcnow()
                X_train, pattern_set = gspan.project(database_train, freq, abs_minsup, flabels, max_pattern_num, H, L, L_hat,
                                                 n_graphs, n_pos, n_neg, pos_index, class_index, neg_index,
                                                 graph_id_to_list_id, mapper=mapper, labels=labels_mapping, model=model,
                                                 constraints=cons)
                tok = datetime.utcnow()
                times.append((tok - tik).total_seconds())
                pattern_length = len(pattern_set)
                pattern_lengths.append(pattern_length)
                X_train = gspan.database_to_matrix(database_train, pattern_set, mapper)
                database_test = fileio.read_file(test_file)
                X_test = gspan.database_to_matrix(database_test, pattern_set, mapper)
                test_labels = np.array(test_labels_all[k])
                results_class = evaluate_binary_split(X_train, train_labels, X_test, clfs, names, class_index)
                results = merge_dicts(results_class, results)
            print "%s Number of features: %s" % (model, np.average(pattern_lengths))
            scores = dict()
            for x in xrange(len(names)):
                scores[names[x]] = []
                y_pred = np.array(results[names[x]]).transpose()
                f1 = f1_score(test_labels, y_pred, average=average_method)
                scores[names[x]].append(f1)
        else:
                # special case top-k
                class_index = 0
                H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = \
                    fileio.preprocessing(database_train, class_index, labels_mapping, model)
                tik = datetime.utcnow()
                X_train, pattern_set = gspan.project(database_train, freq, abs_minsup, flabels, max_pattern_num, H, L, L_hat,
                                                 n_graphs, n_pos, n_neg, pos_index, class_index, neg_index,
                                                 graph_id_to_list_id, mapper=mapper, labels=labels_mapping, model=model,
                                                 constraints=cons)
                tok = datetime.utcnow()
                times.append((tok - tik).total_seconds())
                pattern_length = len(pattern_set)
                print "%s Number of features: %s" % (model, pattern_length)
                pattern_lengths.append(pattern_length)
                X_train = gspan.database_to_matrix(database_train, pattern_set, mapper)
                database_test = fileio.read_file(test_file)
                X_test = gspan.database_to_matrix(database_test, pattern_set, mapper)
                test_labels = np.array(test_labels_all[k])
                scores = evaluate_multilabel(X_train, train_labels, X_test, test_labels, clfs, names)
        scores_combined = merge_dicts(scores, scores_combined)
    print "%s Scores: %s" % (model, scores_combined)
    for i in xrange(len(clfs)):
        q.put([len(cons[0]), len(cons[1]), np.average(pattern_lengths), np.average(scores[names[i]]), model, np.average(times), names[i]])
    return True


def evaluate_binary_split(train, train_labels, test, clfs, names, class_index):
    if len(names) != len(clfs):
        print "Classifiers and Names must match"
        exit(0)
    results = {}
    # initialize classifier predictions with empty lists
    for i in xrange(len(names)):
        results[names[i]] = []
    for x, clf in enumerate(clfs):
        clf.fit(train, train_labels[:, class_index])
        y_pred = clf.predict(test)
        results[names[x]].append(y_pred)
    return results


def evaluate_multilabel(train, train_labels, test, test_labels, clfs, names, average_method='weighted'):
    if len(names) != len(clfs):
        print "Classifiers and Names must match"
        exit(0)
    scores = {}
    # initialize classifier predictions with empty lists
    for i in xrange(len(names)):
        scores[names[i]] = []
    for x, clf in enumerate(clfs):
        clf_tmp = OneVsRestClassifier(clf)
        clf_tmp.fit(train, train_labels)
        y_pred = clf_tmp.predict(test)
        f1 = f1_score(test_labels, y_pred, average=average_method)
        scores[names[x]].append(f1)
    return scores


if __name__ == '__main__':
    #path = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries"
    #etl = MovieEtl(path)
    path = "D:\\Dissertation\\Data Sets\\Manufacturing"
    etl = SimulationEtl(path)
    k_fold = 5

    # set True if ETL needs to run preparation first
    prepare = False
    if prepare:
        etl.prepare_training_files(k_fold)

    manager = mp.Manager()
    q = manager.Queue()
    watcher = mp.Process(target=result_write_listener, args=(path, q))
    watcher.start()

    c1 = svm.SVC()
    c2 = GaussianNB()
    c3 = KNeighborsClassifier()
    clfs = [c1, c2, c3]
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]

    models = ["top-k", "greedy", "gMGFL"]
    cons = ([], [])
    min_sup = 0.02
    max_p_num = [5, 10, 15]
    for pattern_num in max_p_num:
        jobs = []
        for model in models:
            p = mp.Process(target=run_model_experiment,
                           args=(etl, model, cons, k_fold, min_sup, clfs, names, pattern_num, q,))
            jobs.append(p)
            p.start()
        # collect results
        for job in jobs:
            job.join()
    q.put("kill")