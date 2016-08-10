__author__ = 'martin'
import sys
import os

import fileio
import gspan
import simulation
from datetime import datetime
import csv
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from rdflib import URIRef, RDF
from sklearn.multiclass import OneVsRestClassifier
from simulation import process_uri, framepickerNew, framepickerOld

if __name__ == '__main__':
    np.random.seed(24)
    num_processes = 500
    min_sup = 0.005
    k_fold = 4
    num_classes = 6
    c1 = OneVsRestClassifier(svm.LinearSVC())
    c2 = OneVsRestClassifier(GaussianNB())
    c3 = OneVsRestClassifier(KNeighborsClassifier())
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = ["greedy", "top-k", "gMGFL"]
    path = "D:\\Dissertation\\Data Sets\\Manufacturing"

    labels_mapping = simulation.execute(num_processes, num_classes, path)

    output_file_test = path + "\\test"
    output_file_train = path + "\\train"
    filelist = []
    for i in xrange(0, num_processes):
        filelist.append(path + "\\execution_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test,
                                                                           labels_mapping, k_fold, RDF.type, process_uri)

    with open("D:\\Dissertation\\Data Sets\\Manufacturing\\classifiers_new_cons.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "accuracy", "model", "runtime", "classifier"])
        class_index = 0
        for m in model:
            for i, classifier in enumerate([c1, c2, c3]):
                for length in [5, 10, 20]:
                    scores = []
                    times = []
                    for k in xrange(0, k_fold):
                        train_file = output_file_train + str(k) + ".txt"
                        test_file = output_file_test + str(k) + ".txt"

                        database_train = fileio.read_file(train_file)
                        print 'Number Graphs Read: ', len(database_train)
                        minsup = int((float(min_sup)*len(database_train)))
                        print minsup

                        database_train, freq, trimmed, flabels = gspan.trim_infrequent_nodes(database_train, minsup)
                        database_train = fileio.read_file(train_file, frequent = freq)
                        cons = ([], [])

                        labels = np.array(graph_labels_train[k])
                        tik = datetime.utcnow()
                        H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = fileio.preproscessing(database_train, class_index, labels_mapping, m)

                        X_train, pattern_set = gspan.project(database_train, freq, minsup, flabels, length, H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id,
                                                             mapper = id_to_uri, labels = labels_mapping, model = m,
                                                             constraints=cons)
                        tok = datetime.utcnow()
                        times.append((tok - tik).total_seconds())
                        X_train = np.array(X_train).T
                        print("Features: " + str(len(X_train[1, :])))
                        for p in pattern_set:
                            print p
                        database_test = fileio.read_file(test_file)
                        X_test = gspan.database_to_vector(database_test, pattern_set, id_to_uri)
                        clf = classifier
                        clf.fit(X_train, labels)
                        y_pred = clf.predict(X_test)
                        labels = np.array(graph_labels_test[k])
                        f1 = f1_score(labels, y_pred)
                        scores.append(f1)
                    writer.writerow([length, np.average(scores), m , np.average(times), names[i]])
                    print(scores)




