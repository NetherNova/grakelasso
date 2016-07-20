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
    num_processes = 50
    min_sup = 0.02
    k_fold = 4
    num_classes = 6
    c1 = OneVsRestClassifier(svm.LinearSVC())
    c2 = OneVsRestClassifier(GaussianNB())
    c3 = OneVsRestClassifier(KNeighborsClassifier())
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = ["knowledge-graph", "top-k", "gMGFL"]
    path = "D:\\Dissertation\\Data Sets\\Manufacturing"

    labels_mapping = simulation.execute(num_processes, num_classes, path)
    graph_entities = [framepickerNew, framepickerOld]

    output_file_test = "D:\\Dissertation\\Data Sets\\Manufacturing\\test"
    output_file_train = "D:\\Dissertation\\Data Sets\\Manufacturing\\train"
    filelist = []
    for i in xrange(0, num_processes):
        filelist.append("D:\\Dissertation\\Data Sets\\Manufacturing\\execution_"+str(i)+"_.rdf")
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

                        L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = fileio.preproscessing(database_train, class_index, labels_mapping)

                        X_train, pattern_set = gspan.project(database_train, freq, minsup, flabels, length, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id,
                                                             mapper = id_to_uri, labels = labels_mapping, model = m,
                                                             constraints=cons, graph_entities = graph_entities)
                        # matrix M = wie stark korrespondiert operation execution i mit exeuction j
                        # constraints - welche beiden instanzen sollte man nicht vergleichen?
                        # basierend auf distanz der labels und indikatoren der graph patterns
                        # ähnlichkeit von patterns - so wie kernel im paper (ist aber nur dot-product kernel auf indicators)?
                        # causation-artiger kernel k(G_i, G_j) = f_g_i  f_g_j
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
                        labels = 1 - np.array(graph_labels_test[k])
                        f1 = f1_score(labels, y_pred)
                        scores.append(f1)
                        writer.writerow([length, np.average(scores), m , np.average(times), names[i]])
                    print(scores)




