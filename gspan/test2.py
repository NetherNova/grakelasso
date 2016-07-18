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

if __name__ == '__main__':
    np.random.seed(24)
    num_processes = 500
    min_sup = 0.02
    k_fold = 4
    c1 = OneVsRestClassifier(svm.LinearSVC())
    c2 = OneVsRestClassifier(GaussianNB())
    c3 = OneVsRestClassifier(KNeighborsClassifier())
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = ["greedy", "top-k", "gMGFL"]

    labels_mapping = simulation.execute(num_processes)
    output_file_test = "D:\\Dissertation\\Data Sets\\Manufacturing\\test"
    output_file_train = "D:\\Dissertation\\Data Sets\\Manufacturing\\train"
    filelist = []
    for i in xrange(0, num_processes):
        filelist.append("D:\\Dissertation\\Data Sets\\Manufacturing\\execution_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test, labels_mapping, k_fold, RDF.type, URIRef(u'http://www.siemens.com/ontology/demonstrator#Process'))

    with open("D:\\Dissertation\\Data Sets\\Manufacturing\\classifiers_new_cons.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "accuracy", "model", "runtime", "classifier"])
        for m in model:
            for i, classifier in enumerate([c1, c2, c3]):
                for c in [True, False]:#for c in xrange(0,11):
                    for l in [5, 10, 20]:#for l in [5]:
                        length = l
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
                            if not c:
                                product_map = None

                            labels = 1 - np.array(graph_labels_train[k])
                            tik = datetime.utcnow()
                            X_train, pattern_set = gspan.project(database_train, freq, minsup, flabels, length,
                                                                 mapper = id_to_uri, labels = labels, model = m, constraints=cons,
                                                                 dependency_matrix = dependency_matrix, entity_dict = entity_dict, product_map = product_map)
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
                        if c:
                            writer.writerow([l, np.average(scores), m + "+constraints", np.average(times), names[i]])
                        else:
                            writer.writerow([l, np.average(scores), m , np.average(times), names[i]])
                        print(scores)




