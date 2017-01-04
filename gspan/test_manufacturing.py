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
from simulation import process_uri, framepickerNew, framepickerOld, label_ml_cons, label_cl_cons

if __name__ == '__main__':
    np.random.seed(24)
    num_processes = 100
    min_sup = 0.03
    k_fold = 4
    num_classes = 6
    c1 = OneVsRestClassifier(svm.LinearSVC())
    c2 = OneVsRestClassifier(GaussianNB())
    c3 = OneVsRestClassifier(KNeighborsClassifier())
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = ["greedy", "top-k", "gMGFL"]
    path = "D:\\Dissertation\\Data Sets\\Manufacturing"

    labels_mapping = simulation.execute(num_processes, num_classes, path)

    ml_cons = label_ml_cons(labels_mapping)
    cl_cons = label_cl_cons(labels_mapping)

    fileio.write_labels_mapping(labels_mapping, path + "\\labels.csv")

    output_file_test = path + "\\test"
    output_file_train = path + "\\train"
    filelist = []
    for i in xrange(0, num_processes):
        filelist.append(path + "\\execution_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test,
                                                                           labels_mapping, k_fold, RDF.type, process_uri)

    database_train = fileio.read_file(output_file_train + str(0) + ".txt")
    statistics = gspan.database_statistics(database_train)
    print statistics

    len_ml_cons = len(ml_cons)
    len_cl_cons = len(cl_cons)
    len_combined = 2500 # len_ml_cons + len_cl_cons
    step_size = len_combined / 4

    with open("D:\\Dissertation\\Data Sets\\Manufacturing\\classifiers_break_cons.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "accuracy", "model", "runtime", "classifier"])
        for c in [True]:
            for num_constraints in range(1, len_combined, step_size):
                if num_constraints < len_ml_cons:
                    cons = (ml_cons[:num_constraints], [])
                else:
                    cons = (ml_cons, cl_cons[:num_constraints - len_ml_cons])
                for m in model:
                    for length in [5]:
                        scores = {names[0] : [], names[1] : [], names[2] : []}
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

                            train_labels = np.array(graph_labels_train[k])
                            tik = datetime.utcnow()
                            pattern_set_global = []
                            class_index = 1
                            #for class_index in xrange(num_classes):
                            H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = fileio.preprocessing(database_train, class_index, labels_mapping, m)
                            X_train, pattern_set_global = gspan.project(database_train, freq, minsup, flabels, length, H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, class_index, neg_index, graph_id_to_list_id,
                                                                     mapper = id_to_uri, labels = labels_mapping, model = m,
                                                                     constraints=cons)
                                #for p in pattern_set:
                                #    if p not in pattern_set_global:
                                #        pattern_set_global.append(p)
                                #if m == "top-k":
                                #    break
                            tok = datetime.utcnow()
                            times.append((tok - tik).total_seconds())
                            X_train = gspan.database_to_matrix(database_train, pattern_set_global, id_to_uri)
                            print("Features: " + str(len(X_train[1, :])))
                            for p in pattern_set_global:
                                print p
                            database_test = fileio.read_file(test_file)
                            X_test = gspan.database_to_matrix(database_test, pattern_set_global, id_to_uri)
                            for i, classifier in enumerate([c1, c2, c3]):
                                clf = classifier
                                clf.fit(X_train, train_labels)
                                y_pred = clf.predict(X_test)
                                test_labels = np.array(graph_labels_test[k])
                                f1 = f1_score(test_labels, y_pred)
                                scores[names[i]].append(f1)
                        if c:
                            model_name = m + "+cons"
                        else:
                            model_name = m
                        writer.writerow([num_constraints, np.average(scores[names[0]]), model_name, np.average(times), names[0]])
                        writer.writerow([num_constraints, np.average(scores[names[1]]), model_name, np.average(times), names[1]])
                        writer.writerow([num_constraints, np.average(scores[names[2]]), model_name, np.average(times), names[2]])
                        #writer.writerow([length, np.average(scores), m , np.average(times), names[i]])
                        print(scores)




