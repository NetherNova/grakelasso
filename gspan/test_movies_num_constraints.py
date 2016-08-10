
__author__ = 'martin'

import fileio
import gspan
from datetime import datetime
import csv
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from rdflib import URIRef, RDF, ConjunctiveGraph
from sklearn.multiclass import OneVsRestClassifier
from movieetl import movie, load_mappings, extract_word_graphs, transform_labels_to_uris, label_ml_cons, label_cl_cons

if __name__ == '__main__':
    np.random.seed(24)
    min_sup = 0.008
    k_fold = 4
    c1 = OneVsRestClassifier(svm.LinearSVC())
    c2 = OneVsRestClassifier(GaussianNB())
    c3 = OneVsRestClassifier(KNeighborsClassifier())
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = ["greedy", "top-k", "gMGFL"]
    path = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries"

    unique_labels = extract_word_graphs(path + "\\plot_summaries.txt", path + "\\movie.metadata.tsv", path)
    label_uris = transform_labels_to_uris(unique_labels)
    labels_mapping, num_classes, num_instances = load_mappings(path)

    ml_cons = label_ml_cons([("http://wordnet-rdf.princeton.edu/wn31/comedy-n", "http://wordnet-rdf.princeton.edu/wn31/black+comedy-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/comedy-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/drama-n", "http://wordnet-rdf.princeton.edu/wn31/romance+film-n")],
                            labels_mapping, label_uris)

    cl_cons = label_cl_cons([("http://wordnet-rdf.princeton.edu/wn31/comedy-n", "http://wordnet-rdf.princeton.edu/wn31/mystery-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/mystery-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/thriller-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n")],
                            labels_mapping, label_uris)

    kg = ConjunctiveGraph()
    #kg.load("D:\\Dissertation\\Data sets\\Movies\\wordnet_tests.nt", format="nt")

    output_file_test = path + "\\test"
    output_file_train = path + "\\train"
    filelist = []
    for i in xrange(0, num_instances):
        filelist.append(path + "\\movie_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test,
                                                                           labels_mapping, k_fold, RDF.type, movie)

    len_ml_cons = len(ml_cons)
    len_cl_cons = len(cl_cons)
    len_combined = len_ml_cons + len_cl_cons
    step_size = len_combined / 5
    with open(path + "\\classifiers_df_8.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "accuracy", "model", "runtime", "classifier"])
        for c in [True]:
            for num_constraints in range(1, len_combined, step_size):
                if num_constraints < len_ml_cons:
                    cons = (ml_cons[:num_constraints], [])
                else:
                    cons = (ml_cons, cl_cons[:num_constraints - len_ml_cons])
                for m in model:
                    for i, classifier in enumerate([c1, c2, c3]):
                        for length in [20]:
                            scores = []
                            times = []
                            for k in xrange(0, k_fold):
                                train_file = output_file_train + str(k) + ".txt"
                                test_file = output_file_test + str(k) + ".txt"

                                database_train = fileio.read_file(train_file)
                                print 'Number Graphs Read: ', len(database_train)
                                minsup = int((float(min_sup)*len(database_train)))
                                print "minsup: ", minsup

                                database_train, freq, trimmed, flabels = gspan.trim_infrequent_nodes(database_train, minsup)
                                database_train = fileio.read_file(train_file, frequent = freq)

                                labels = np.array(graph_labels_train[k])
                                tik = datetime.utcnow()
                                pattern_set_global = []
                                for class_index in xrange(num_classes):
                                    label_uri = label_uris[class_index]
                                    H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = fileio.preproscessing(database_train, class_index, labels_mapping, m)
                                    X_train, pattern_set = gspan.project(database_train, freq, minsup, flabels, length, H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id,
                                                                         mapper = id_to_uri, labels = labels_mapping, model = m,
                                                                         constraints=cons, kg = kg, label_uri = label_uri)
                                    for p in pattern_set:
                                        if p not in pattern_set_global:
                                            pattern_set_global.append(p)
                                    if m == "top-k":
                                        break
                                tok = datetime.utcnow()
                                times.append((tok - tik).total_seconds())
                                #X_train = np.array(X_train).T
                                X_train = gspan.database_to_vector(database_train, pattern_set_global, id_to_uri)
                                print("Features: " + str(len(pattern_set_global)))
                                for p in pattern_set_global:
                                    print p
                                #X_train = gspan.path_features(X_train, labels, pattern_set_global, kg)
                                database_test = fileio.read_file(test_file)
                                X_test = gspan.database_to_vector(database_test, pattern_set_global, id_to_uri)
                                clf = classifier
                                clf.fit(X_train, labels)
                                y_pred = clf.predict(X_test)
                                labels = np.array(graph_labels_test[k])
                                f1 = f1_score(labels, y_pred)
                                scores.append(f1)
                            if c:
                                model_name = m + "+cons"
                            else:
                                model_name = m
                            writer.writerow([num_constraints, np.average(scores), model_name, np.average(times), names[i]])
                            print(scores)




