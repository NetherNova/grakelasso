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
    min_sup = 0.01
    k_fold = 4
    c1 = OneVsRestClassifier(svm.LinearSVC())
    c2 = OneVsRestClassifier(GaussianNB())
    c3 = OneVsRestClassifier(KNeighborsClassifier())
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = ["gMLC", "greedy", "top-k", "gMGFL"]
    path = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries"

    unique_labels = extract_word_graphs(path + "\\plot_summaries.txt", path + "\\movie.metadata.tsv", path)
    label_uris = transform_labels_to_uris(unique_labels)
    labels_mapping, num_classes, num_instances = load_mappings(path)

    ml_cons = label_ml_cons([#("http://wordnet-rdf.princeton.edu/wn31/mystery-n", "http://wordnet-rdf.princeton.edu/wn31/horror-n"),
                             #("http://wordnet-rdf.princeton.edu/wn31/thriller-n", "http://wordnet-rdf.princeton.edu/wn31/horror-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/thriller-n", "http://wordnet-rdf.princeton.edu/wn31/crime+fiction-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/mystery-n", "http://wordnet-rdf.princeton.edu/wn31/crime+fiction-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/science+fiction-n", "http://wordnet-rdf.princeton.edu/wn31/crime+fiction-n"),
                             #("http://wordnet-rdf.princeton.edu/wn31/science+fiction-n", "http://wordnet-rdf.princeton.edu/wn31/drama-n"),
                             #("http://wordnet-rdf.princeton.edu/wn31/science+fiction-n", "http://wordnet-rdf.princeton.edu/wn31/horror-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/science+fiction-n", "http://wordnet-rdf.princeton.edu/wn31/thriller-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/science+fiction-n", "http://wordnet-rdf.princeton.edu/wn31/mystery-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/comedy-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/comedy-n", "http://wordnet-rdf.princeton.edu/wn31/black+comedy-n"),
                             #("http://wordnet-rdf.princeton.edu/wn31/drama-n", "http://wordnet-rdf.princeton.edu/wn31/romance+film-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/romance+film-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n")],
                            labels_mapping, label_uris)

    cl_cons = []
    """label_cl_cons([("http://wordnet-rdf.princeton.edu/wn31/comedy-n", "http://wordnet-rdf.princeton.edu/wn31/mystery-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/mystery-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n"),
                             ("http://wordnet-rdf.princeton.edu/wn31/thriller-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n")],
                            labels_mapping, label_uris)
    """
    output_file_test = path + "\\test"
    output_file_train = path + "\\train"
    filelist = []
    for i in xrange(0, num_instances):
        filelist.append(path + "\\movie_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test,
                                                                           labels_mapping, k_fold, RDF.type, movie)

    database_train = fileio.read_file(output_file_train + str(0) + ".txt")
    statistics = gspan.database_statistics(database_train)
    print statistics

    with open(path + "\\classifiers_df_8_all_constraints.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "accuracy", "model", "runtime", "classifier"])
        for c in [True, False]:
            if c:
                cons = (ml_cons, cl_cons)
            else:
                cons = ([], [])
            for m in model:
                for length in [5, 10, 20]:
                    scores = {names[0] : [], names[1] : [], names[2] : []}
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

                        train_labels = np.array(graph_labels_train[k])
                        tik = datetime.utcnow()
                        pattern_set_global = []
                        #for class_index in xrange(num_classes):
                        class_index = 0
                        label_uri = label_uris[class_index]
                        H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = fileio.preproscessing(database_train, class_index, labels_mapping, m)
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
                        X_train = gspan.database_to_vector(database_train, pattern_set_global, id_to_uri)
                        print("Features: " + str(len(pattern_set_global)))
                        for p in pattern_set_global:
                            print p
                        database_test = fileio.read_file(test_file)
                        X_test = gspan.database_to_vector(database_test, pattern_set_global, id_to_uri)
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
                    writer.writerow([length, np.average(scores[names[0]]), model_name, np.average(times), names[0]])
                    writer.writerow([length, np.average(scores[names[1]]), model_name, np.average(times), names[1]])
                    writer.writerow([length, np.average(scores[names[2]]), model_name, np.average(times), names[2]])
                    print(scores)




