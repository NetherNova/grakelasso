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
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from rdflib import URIRef, RDF, ConjunctiveGraph
from sklearn.multiclass import OneVsRestClassifier
from movieetl import movie, load_mappings, extract_word_graphs, transform_labels_to_uris, load_labels
from constraints import label_ml_cons, label_ml_cons_new, label_cl_cons, label_cl_cons_new
import pickle


def parameter_search(clf, X, y):
    C_range = np.logspace(-2, 3, 5)
    gamma_range = np.logspace(-9, 2, 5)
    param_grid = {'gamma': gamma_range, 'C': C_range}
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X, y)
    print "Best Parameters: " + str(grid.best_params_)
    print "Best Score: " + str(grid.best_score_)
    return grid.best_estimator_


def dump_file(data, path, file):
    pickle.dump(data, open(path + "\\" + file, "w"))


def dump_meta_data(path, id_to_uri, graph_labels_train, graph_labels_test):
    dump_file(id_to_uri, path, "\\id_to_uri.pickle")
    dump_file(graph_labels_train, path, "\\graph_labels_train.pickle")
    dump_file(graph_labels_test, path, "\\graph_labels_test.pickle")


def load_meta_data(path):
    id_to_uri = pickle.load(open(path + "\\id_to_uri.pickle", "r"))
    graph_labels_train = pickle.load(open(path + "\\graph_labels_train.pickle", "r"))
    graph_labels_test = pickle.load(open(path + "\\graph_labels_test.pickle", "r"))
    return id_to_uri, graph_labels_train, graph_labels_test


def prepare_training_files(path, k_fold):
    print("Extracting RDF from text...")
    unique_labels = extract_word_graphs(path + "\\plot_summaries.txt", path + "\\movie.metadata.tsv", path)
    labels_mapping, num_classes, num_instances = load_mappings(path)
    filelist = []
    print "Creating %s simple graphs..." % (str(num_instances))
    output_file_test = path + "\\test"
    output_file_train = path + "\\train"
    for i in xrange(0, num_instances):
        filelist.append(path + "\\movie_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test,
                                                                           labels_mapping, k_fold, RDF.type, movie)
    dump_meta_data(path, id_to_uri, graph_labels_train, graph_labels_test)
    print "Dumped %s entity training and test files for %s folds" % (str(movie), str(k_fold))


def load_training(path):
    id_to_uri, graph_labels_train, graph_labels_test = load_meta_data(path)
    labels_mapping, num_classes, num_instances = load_mappings(path)
    unique_labels = load_labels(path)
    label_uris = transform_labels_to_uris(unique_labels)
    return id_to_uri, graph_labels_train, graph_labels_test, unique_labels, label_uris, labels_mapping, num_classes


if __name__ == '__main__':
    path = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries"
    k_fold = 4
    prepare = False
    if prepare:
        prepare_training_files(path, k_fold)
        exit(0)

    np.random.seed(24)
    min_sup = 0.01
    #c1 = svm.SVC(kernel='rbf')
    c1 = svm.SVC(kernel='poly')
    #c2 = GaussianNB()
    #c3 = KNeighborsClassifier()
    # c3 = svm.SVC(kernel='poly')

    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    #model = ["gmlc", "top-k"]
    model = ["top-k"]

    id_to_uri, graph_labels_train, graph_labels_test, unique_labels, label_uris, labels_mapping, num_classes = load_training(path)

    label_ml_cons_pairs = [#("http://wordnet-rdf.princeton.edu/wn31/mystery-n", "http://wordnet-rdf.princeton.edu/wn31/horror-n"),
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
                             ("http://wordnet-rdf.princeton.edu/wn31/romance+film-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n")]

    ml_cons = label_ml_cons_new(label_ml_cons_pairs, labels_mapping, label_uris, 0.01)

    ml_cons = list(ml_cons)
    #cl_cons = label_cl_cons([("http://wordnet-rdf.princeton.edu/wn31/comedy-n", "http://wordnet-rdf.princeton.edu/wn31/mystery-n"),
    #                         ("http://wordnet-rdf.princeton.edu/wn31/mystery-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n"),
    #                         ("http://wordnet-rdf.princeton.edu/wn31/thriller-n", "http://wordnet-rdf.princeton.edu/wn31/romantic+comedy-n")],
    #                        labels_mapping, label_uris)

    cl_cons = label_cl_cons_new(label_ml_cons_pairs, labels_mapping, label_uris, 0.01)
    cl_cons = list(cl_cons)

    output_file_test = path + "\\test"
    output_file_train = path + "\\train"

    database_train = fileio.read_file(output_file_train + str(0) + ".txt")
    statistics = gspan.database_statistics(database_train)
    print statistics

    len_ml_cons = len(ml_cons)
    len_cl_cons = len(cl_cons)
    len_combined = len_ml_cons + len_cl_cons
    num_runs = 4
    step_size = len_combined / num_runs
    print("Must-Link Constraints: " + str(len_ml_cons))
    print("Cannot-Link Constraints: " + str(len_cl_cons))

    with open(path + "\\classifiers_fair_movies.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "accuracy", "model", "runtime", "classifier"])
        for num_constraints in range(0, len_combined, step_size):
            if num_constraints < len_ml_cons:
                cons = (ml_cons[:num_constraints], [])
            else:
                cons = (ml_cons, cl_cons[:num_constraints - len_ml_cons])
            for m in model:
                print "Model: " + m
                for length in [100]:
                    scores = {names[0]: [], names[1]: [], names[2]: []}
                    times = []
                    for k in xrange(0, k_fold):
                        print "K-fold: " + str(k)
                        train_file = output_file_train + str(k) + ".txt"
                        test_file = output_file_test + str(k) + ".txt"
                        database_train = fileio.read_file(train_file)
                        print 'Number Graphs Read: ', len(database_train)
                        minsup = int((float(min_sup)*len(database_train)))
                        print "minsup: ", minsup
                        database_train, freq, trimmed, flabels = gspan.trim_infrequent_nodes(database_train, minsup)
                        database_train = fileio.read_file(train_file, frequent=freq)
                        train_labels = np.array(graph_labels_train[k])
                        if m == "top-k":
                            class_index = 0
                            tik = datetime.utcnow()
                            H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = fileio.preprocessing(database_train, class_index, labels_mapping, m)
                            X_train, pattern_set_global = gspan.project(database_train, freq, minsup, flabels, length, H,
                                                                        L, L_hat, n_graphs, n_pos, n_neg, pos_index, class_index, neg_index, graph_id_to_list_id,
                                                                        mapper=id_to_uri, labels=labels_mapping, model=m,
                                                                        constraints=cons)
                            tok = datetime.utcnow()
                            times.append((tok - tik).total_seconds())
                            X_train = gspan.database_to_matrix(database_train, pattern_set_global, id_to_uri)
                            dump_file(X_train, path, "train_topk_fold" + str(num_constraints) + "_" + str(k) + ".pickle")
                            dump_file(train_labels, path, "train_labels_topk_fold" + str(num_constraints) + "_" + str(k) + ".pickle")
                            print("Features: " + str(len(pattern_set_global)))
                            for p in pattern_set_global:
                                print p
                            database_test = fileio.read_file(test_file)
                            X_test = gspan.database_to_matrix(database_test, pattern_set_global, id_to_uri)
                            test_labels = np.array(graph_labels_test[k])
                            dump_file(X_test, path, "test_topk_fold" + str(num_constraints) + "_" + str(k) + ".pickle")
                            dump_file(test_labels, path, "test_labels_topk_fold" + str(num_constraints) + "_" + str(k) + ".pickle")

                            #for i, classifier in enumerate([c1, c2, c3]):
                            for i, classifier in enumerate([c1]):
                                clf = OneVsRestClassifier(classifier)
                                clf.fit(X_train, train_labels)
                                y_pred = clf.predict(X_test)
                                f1 = f1_score(test_labels, y_pred, average='macro')
                                scores[names[i]].append(f1)
                        else:
                            predictions_agg = {names[0]: [], names[1]: [], names[2]: []}
                            test_labels = np.array(graph_labels_test[k])
                            for class_index in xrange(num_classes):
                                print "Class Label: " + [k_tmp for k_tmp, v_tmp in unique_labels.items() if v_tmp == class_index][0]
                                tik = datetime.utcnow()
                                label_uri = label_uris[class_index]

                                H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = fileio.preprocessing(database_train, class_index, labels_mapping, m)
                                X_train, pattern_set_global = gspan.project(database_train, freq, minsup, flabels, length, H,
                                                                            L, L_hat, n_graphs, n_pos, n_neg, pos_index, class_index, neg_index, graph_id_to_list_id,
                                                                            mapper=id_to_uri, labels=labels_mapping, model=m,
                                                                            constraints=cons)
                                tok = datetime.utcnow()
                                times.append((tok - tik).total_seconds())
                                X_train = gspan.database_to_matrix(database_train, pattern_set_global, id_to_uri)
                                dump_file(X_train, path, "train_" + m + str(num_constraints) + "_fold" + str(k) + ".pickle")
                                dump_file(train_labels, path, "train_labels" + m + str(num_constraints) + "_fold" + str(k) + ".pickle")
                                print("Features: " + str(len(pattern_set_global)))
                                for p in pattern_set_global:
                                    print p
                                database_test = fileio.read_file(test_file)
                                X_test = gspan.database_to_matrix(database_test, pattern_set_global, id_to_uri)
                                dump_file(X_test, path, "test_" + m + str(num_constraints) + "_fold" + str(k) + ".pickle")
                                dump_file(test_labels, path, "test_labels" + m + str(num_constraints) + "_fold" + str(k) + ".pickle")
                                # Set train_labels only to current class_index
                                train_labels_class = train_labels[:, class_index]
                                #for i, classifier in enumerate([c1, c2, c3]):
                                for i, classifier in enumerate([c1]):
                                    if i != 0:
                                        clf = classifier
                                    else:
                                        clf = parameter_search(classifier, X_train, train_labels_class)
                                    clf.fit(X_train, train_labels_class)
                                    y_pred = clf.predict(X_test)
                                    predictions_agg[names[i]].append(y_pred)
                            #for i, classifier in enumerate([c1, c2, c3]):
                            for i, classifier in enumerate([c1]):
                                y_pred = np.array(predictions_agg[names[i]]).transpose()
                                f1 = f1_score(test_labels, y_pred, average='macro')
                                scores[names[i]].append(f1)
                    model_name = m + "+cons"
                    writer.writerow([num_constraints, np.average(scores[names[0]]), model_name, np.average(times), names[0]])
                    #writer.writerow([num_constraints, np.average(scores[names[1]]), model_name, np.average(times), names[1]])
                    #writer.writerow([num_constraints, np.average(scores[names[2]]), model_name, np.average(times), names[2]])
                    print(scores)