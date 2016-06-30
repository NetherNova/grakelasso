__author__ = 'martin'
from rdflib import ConjunctiveGraph, URIRef, Literal
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

# from SPARQLWrapper import SPARQLWrapper
#
# import urllib2
#
# proxy = urllib2.ProxyHandler({'http': 'http://z003d3uk:Huluvu5954@proxyfarm-mch.inac.siemens.net:84',
#                               'https': 'https://z003d3uk:Huluvu5954@proxyfarm-mch.inac.siemens.net:84'})
# auth = urllib2.HTTPBasicAuthHandler()
# opener = urllib2.build_opener(proxy, auth, urllib2.HTTPHandler)
# urllib2.install_opener(opener)


pos_movies = [
    "http://www.wikidata.org/entity/Q80379",
"http://www.wikidata.org/entity/Q116845",
"http://www.wikidata.org/entity/Q152531",
"http://www.wikidata.org/entity/Q152780",
"http://www.wikidata.org/entity/Q217020",
"http://www.wikidata.org/entity/Q243983",
"http://www.wikidata.org/entity/Q273686",
"http://www.wikidata.org/entity/Q204374",
"http://www.wikidata.org/entity/Q206124",
"http://www.wikidata.org/entity/Q206576",
"http://www.wikidata.org/entity/Q474033",
"http://www.wikidata.org/entity/Q474093",
"http://www.wikidata.org/entity/Q571032",
"http://www.wikidata.org/entity/Q732960",
"http://www.wikidata.org/entity/Q762877",
"http://www.wikidata.org/entity/Q303678",
"http://www.wikidata.org/entity/Q859448",
"http://www.wikidata.org/entity/Q1029212",
"http://www.wikidata.org/entity/Q1132978",
"http://www.wikidata.org/entity/Q1139311",
"http://www.wikidata.org/entity/Q1156089",
"http://www.wikidata.org/entity/Q1336326",
"http://www.wikidata.org/entity/Q3423831",
"http://www.wikidata.org/entity/Q4103201",
"http://www.wikidata.org/entity/Q7137004",
"http://www.wikidata.org/entity/Q28891",
"http://www.wikidata.org/entity/Q32433",
"http://www.wikidata.org/entity/Q59653",
"http://www.wikidata.org/entity/Q4941",
"http://www.wikidata.org/entity/Q28891",
"http://www.wikidata.org/entity/Q80379",
"http://www.wikidata.org/entity/Q166462",
"http://www.wikidata.org/entity/Q174385",
"http://www.wikidata.org/entity/Q205028",
"http://www.wikidata.org/entity/Q212965",
"http://www.wikidata.org/entity/Q1100064",
"http://www.wikidata.org/entity/Q1164767",
"http://www.wikidata.org/entity/Q7137004"
]


def parse_data():
    g = ConjunctiveGraph()
    f = open("C:\\Users\\z003d3uk\\Downloads\\movies_2010_2013_USA.tsv", "r")
    for l in f:
        triple = l.split("\t")
        if len(triple) != 3:
            continue
        s = URIRef(triple[0].strip())
        p = URIRef(triple[1].strip())
        if not triple[2].startswith("http"):
            o = Literal(triple[2].strip())
        else:
            o = URIRef(triple[2].strip())
        g.add((s,p,o))
    return g


def create_files(g):
    movies = g.subjects(URIRef("http://www.wikidata.org/prop/direct/P31"), URIRef("http://www.wikidata.org/entity/Q11424"))
    pos_labels = set()
    counter = 0
    pos_movie_list = [URIRef(mov) for mov in pos_movies]
    for i, m in enumerate(movies):
        list_of_actors = []
        triples = g.triples((m, None, None))
        g_temp = ConjunctiveGraph()
        if m not in pos_movie_list and np.random.random() <= 0.7:
            continue
        for t in triples:
            s = t[0]
            p = t[1]
            o = t[2]
            if p == URIRef("http://www.wikidata.org/prop/direct/P166"):
                pos_labels.add(counter)
            if s == m:
                s = URIRef("http://www.wikidata.org/entity/Q11424")
            g_temp.add((s, p, o))
            for t2 in g.triples((o, None, None)):
                s2 = t2[0]
                p2 = t2[1]
                o2 = t2[2]
                if p2 == URIRef("https://www.wikidata.org/wiki/Property:P161"):
                    o2 = URIRef("https://www.wikidata.org/wiki/Q33999")
                    list_of_actors.append(t2[2])
                if s2 in list_of_actors:
                    s2 = URIRef("https://www.wikidata.org/wiki/Q33999")
                g_temp.add((s2, p2, o2))
        g_temp.serialize("D:\\Dissertation\\Data Sets\\Movies\\movie_"+str(counter)+"_.rdf")
        counter += 1
    return pos_labels, counter

if __name__ == "__main__":
    #g = parse_data()
    #pos_labels, num_movies = create_files(g)
    pos_labels = [3, 4, 140, 13, 131, 21, 23, 152, 132, 156, 30, 33, 164, 167, 40, 171, 46, 182, 201, 56, 188, 61, 72, 73, 205, 79, 184, 83, 207, 92, 94, 99, 115, 120, 125]
    num_movies = 207
    print(pos_labels)
    k_fold = 4
    min_sup = 0.001

    c1 = svm.LinearSVC()
    c2 = GaussianNB()
    c3 = KNeighborsClassifier()
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = [ "greedy", "top-k", "gMGFL"]

    cl_constraints = [
        ("*", URIRef("http://schema.org/version"), "*")
   ]

    ml_constraints = [
        ("*", URIRef("http://www.wikidata.org/prop/direct/P1411"), "*")
    ]

    output_file_test = "D:\\Dissertation\\Data Sets\\Movies\\test"
    output_file_train = "D:\\Dissertation\\Data Sets\\Movies\\train"
    filelist = []
    for i in xrange(0, num_movies):
        filelist.append("D:\\Dissertation\\Data Sets\\Movies\\movie_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test, pos_labels, k_fold,
                                                                           URIRef("http://www.wikidata.org/prop/direct/P31"),
                                                                           URIRef("http://www.wikidata.org/entity/Q11424"))

    with open("D:\\Dissertation\\Data Sets\\Movies\\classifier.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "accuracy", "model", "runtime", "classifier"])
        for m in model:
            for i, classifier in enumerate([c1, c2, c3]):
                for c in [True, False]:     #for c in xrange(0,11):
                    for l in [1, 2, 3, 4]:      #for l in [5]:
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
                            if c:
                                cons = (cl_constraints, ml_constraints)
                            else:
                                cons = ([], [])
                            labels = np.array(graph_labels_train[k])
                            tik = datetime.utcnow()
                            X_train, pattern_set = gspan.project(database_train, freq, minsup, flabels, length, mapper = id_to_uri, labels = labels, model = m, constraints=cons)
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
                        if c:
                            writer.writerow([l, np.average(scores), m + "+constraints", np.average(times), names[i]])
                        else:
                            writer.writerow([l, np.average(scores), m , np.average(times), names[i]])
                        print(scores)
