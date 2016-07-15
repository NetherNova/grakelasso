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

if __name__ == '__main__':
    np.random.seed(24)
    num_processes = 500
    min_sup = 0.02
    k_fold = 4
    c1 = svm.LinearSVC()
    c2 = GaussianNB()
    c3 = KNeighborsClassifier()
    names = ["Linear SVM", "Naive Bayes", "Nearest Neighbor"]
    model = ["gMGFL"] #[ "greedy", "top-k", "gMGFL"]

    cl_constraints = [
        (URIRef("http://www.siemens.com/ontology/demonstrator#Station1b"), URIRef("http://www.siemens.com/ontology/demonstrator#Conveyor1B"), URIRef("http://www.siemens.com/ontology/demonstrator#hasPart") ),

        (URIRef("http://www.siemens.com/ontology/demonstrator#Process"), URIRef("http://www.siemens.com/ontology/demonstrator#Operation/PreparationB"), URIRef("http://www.siemens.com/ontology/demonstrator#executedOperation")),

        (URIRef("http://www.siemens.com/ontology/demonstrator#Process"), URIRef("http://www.siemens.com/ontology/demonstrator#Operation/PreparationA"), URIRef("http://www.siemens.com/ontology/demonstrator#executedOperation")),

        (URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Assembly1"),URIRef("http://www.siemens.com/ontology/demonstrator#Station2"),  URIRef("http://www.siemens.com/ontology/demonstrator#usedEquipment")),

        (URIRef("http://www.siemens.com/ontology/demonstrator#ScrewDriver-A"), URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Big"), URIRef("http://www.siemens.com/ontology/demonstrator#isMadeOf")),

        (URIRef("http://www.siemens.com/ontology/demonstrator#ScrewDriver-A"), URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Small"), URIRef("http://www.siemens.com/ontology/demonstrator#isMadeOf")),

        (URIRef("http://www.siemens.com/ontology/demonstrator#Operation/WholeAssembly"),  URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Assembly1"),  URIRef("http://www.siemens.com/ontology/demonstrator#hasFollower"))
    ]


    ml_constraints = [
        (URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart"), URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin1"), URIRef("http://www.siemens.com/ontology/demonstrator#hasPart")),
        (URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart2"), URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin1"), URIRef("http://www.siemens.com/ontology/demonstrator#hasPart")),
        (URIRef("http://www.siemens.com/ontology/demonstrator#Station2"), URIRef("http://www.siemens.com/ontology/demonstrator#ToolB"), URIRef("http://www.siemens.com/ontology/demonstrator#hasProperty")),
        (URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-A"), URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone2"), URIRef("http://www.siemens.com/ontology/demonstrator#hasProperty")),
        (URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-B"), URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone2"), URIRef("http://www.siemens.com/ontology/demonstrator#hasProperty")),
        (URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Finishing"), URIRef("http://www.siemens.com/ontology/demonstrator#Operation/PreparationA"), URIRef("http://www.siemens.com/ontology/demonstrator#hasFollower"))
                ]

    dependency_matrix = np.array([ [1, 1], [0, 1] ])
    entity_dict = {"http://www.siemens.com/ontology/demonstrator#SpecialPart" : 0, "http://www.siemens.com/ontology/demonstrator#Fixture-Pin1" : 1 }

    pos_labels, product_map = simulation.execute(num_processes)
    #pos_labels = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 159, 160, 161, 162, 163, 164, 165, 167, 168, 170, 171, 172, 174, 175, 176, 177, 178, 179, 182, 183, 185, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 269, 270, 271, 272, 273, 274, 275, 277, 279, 280, 281, 282, 284, 287, 288, 289, 290, 291, 293, 294, 295, 297, 298, 299, 300, 301, 302, 304, 305, 306, 308, 309, 310, 311, 312, 313, 314, 315, 316, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 362, 363, 364, 365, 366, 367, 368, 369, 370, 372, 373, 375, 376, 377, 378, 379, 380, 381, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 419, 420, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 439, 440, 441, 442, 445, 446, 447, 448, 449, 450, 451, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 494, 495, 496, 497, 498, 499]
    print(pos_labels)
    output_file_test = "D:\\Dissertation\\Data Sets\\Manufacturing\\test"
    output_file_train = "D:\\Dissertation\\Data Sets\\Manufacturing\\train"
    filelist = []
    for i in xrange(0, num_processes):
        filelist.append("D:\\Dissertation\\Data Sets\\Manufacturing\\execution_"+str(i)+"_.rdf")
    id_to_uri, graph_labels_train, graph_labels_test = fileio.create_graph(filelist, output_file_train, output_file_test, pos_labels, k_fold, RDF.type, URIRef(u'http://www.siemens.com/ontology/demonstrator#Process'))

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
                            #
                            # if c:
                            #     cons = (cl_constraints, ml_constraints)
                            # else:
                            #     cons = ([], [])
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




