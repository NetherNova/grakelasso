__author__ = 'martin'

import unittest
from gspan import gspan
from gspan.constraints import label_ml_cons, label_cl_cons
from gspan import fileio
import numpy as np


graphs = fileio.read_file("..\\data\\testgraph.txt")


class GraphInstancesTest(unittest.TestCase):
    def test(self):
        self.assertEqual(len(graphs), 5)
        self.assertTrue(graphs[0].id == 0)
        self.assertTrue(graphs[1].id == 1)


class ConstraintsTest(unittest.TestCase):
    def test(self):
        list_of_label_pairs = [("Class A", "Class B")]
        # 4 instances with size 3 label lists
        labels_mapping = {0: np.array([0, 1, 1]), 1: np.array([0, 0, 1]), 2: np.array([1, 1, 0]), 3: np.array([1, 1, 0])}
        # label ids to their uri / string representation
        label_uris = {0: "Class A", 1: "Class B", 2: "Class C"}
        test_constraints = label_ml_cons(list_of_label_pairs, labels_mapping, label_uris)
        self.assertEqual(len(test_constraints), 1)
        self.assertEqual(test_constraints, set([(2, 3)]))

        test_constraints = label_cl_cons(list_of_label_pairs, labels_mapping, label_uris)
        self.assertEqual(test_constraints, set([(0, 1), (1, 2), (1, 3)]))


class PatternEnumTest(unittest.TestCase):
    def test(self):
        # 5 graphs, 0-4 positive, 5 is a negative
        min_support = 1
        n_graphs = len(graphs)
        n_patterns = 3
        pos_index = range(0, n_graphs)
        n_pos = len(pos_index)
        n_neg = n_graphs - n_pos
        neg_index = [n_graphs]
        # identity dictionary
        graph_id_to_list = dict(zip(range(0, n_graphs + 1), range(0, n_graphs + 1)))
        # labels of nodes and edges (types)
        mapper = {0: "Node 1", 1: "Node 2", 2: "Node 3", 11: "Edge a", 12: "Edge b"}
        labels = {0: "Class A", 1: "Class B"}
        pattern_scoring_model = "top-k"
        # Must-link and Cannot-Link Constraints (both empty)
        instance_constraints = ([], [])
        instance_hits, patterns = gspan.project(graphs, [], min_support, [], n_patterns, None, None, None, n_graphs,
                                                n_pos, n_neg, pos_index, 0, neg_index, graph_id_to_list, mapper, labels,
                                                pattern_scoring_model, instance_constraints)
        X = np.array(instance_hits).transpose()
        self.assertEqual(X.shape, (n_graphs, n_patterns))


class BranchAndBoundTest(unittest.TestCase):
    def test(self):
        pass