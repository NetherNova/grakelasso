__author__ = 'martin'

import unittest
from gspan import gspan
from gspan.constraints import label_ml_cons, label_cl_cons
from gspan import fileio
import numpy as np


class GspanTestCase(unittest.TestCase):
    def setUp(self):
        self.graphs = fileio.read_file("..\\data\\testgraph.txt")


class GraphInstancesTest(GspanTestCase):
    def test_io(self):
        self.assertEqual(len(self.graphs), 5)
        self.assertTrue(self.graphs[0].id == 0)
        self.assertTrue(self.graphs[1].id == 1)


class ConstraintsTest(GspanTestCase):
    def test_pairwise_constraints(self):
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


class PatternEnumTest(GspanTestCase):
    def test_basic_enum(self):
        # 5 graphs, 0-4 positive, 5 is a negative
        min_support = 1
        n_graphs = len(self.graphs)
        n_patterns = 3
        pos_index = range(0, n_graphs-1)
        n_pos = len(pos_index)
        n_neg = n_graphs - n_pos
        neg_index = [n_graphs-1]
        # identity dictionary
        graph_id_to_list = dict(zip(range(0, n_graphs), range(0, n_graphs)))
        # labels of nodes and edges (types)
        mapper = {0: "Node 1", 1: "Node 2", 2: "Node 3", 11: "Edge a", 12: "Edge b"}
        labels = {0: "Class A", 1: "Class B"}
        pattern_scoring_model = "top-k"
        # Must-link and Cannot-Link Constraints (both empty)
        instance_constraints = ([], [])
        instance_hits, patterns = gspan.project(self.graphs, [], min_support, [], n_patterns, None, None, None, n_graphs,
                                                n_pos, n_neg, pos_index, 0, neg_index, graph_id_to_list, mapper, labels,
                                                pattern_scoring_model, instance_constraints)
        X = np.array(instance_hits).transpose()
        self.assertEqual(X.shape, (n_graphs, n_patterns))


class BranchAndBoundTest(GspanTestCase):
    def test_greedy(self):
        min_support = 1
        n_graphs = len(self.graphs)
        n_patterns = 2
        pos_index = range(0, n_graphs-1)
        n_pos = len(pos_index)
        n_neg = n_graphs - n_pos
        neg_index = [n_graphs-1]
        # identity dictionary
        graph_id_to_list = dict(zip(range(0, n_graphs), range(0, n_graphs)))
        # labels of nodes and edges (types)
        mapper = {0: "Node 1", 1: "Node 2", 2: "Node 3", 11: "Edge a", 12: "Edge b"}
        labels = {0: "Class A", 1: "Class B"}
        pattern_scoring_model = "greedy"
        # Must-link and Cannot-Link Constraints (both empty)
        instance_constraints = ([], [])
        instance_hits, patterns = gspan.project(self.graphs, [], min_support, [], n_patterns, None, None, None, n_graphs,
                                                n_pos, n_neg, pos_index, 0, neg_index, graph_id_to_list, mapper, labels,
                                                pattern_scoring_model, instance_constraints)
        self.assertEquals(len(instance_hits), n_patterns)
        self.assertTrue(np.array_equal(np.array([0., 0., 0., 0., 1.]), instance_hits[0]))
        self.assertTrue(np.array_equal(np.array([1., 1., 1., 1., 0.]), instance_hits[1]))

    def test_gmgfl(self):
        min_support = 1
        n_patterns = 2
        # labels of nodes and edges (types)
        mapper = {0: "Node 1", 1: "Node 2", 2: "Node 3", 11: "Edge a", 12: "Edge b"}
        labels = {0: "Class A", 1: "Class B"}
        labels_mapping = {0: np.array([1]), 1: np.array([1]), 2: np.array([1]), 3: np.array([1]), 4: np.array([0])}
        pattern_scoring_model = "gMGFL"
        # Must-link and Cannot-Link Constraints (both empty)
        instance_constraints = ([], [])
        H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = \
            fileio.preprocessing(self.graphs, 0, labels_mapping, pattern_scoring_model)
        instance_hits, patterns = gspan.project(self.graphs, [], min_support, [], n_patterns, H, L, L_hat, n_graphs,
                                                n_pos, n_neg, pos_index, 0, neg_index, graph_id_to_list_id, mapper, labels,
                                                pattern_scoring_model, instance_constraints)
        X = np.array(instance_hits).transpose()
        self.assertEqual(X.shape, (n_graphs, n_patterns))

    def test_gmlc(self):
        min_support = 1
        n_patterns = 2
        # labels of nodes and edges (types)
        mapper = {0: "Node 1", 1: "Node 2", 2: "Node 3", 11: "Edge a", 12: "Edge b"}
        labels = {0: "Class A", 1: "Class B"}
        labels_mapping = {0: np.array([1, 0, 1]), 1: np.array([1, 0, 1]), 2: np.array([0, 0, 1]), 3: np.array([1, 0, 1]), 4: np.array([1, 1, 1])}
        pattern_scoring_model = "gMLC"
        # Must-link and Cannot-Link Constraints (both empty)
        instance_constraints = ([], [])
        H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id = \
            fileio.preprocessing(self.graphs, 0, labels_mapping, pattern_scoring_model)
        instance_hits, patterns = gspan.project(self.graphs, [], min_support, [], n_patterns, H, L, L_hat, n_graphs,
                                                n_pos, n_neg, pos_index, 0, neg_index, graph_id_to_list_id, mapper, labels,
                                                pattern_scoring_model, instance_constraints)
        X = np.array(instance_hits).transpose()
        self.assertEqual(X.shape, (n_graphs, n_patterns))


if __name__ == '__main__':
    unittest.main()