__author__ = 'martin'

import etl
import graph
import pandas as pd
import numpy as np
from sklearn import cross_validation
import os


class AmbergEtl(etl.Etl):
    def __init__(self, path):
        super(AmbergEtl, self).__init__(path)
        self.event_filename = "WatchCAT_Meldetext_01.csv"
        self.variant_context = "variants.csv"
        self.module_context = "modules.csv"
        self.timestamp = "Timestamp"
        self.message = "Meldetext"
        self.module = "Module"
        self.variant = "MLFB"
        self.event_type_label = 10001     # should not have more than 10k different events
        self.link_label = 10000
        self.event_to_id = {}
        self.entity_to_id = {}
        self.id_to_uri = {self.event_type_label: "Event", self.link_label: "Link"}
        self.graphs = []
        self.label_mappings = {}

    def prepare_training_files(self, k_fold):
        # TODO: Parameterize to adding a) Variant b) Machine Connection Context
        # Sliding window across events log --> each window is one graph
        table = pd.read_csv(self.path + "\\" + self.event_filename, sep=";")
        table[self.timestamp] = pd.to_datetime(table[self.timestamp])
        table = table.set_index(pd.DatetimeIndex(table[self.timestamp]))
        table = table.sort_index()
        self.events_to_ids(table[self.message])
        window_size_minutes = 1
        # loop
        self.loop_through_events(table, window_size_minutes)
        # From list of graphs, create train and test data according to k_fold
        graph_labels_train, graph_labels_test = self.create_k_fold_split(k_fold)
        self.dump_meta_data(self.id_to_uri, graph_labels_train, graph_labels_test, self.label_mappings, self.event_to_id.keys())

    def load_training_files(self):
        self.load_meta_data()
        num_classes = len(self.label_mappings[self.label_mappings.keys()[0]])
        return self.id_to_uri, self.graph_labels_train, self.graph_labels_test, self.label_mappings, num_classes

    def loop_through_events(self, table, window_size_minutes):
        # windows = pd.date_range(first_row, last_row, freq="5min")
        for i in xrange(table.shape[0]):
            window_start = table.iloc[i][self.timestamp]
            window_end = window_start + pd.DateOffset(minutes=window_size_minutes)

            window_table = table[window_start:window_end]
            if len(self.graphs) < 300:
                self.add_graph(window_table)
                predictions_start = window_end + pd.DateOffset(seconds=1)   # do not include current window_end
                predictions_end = predictions_start + pd.DateOffset(minutes=window_size_minutes)
                self.generate_predictions(table[predictions_start:predictions_end], i)
            else:
                return

    def generate_predictions(self, window_table, i):
        labels = np.zeros(len(self.event_to_id), dtype=int)
        for j in xrange(window_table.shape[0]):
            row = window_table.iloc[j]
            label_index = self.event_to_id[row[self.message]]
            labels[label_index] = 1
        self.label_mappings[i] = labels

    def add_graph(self, window_table):
        g = graph.Graph()
        g.id = 0
        local_node_id_counter = 0
        # TODO: parts of variant attached
        # TODO: machines attached
        previous_event_node_id = None
        for i in xrange(window_table.shape[0]):
            row = window_table.iloc[i]
            event_node = graph.Node()
            event_node.id = local_node_id_counter
            local_node_id_counter += 1
            event_node.label = self.event_type_label
            # TODO: has event id self.event_to_id ...
            g.nodes.append(event_node)
            if previous_event_node_id:
                before_edge = graph.Edge()
                before_edge.fromn = event_node.id
                before_edge.to = previous_event_node_id
                before_edge.label = self.link_label
                before_edge.id = g.nedges
                g.nedges += 1
                event_node.edges.append(before_edge)
            previous_event_node_id = event_node.id

            timestamp, terms, module, variant = self.process_row(row)
            for term in terms:
                term_node = graph.Node()
                g.nodes.append(term_node)
                term_node.id = local_node_id_counter
                local_node_id_counter += 1
                term_node.label = self.entity_to_id[term]
                tmp_edge = graph.Edge()
                tmp_edge.fromn = event_node.id
                tmp_edge.to = term_node.id
                tmp_edge.label = self.link_label
                tmp_edge.id = g.nedges
                g.nedges += 1
                event_node.edges.append(tmp_edge)
        self.graphs.append(g)

    def events_to_ids(self, events):
        # first give unique ids to all event types
        for s in events:
            self.event_to_id.setdefault(s, len(self.event_to_id))
            for term in s.split(" "):
                # then give unique ids to all terms, starting from a max
                num = len(self.entity_to_id) + self.event_type_label + 1
                self.entity_to_id.setdefault(term, num)
                # this is the mapper
                self.id_to_uri[num] = term

    def process_row(self, row):
        timestamp = row[self.timestamp]
        event_text = row[self.message]
        module = row[self.module]
        variant = row[self.variant]
        terms = event_text.split(" ")
        return timestamp, terms, module, variant

    def create_k_fold_split(self, k_fold):
        i_fold = 0
        graph_labels_train = []
        graph_labels_test = []
        for train_index, test_index in cross_validation.KFold(len(self.graphs), n_folds=k_fold):
            train = True
            test = True
            output_train_tmp = self.path + "\\train" + str(i_fold) + ".txt"
            output_test_tmp = self.path + "\\test" + str(i_fold) + ".txt"

            # delete train and test output files
            try:
                os.remove(output_train_tmp)
            except OSError:
                pass
            try:
                os.remove(output_test_tmp)
            except OSError:
                pass
            # First round train then test
            while train or test:
                index = None
                graph_labels_tmp = []
                graph_labels_list_tmp = None
                if train:
                    index = train_index
                    output_tmp = output_train_tmp
                    train = False
                    graph_labels_list_tmp = graph_labels_train
                else:
                    index = test_index
                    output_tmp = output_test_tmp
                    test = False
                    graph_labels_list_tmp = graph_labels_test
                with open(output_tmp, "a") as tf:
                    for i in index:
                        tf.write(self.graphs[i].graph_to_string_without_id())
                        graph_labels_tmp.append(self.label_mappings[i])
                graph_labels_list_tmp.append(graph_labels_tmp)
            i_fold += 1
        return graph_labels_train, graph_labels_test

    def get_constraints(self):
        return ([], [])


if __name__ == '__main__':
    amberg_etl = AmbergEtl("D:\\Dissertation\\Data Sets\\Event Variant Amberg")
    amberg_etl.prepare_training_files(3)