__author__ = 'martin'
import pickle


class Etl(object):
    def __init__(self, path):
        self.path = path
        self.label_mappings_filename = "label_mappings.pickle"
        self.unique_labels_filename = "unique_labels.pickle"
        self.id_to_uri_filename = "id_to_uri.pickle"
        self.graph_labels_train_filename = "graph_labels_train.pickle"
        self.graph_labels_test_filename = "graph_labels_test.pickle"

    def dump_pickle_file(self, data, file_name):
        pickle.dump(data, open(self.path + "\\" + file_name, "w"))

    def load_pickle_file(self, file_name):
        return pickle.load(open(self.path + "\\" + file_name, "r"))

    def dump_meta_data(self, id_to_uri, graph_labels_train, graph_labels_test):
        self.dump_pickle_file(id_to_uri, self.id_to_uri_filename)
        self.dump_pickle_file(graph_labels_train, self.graph_labels_train_filename)
        self.dump_pickle_file(graph_labels_test, self.graph_labels_test_filename)

    def load_meta_data(self):
        id_to_uri = self.load_pickle_file(self.id_to_uri_filename)
        graph_labels_train = self.load_pickle_file(self.graph_labels_train_filename)
        graph_labels_test = self.load_pickle_file(self.graph_labels_test_filename)
        return id_to_uri, graph_labels_train, graph_labels_test

    def prepare_training_files(self, k_fold):
        """
        Creates training and test graph files with respect to k_fold.
        :param path: the folder
        :param k_fold:
        :return:
        """
        raise NotImplementedError("Abstract Class")

    def load_training_files(self):
        """
        :param path:
        :return: id_to_uri, graph_labels_train, graph_labels_test, unique_labels, label_uris, labels_mapping, num_classes
        """
        raise NotImplementedError("Abstract Class")

    def load_labels(self):
        raise NotImplementedError("Abstract Class")

    def transform_labels_to_uris(self, unique_labels):
        raise NotImplementedError("Abstract Class")