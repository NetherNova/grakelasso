__author__ = 'martin'


class Etl(object):
    label_mappings_filename = "label_mappings.pickle"

    def __init__(self):
        self.filename = ""

    def prepare_training_files(self, path, k_fold):
        """
        Creates training and test graph files with respect to k_fold.
        :param path: the folder
        :param k_fold:
        :return:
        """
        pass

    def load_training_files(self, path):
        """

        :param path:
        :return: id_to_uri, graph_labels_train, graph_labels_test, unique_labels, label_uris, labels_mapping, num_classes
        """
        pass