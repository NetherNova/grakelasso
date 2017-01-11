__author__ = 'martin'

from sklearn.feature_extraction.text import CountVectorizer
from rdflib import ConjunctiveGraph, URIRef, RDF, BNode, Literal
import numpy as np
import pickle
import fileio
import etl


class MovieEtl(etl.Etl):
    def __init__(self, path):
        super(MovieEtl, self).__init__(path)
        self.summaries_filename = "plot_summaries.txt"
        self.movie_filename = "movie.metadata.tsv"
        self.character_filename = "character.metadata.tsv"
        self.link = URIRef("http://pub.com#link")
        self.movie = URIRef("http://pub.com#movie")
        self.hasName = URIRef("http://pub.com#hasName")
        self.hasGender = URIRef("http://pub.com#hasGender")
        self.hasAge = URIRef("http://pub.com#hasAge")
        self.plays = URIRef("http://pub.com#plays")
        self.restricted_label_list = ["Crime Fiction", "Mystery", "Horror", "Science Fiction", "Thriller", "Drama",
                                      "Romantic comedy", "Romance Film", "Comedy", "Black comedy"]

    def prepare_training_files(self, k_fold):
        """
        Creates training and test graph files with respect to k_fold.
        :param self.path: the folder
        :param k_fold:
        :return:
        """
        print "Extracting RDF from text..."
        unique_labels = self.extract_word_graphs()
        labels_mapping, num_classes, num_instances = self.load_mappings()
        filelist = []
        print "Creating %s simple graphs..." % (str(num_instances))
        output_file_test = self.path + "\\test"
        output_file_train = self.path + "\\train"
        for i in xrange(0, num_instances):
            filelist.append(self.path + "\\movie_"+str(i)+"_.rdf")
        id_to_uri, graph_labels_train, graph_labels_test = \
            fileio.create_graph(filelist, output_file_train, output_file_test, labels_mapping, k_fold, RDF.type, self.movie)
        self.dump_meta_data(id_to_uri, graph_labels_train, graph_labels_test)
        print "Dumped %s entity training and test files for %s folds" % (str(self.movie), str(k_fold))

    def load_training_files(self):
        """
        :param self.path:
        :return: id_to_uri, graph_labels_train, graph_labels_test, unique_labels, label_uris, labels_mapping, num_classes
        """
        id_to_uri, graph_labels_train, graph_labels_test = self.load_meta_data()
        labels_mapping, num_classes, num_instances = self.load_mappings()
        return id_to_uri, graph_labels_train, graph_labels_test, labels_mapping, num_classes

    def load_labels(self):
        return pickle.load(open(self.path + "\\" + self.unique_labels_filename, "r"))

    def load_mappings(self):
        label_mappings = pickle.load(open(self.path + "\\" + self.label_mappings_filename, "r"))
        num_classes = len(label_mappings[label_mappings.keys()[0]])
        num_instances = len(label_mappings)
        return label_mappings, num_classes, num_instances

    def extract_word_graphs(self):
        min_document_freq = 5
        max_document_freq = 1500
        max_lines = 1500
        genres_dict = dict()
        unique_labels = dict()
        label_counter = 0
        label_mappings = dict()
        label_id_to_freebase = dict()
        # get labels list (genres) for each movie
        with open(self.path + "\\" + self.movie_filename, "r") as f_meta:
            for line in f_meta:
                genres = line.split("\t")[8]
                m_id = line.split("\t")[0]
                genre_list = genres.split(",")
                label_list = []
                for genre in genre_list:
                    genre = genre.replace('"', '')
                    genre = genre.replace("}", "")
                    genre = genre.replace("{", "")
                    item_list = genre.split(":")
                    if len(item_list) != 2:
                        continue
                    freebase_id, label = item_list[0], item_list[1]
                    freebase_id_stripped = freebase_id.strip()
                    label_stripped = label.strip()
                    if label_stripped not in self.restricted_label_list:
                        continue
                    if label_stripped in unique_labels:
                        label_id = unique_labels[label_stripped]
                    else:
                        unique_labels[label_stripped] = label_counter
                        label_id = label_counter
                        label_counter += 1
                    if label_id in label_id_to_freebase:
                        test_freebase = label_id_to_freebase[label_id]
                    else:
                        label_id_to_freebase[label_id] = freebase_id_stripped
                    label_list.append(label_id)
                genres_dict[m_id] = label_list
        num_classes = len(unique_labels)
        docs = []
        inst_idx = 0
        movie_id_to_instance = dict()
        to_skip = 0
        line_count = 0
        # test that every instance has at least one label
        with open(self.path + "\\" + self.summaries_filename, "r") as f:
            for line in f:
                line_count += 1
                if line_count < to_skip:
                    continue
                if inst_idx >= max_lines:
                    break
                id, text = line.split("\t")
                movie_id_to_instance[id.strip()] = inst_idx
                try:
                    genre_list = genres_dict[id]
                except KeyError:
                    continue
                class_vector = np.zeros(num_classes)
                for g in genre_list:
                    class_vector[g] = 1
                if np.all(class_vector == 0):
                    continue
                label_mappings[inst_idx] = class_vector
                docs.append(text)
                inst_idx += 1
        self.dump_pickle_file(label_mappings, self.label_mappings_filename)
        count_model = CountVectorizer(ngram_range=(1,1), min_df=min_document_freq, stop_words="english", max_df=max_document_freq)
        count_model.fit(docs)
        num_inst = len(docs)
        for i in xrange(num_inst):
            self.convert_to_graph(docs[i], i, count_model)
        self.dump_pickle_file(unique_labels, self.unique_labels_filename)
        return unique_labels

    def extract_actor_graph(self):
        genres_dict = dict()
        unique_labels = dict()
        label_counter = 0
        label_mappings = dict()
        label_id_to_freebase = dict()
        with open(self.path + "\\" + self.movie_filename, "r") as f_meta:
            for line in f_meta:
                genres = line.split("\t")[8]
                m_id = line.split("\t")[0]
                genre_list = genres.split(",")
                label_list = []
                for genre in genre_list:
                    genre = genre.replace('"', '')
                    genre = genre.replace("}", "")
                    genre = genre.replace("{", "")
                    item_list = genre.split(":")
                    if len(item_list) != 2:
                        continue
                    freebase_id, label = item_list[0], item_list[1]
                    freebase_id_stripped = freebase_id.strip()
                    label_stripped = label.strip()
                    try:
                        label_id = unique_labels[label_stripped]
                    except KeyError:
                        unique_labels[label_stripped] = label_counter
                        label_id = label_counter
                        label_counter += 1
                    try:
                        test_freebase = label_id_to_freebase[label_id]
                    except KeyError:
                        label_id_to_freebase[label_id] = freebase_id_stripped
                    label_list.append(label_id)
                genres_dict[m_id] = label_list
        num_classes = len(unique_labels)
        current_id = None
        g = ConjunctiveGraph()
        movie_counter = -1
        max_movies = 100
        with open(self.path + "\\" + self.character_filename, "r") as f:
            for line in f:
                if movie_counter == max_movies:
                    break
                attributes = line.split("\t")
                m_id = attributes[0].strip()
                if current_id != m_id:
                    if movie_counter == -1:
                        pass
                    else:
                        g.serialize(self.path + "\\movie_" + str(movie_counter) + "_.rdf")
                        try:
                            genre_list = genres_dict[m_id]
                        except KeyError:
                            continue
                        class_vector = np.zeros(num_classes)
                        for g in genre_list:
                            class_vector[g] = 1
                        label_mappings[movie_counter] = class_vector
                    movie_counter += 1
                    current_id = m_id
                    g = self.actor_graph(m_id, attributes, start=True, graph=None)
                else:
                    g = self.actor_graph(m_id, attributes, start=False, graph=g)
        pickle.dump(label_mappings, open(self.path + "\\" + self.label_mappings_filename, "w"))

    def actor_graph(self, m_id, attributes, start=True, graph=None):
        if start:
            g = ConjunctiveGraph()
        else:
            g = graph
        movie_ref = URIRef("http://pub.com/movie#" + str(m_id))
        g.add((movie_ref, RDF.type, self.movie))
        actor_bnode = BNode()
        g.add((movie_ref, self.link, actor_bnode))
        char_gender = attributes[5].strip()
        actor_name = attributes[8].strip()
        actor_age = attributes[9].strip()
        if actor_age < 30:
            actor_age = "young"
        elif actor_age >= 30 and actor_age < 50:
            actor_age = "middle"
        else:
            actor_age = "old"
        g.add((actor_bnode, self.hasGender, Literal(char_gender)))
        g.add((actor_bnode, self.hasName, Literal(actor_name)))
        g.add((actor_bnode, self.hasAge, Literal(actor_age)))
        return g

    def convert_to_graph(self, words, index, count_model):
        g = ConjunctiveGraph()
        g.add((self.movie, RDF.type, self.movie))
        word_list = [w.lower().strip() for w in words.split()]
        prev_word = None
        prev_index = 0
        for i, w in enumerate(word_list):
            if w in count_model.vocabulary_:
                word_uri = self.term_to_wordnet_uri(w)
                if prev_word is not None and i - prev_index < 3:
                    g.add((prev_word, self.link, word_uri))
                prev_word = word_uri
                prev_index = i
        g.serialize(self.path + "\\movie_" + str(index) + "_.rdf")

    def transform_labels_to_uris(self, unique_labels):
        label_uris = dict()
        for k, v in unique_labels.items():
            label_uris[v] = self.term_to_wordnet_uri(k)
        return label_uris

    def term_to_wordnet_uri(self, term):
        base_uri = "http://wordnet-rdf.princeton.edu/wn31/"
        list_of_words = term.split()
        num_words = len(list_of_words)
        temp_uri = base_uri
        if len(list_of_words) > 1:
            for i, word in enumerate(list_of_words):
                word = word.lower()
                if i == num_words - 1:
                    temp_uri += word + "-n"
                else:
                    temp_uri += word + "+"
            return URIRef(temp_uri)
        else:
            return URIRef(base_uri + term.lower() + "-n")


if __name__ == '__main__':
    out_path = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries"
    movie_etl = MovieEtl(out_path)
    movie_etl.prepare_training_files(5)
