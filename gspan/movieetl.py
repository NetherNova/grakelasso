__author__ = 'martin'

from sklearn.feature_extraction.text import CountVectorizer
from rdflib import ConjunctiveGraph, URIRef, RDF, BNode, Literal
import numpy as np
import pickle
import fileio

link = URIRef("http://pub.com#link")
movie = URIRef("http://pub.com#movie")
hasName = URIRef("http://pub.com#hasName")
hasGender = URIRef("http://pub.com#hasGender")
hasAge = URIRef("http://pub.com#hasAge")
plays = URIRef("http://pub.com#plays")

restricted_label_list = ["Crime Fiction", "Mystery", "Horror", "Science Fiction", "Thriller", "Drama", "Romantic comedy",
                         "Romance Film", "Comedy", "Black comedy"]


def extract_word_graphs(text_file, meta_file, out_path):
    genres_dict = dict()
    unique_labels = dict()
    label_counter = 0
    label_mappings = dict()
    label_id_to_freebase = dict()
    with open(meta_file, "r") as f_meta:
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
                if label_stripped not in restricted_label_list:
                    continue
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
    docs = []
    inst_idx = 0
    movie_id_to_instance = dict()
    to_skip = 0
    max_lines = 1500
    line_count = 0
    # test that every instance has at least one label
    with open(text_file, "r") as f:
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
    pickle.dump(label_mappings, open(out_path + "\\label_mappings.pickle", "w"))
    count_model = CountVectorizer(ngram_range=(1,1), min_df=5, stop_words="english", max_df=1500)
    count_model.fit(docs)
    #X = csr_matrix(count_model.fit_transform(docs))
    num_inst = len(docs)
    for i in xrange(num_inst):
        convert_to_graph(docs[i], i, count_model, out_path)
    pickle.dump(unique_labels, open(out_path + "\\unique_labels.pickle", "w"))
    return unique_labels


def load_labels(path):
    return pickle.load(open(path + "\\unique_labels.pickle", "r"))


def extract_actor_graph(character_file, meta_file, out_path):
    genres_dict = dict()
    unique_labels = dict()
    label_counter = 0
    label_mappings = dict()
    label_id_to_freebase = dict()
    with open(meta_file, "r") as f_meta:
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
    with open(character_file, "r") as f:
        for line in f:
            if movie_counter == max_movies:
                break
            attributes = line.split("\t")
            m_id = attributes[0].strip()
            if current_id != m_id:
                if movie_counter == -1:
                    pass
                else:
                    g.serialize(out_path + "\\movie_" + str(movie_counter) + "_.rdf")
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
                g = actor_graph(m_id, attributes, start=True, graph=None)
            else:
                g = actor_graph(m_id, attributes, start=False, graph=g)
    pickle.dump(label_mappings, open(out_path + "\\label_mappings.pickle", "w"))
    return True


def actor_graph(m_id, attributes, start=True, graph=None):
    if start:
        g = ConjunctiveGraph()
    else:
        g = graph
    movie_ref = URIRef("http://pub.com/movie#" + str(m_id))
    g.add((movie_ref, RDF.type, movie))
    actor_bnode = BNode()
    g.add((movie_ref, link, actor_bnode))
    char_name = attributes[3].strip()
    char_gender = attributes[5].strip()
    actor_name = attributes[8].strip()
    actor_age = attributes[9].strip()
    if actor_age < 30:
        actor_age = "young"
    elif actor_age >= 30 and actor_age < 50:
        actor_age = "middle"
    else:
        actor_age = "old"
    # g.add((actor_bnode, plays, Literal(char_name)))
    g.add((actor_bnode, hasGender, Literal(char_gender)))
    g.add((actor_bnode, hasName, Literal(actor_name)))
    g.add((actor_bnode, hasAge, Literal(actor_age)))
    return g


def convert_to_graph(words, index, count_model, path):
    g = ConjunctiveGraph()
    g.add((movie, RDF.type, movie))
    word_list = [w.lower().strip() for w in words.split()]
    prev_word = None
    prev_index = 0
    for i, w in enumerate(word_list):
        if w in count_model.vocabulary_:
            word_uri = term_to_uri(w)
            if prev_word is not None and i - prev_index < 3:
                #g.add((movie, link, word_uri))
                g.add((prev_word, link, word_uri))
            prev_word = word_uri
            prev_index = i
    g.serialize(path + "\\movie_" + str(index) + "_.rdf")


def load_mappings(path):
    label_mappings = pickle.load(open(path + "\\label_mappings.pickle", "r"))
    num_classes = len(label_mappings[label_mappings.keys()[0]])
    num_instances = len(label_mappings)
    return label_mappings, num_classes, num_instances


def transform_labels_to_uris(unique_labels):
    label_uris = dict()
    for k, v in unique_labels.items():
        label_uris[v] = term_to_uri(k)
    return label_uris


def term_to_uri(term):
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
    text_file = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries\\plot_summaries.txt"
    char_file = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries\\character.metadata.tsv"
    meta_file = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries\\movie.metadata.tsv"
    out_path = "D:\\Dissertation\\Data Sets\\Movies\\MovieSummaries"
    extract_actor_graph(char_file, meta_file, out_path)
