__author__ = 'martin'

import numpy as np


# TODO: create constraints from (rdf)-graph


def map_label_pair_to_ids(label1, label2, label_uris):
    ind1 = None
    ind2 = None
    for k, v in label_uris.iteritems():
            if label1 == str(v):
                ind1 = k
            if label2 == str(v):
                ind2 = k
    return ind1, ind2


def label_ml_cons(list_of_label_pairs, labels_mapping, label_uris):
    """ML-constraint holds if all labels share a link or common ancestor"""
    set_of_ml_pairs = set()
    set_of_label_pairs = set()
    for i, (l1, l2) in enumerate(list_of_label_pairs):
        ind1, ind2 = map_label_pair_to_ids(l1, l2, label_uris)
        set_of_label_pairs.add((ind1, ind2))
    for i, item in enumerate(labels_mapping.items()):
        labels1 = item[1] # value of dictionary is binary vector
        labels1 = labels1.nonzero()[0]
        # start second loop at i-th position
        for item2 in labels_mapping.items()[i+1:]:
            labels2 = item2[1]
            labels2 = labels2.nonzero()[0]
            add = True
            if len(labels1) == 1 and len(labels2) == 1 and np.all(labels1 == labels2):
                set_of_ml_pairs.add((item[0], item2[0]))
                continue
            for label1 in labels1:
                for label2 in labels2:
                    if label1 == label2:
                        continue
                    if (label1, label2) not in set_of_label_pairs and (label2, label1) not in set_of_label_pairs:
                        # as soon as one label combination fails
                        add = False
                        break
            if not add:
                continue
            else:
                set_of_ml_pairs.add((item[0], item2[0]))
    return set_of_ml_pairs


def label_ml_cons_new(list_of_label_pairs, labels_mapping, label_uris, relax_coeff):
    """ML-constraint perfect match strategy"""
    set_of_ml_pairs = set()
    set_of_label_pairs = set()
    for i, (l1, l2) in enumerate(list_of_label_pairs):
        ind1, ind2 = map_label_pair_to_ids(l1, l2, label_uris)
        set_of_label_pairs.add((ind1, ind2))
        # list_of_label_pairs[i] = (ind1, ind2)
    for i, item in enumerate(labels_mapping.items()):
        labels1 = item[1] # value of dictionary is binary vector
        labels1 = labels1.nonzero()[0]
        # start second loop at i-th position
        for item2 in labels_mapping.items()[i+1:]:
            labels2 = item2[1]
            labels2 = labels2.nonzero()[0]
            if len(labels2) != len(labels1):
                continue
            elif len(labels2) == 1 and len(labels1) == 1:
                continue
            elif np.all(labels1 == labels2) and np.random.random() < relax_coeff:
                set_of_ml_pairs.add((item[0], item2[0]))
    return set_of_ml_pairs


def label_cl_cons(list_of_label_pairs, labels_mapping, label_uris):
    """ML-constraint holds if all labels share a link or common ancestor"""
    set_of_cl_pairs = set()
    set_of_label_pairs = set()
    for i, (l1, l2) in enumerate(list_of_label_pairs):
        ind1, ind2 = map_label_pair_to_ids(l1, l2, label_uris)
        set_of_label_pairs.add((ind1, ind2))
    for i, item in enumerate(labels_mapping.items()):
        labels1 = item[1] # value of dictionary is binary vector
        labels1 = labels1.nonzero()[0]
        # start second loop at i-th position
        for item2 in labels_mapping.items()[i+1:]:
            labels2 = item2[1]
            labels2 = labels2.nonzero()[0]
            add = True
            if len(labels1) == 1 and len(labels2) == 1 and np.all(labels1 == labels2):
                set_of_cl_pairs.add((item[0], item2[0]))
                continue
            for label1 in labels1:
                for label2 in labels2:
                    if label1 == label2:
                        continue
                    if (label1, label2) in set_of_label_pairs or (label2, label1) in set_of_label_pairs:
                        # as soon as one label combination holds
                        add = False
                        break
            if not add:
                continue
            else:
                set_of_cl_pairs.add((item[0], item2[0]))
    return set_of_cl_pairs


def label_cl_cons_new(list_of_label_pairs, labels_mapping, label_uris, relax_coeff):
    """ML-constraint holds if all labels share a link or common ancestor"""
    set_of_cl_pairs = set()
    set_of_label_pairs = set()
    for i, (l1, l2) in enumerate(list_of_label_pairs):
        ind1, ind2 = map_label_pair_to_ids(l1, l2, label_uris)
        set_of_label_pairs.add((ind1, ind2))
        # list_of_label_pairs[i] = (ind1, ind2)
    for i, item in enumerate(labels_mapping.items()):
        labels1 = item[1] # value of dictionary is binary vector
        labels1 = labels1.nonzero()[0]
        # start second loop at i-th position
        for item2 in labels_mapping.items()[i+1:]:
            labels2 = item2[1]
            labels2 = labels2.nonzero()[0]
            add = True
            perfect_match = np.all(labels1 == labels2)
            if perfect_match:
                continue
            for x, label1 in enumerate(labels1):
                for label2 in labels2:
                    if label1 == label2:
                        continue
                    if (label1, label2) in set_of_label_pairs or (label2, label1) in set_of_label_pairs:
                        # as soon as one label combination fails
                        add = False
                        break
            if not add:
                continue
            elif np.random.random() < relax_coeff:
                set_of_cl_pairs.add((item[0], item2[0]))
    return set_of_cl_pairs