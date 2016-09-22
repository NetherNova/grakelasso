import graph
import sys
import os
import csv
from rdflib import ConjunctiveGraph, URIRef, RDF, BNode
import numpy as np
from collections import defaultdict
from sklearn import cross_validation
from simulation import ID
from sklearn.metrics.pairwise import pairwise_kernels

def read_file(filename, frequent=[]):
	ret = []

	count = 0
	id_map = {}
	node_id = 0
	labels = []
	
	for line in open(filename, 'r'):
		if line[0] == 't':
			if count > 0:
				ret.append(g)
				id_map = {}
				labels = []
				node_id = 0
				edge_id = 0

			g = graph.Graph()
			g.id = count
			count += 1
			continue

		if line[0] == 'v':
			nid, label = [int(x) for x in line.split()[1:]]
			labels.append(label)
			if len(frequent) > 0:
				if label not in frequent:
					#print 'deleting node',g.id,nid,label
					continue
				
			n = graph.Node()
			n.id = node_id
			n.label = label
			id_map[nid] = node_id
			g.nodes.append(n)
			node_id += 1
			continue

		if line[0] == 'e':
			fromn, to, label = [int(x) for x in line.split()[1:]]
			label_from = labels[fromn]
			label_to = labels[to]
			if len(frequent) > 0:
				if label_from not in frequent or \
						label_to not in frequent:
					#print 'deleting edge',g.id, fromn, to, label
					continue

			e = graph.Edge()
			e.id = g.nedges
			g.nedges += 1
			e.fromn = id_map[fromn]
			e.to = id_map[to]
			e.label = label

			g.nodes[e.fromn].edges.append(e)

			e2 = graph.Edge()
			e2.fromn = e.to
			e2.to = e.fromn
			e2.label = e.label
			e2.id = e.id
			g.nodes[e.to].edges.append(e2)		

	ret.append(g)
	return ret

global entity_counter
global relation_counter

def create_graph(filelist, output_train, output_test, pos_graphs, cv, predicate, ob):
	global relation_counter
	relation_counter = 1000000
	global entity_counter
	global local_entity_counter
	global local_entity_map
	global id_to_uri
	id_to_uri = dict()

	entity_counter = 0
	entity_map = dict()
	relation_map = dict()
	graph_labels_train = []
	graph_labels_test = []
	filelist = np.array(filelist)
	i_fold = 0
	for train_index, test_index in cross_validation.KFold(len(filelist), n_folds=cv):
		train = True
		test = True
		filelist_train = filelist[train_index]
		filelist_test = filelist[test_index]

		output_train_tmp = output_train + str(i_fold) + ".txt"
		output_test_tmp = output_test + str(i_fold) + ".txt"

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
			graph_labels_tmp = []
			filelist_tmp = None
			graph_labels_list_tmp = None
			if train:
				filelist_tmp = filelist_train
				output_tmp = output_train_tmp
				train = False
				graph_labels_list_tmp = graph_labels_train
			else:
				filelist_tmp = filelist_test
				output_tmp = output_test_tmp
				test = False
				graph_labels_list_tmp = graph_labels_test
			for f in filelist_tmp:
				num = int(f.split("_")[1])
				labels = pos_graphs[num]
				graph_labels_tmp.append(labels)
				g = ConjunctiveGraph()
				g.load(open(f, "rb"))
				operations = list(g.subjects(predicate, ob))
				with open(output_tmp, "a") as tf:
					o = operations[0]
					entity_set = set()
					edge_set = []
					local_entity_counter = 0
					local_entity_map = []
					local_entity_counter = 0
					local_entity_map = dict()
					dfs_triples(entity_set, entity_map, edge_set, relation_map, g, o)
					#id = list(g.objects(o, ID))[0]
					tf.write("t")
					tf.write("\n")
					for (local_id, global_id) in sorted(entity_set, key=lambda x: x[0]):
						tf.write("v" + " " + str(local_id) + " " + str(global_id))
						tf.write("\n")
					for (s,p,o) in edge_set:
						tf.write("e" + " " + str(s) + " " + str(o) + " " + str(p))
						tf.write("\n")
			graph_labels_list_tmp.append(graph_labels_tmp)
		i_fold += 1
	return id_to_uri, graph_labels_train, graph_labels_test

def dfs_triples(entity_set, entity_map, edge_set, relation_map, graph, subject):
	global relation_counter
	global entity_counter
	global local_entity_counter
	global local_entity_map
	global id_to_uri

	triples = graph.triples((None, None, None))
	for (s,p,o) in triples:
		s_global = None
		o_global = None
		if p == RDF.type:
			continue
		if s.startswith("N"):
			s_global = "Blank"
		if o.startswith("N"):
			o_global = "Blank"
		if s_global:
			try:
				s_id = entity_map[s_global]
			except KeyError:
				entity_map[s_global] = entity_counter
				s_id = entity_counter
				entity_counter += 1
				id_to_uri[s_id] = s_global
		else:
			try:
				s_id = entity_map[s]
			except KeyError:
				entity_map[s] = entity_counter
				s_id = entity_counter
				entity_counter += 1
				id_to_uri[s_id] = s
		if o_global:
			try:
				o_id = entity_map[o_global]
			except KeyError:
				entity_map[o_global] = entity_counter
				o_id = entity_counter
				entity_counter += 1
				id_to_uri[o_id] = o_global
		else:
			try:
				o_id = entity_map[o]
			except KeyError:
				entity_map[o] = entity_counter
				o_id = entity_counter
				entity_counter += 1
				id_to_uri[o_id] = o
		try:
			p_id = entity_map[p]
		except KeyError:
			entity_map[p] = relation_counter
			p_id = relation_counter
			relation_counter += 1
			id_to_uri[p_id] = p
		try:
			s_local_id = local_entity_map[s]
		except KeyError:
			local_entity_map[s] = local_entity_counter
			s_local_id = local_entity_counter
			local_entity_counter += 1
		try:
			o_local_id = local_entity_map[o]
		except KeyError:
			local_entity_map[o] = local_entity_counter
			o_local_id = local_entity_counter
			local_entity_counter += 1
					# no circles
		if (s_local_id, p_id, o_local_id) in edge_set: 	# no circles
			return
		entity_set.add( (s_local_id, s_id) )
		entity_set.add( (o_local_id, o_id) )
		edge_set.append((s_local_id, p_id, o_local_id))
		#dfs_triples(entity_set, entity_map, edge_set, relation_map, graph, o)


def load_matlab(filename, key):
	mat = scipy.io.loadmat(filename)
	labels = mat[l + key.lower()]
	for i, inst in enumerate(mat[key]):
		adj = mat[key][0][i][0]
		edge = None


# type can be one of ['r' (default), 'o']
def propositionalize_rdf(rdf_files, output_train, output_test, pos_graphs, k_fold, type="r"):
	graph_labels_train = []
	graph_labels_test = []
	triple_counter = 0
	triple_dict = defaultdict(int)

	rdf_files = np.array(rdf_files)
	i_fold = 0
	for train_index, test_index in cross_validation.KFold(len(rdf_files), n_folds=k_fold):
		train = True
		test = True
		filelist_train = rdf_files[train_index]
		filelist_test = rdf_files[test_index]

		output_train_tmp = output_train + str(i_fold) + ".txt"
		output_test_tmp = output_test + str(i_fold) + ".txt"

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
			list_of_feature_sets = []
			graph_labels_tmp = []
			filelist_tmp = None
			graph_labels_list_tmp = None
			if train:
				filelist_tmp = filelist_train
				output_tmp = output_train_tmp
				train = False
				graph_labels_list_tmp = graph_labels_train
			else:
				filelist_tmp = filelist_test
				output_tmp = output_test_tmp
				test = False
				graph_labels_list_tmp = graph_labels_test
			for f in filelist_tmp:
				num = int(f.split("_")[1])
				if num in pos_graphs:
					graph_labels_tmp.append(1)
				else:
					graph_labels_tmp.append(0)
				feature_set = set()
				g = ConjunctiveGraph()
				g.load(open(f, 'rb'))
				for t in g.triples((None, None, None)):
					s,p,o = t
					if t[0].startswith("N"):
						s = "Blank"
					if t[2].startswith("N"):
						o = "Blank"
					if type == "o":
						p = "relation"
					if triple_dict[(s,p,o)] == 0:
						triple_dict[(s,p,o)] = triple_counter
						feature_set.add(triple_counter)
						# write 1 on triple position in tabular
						triple_counter += 1
					else:
						pos = triple_dict[(s,p,o)]
						feature_set.add(pos)
						# write 1 on triple position in tabular
				list_of_feature_sets.append(feature_set)
			graph_labels_list_tmp.append(graph_labels_tmp)
			X = []
			for instance in list_of_feature_sets:
				ins = np.zeros(200)
				for pos in instance:
					ins[pos] = 1
				X.append(ins)
			f = open(output_tmp, 'wt')
			try:
				writer = csv.writer(f)
				for ins in X:
					writer.writerow( list(ins) )
			finally:
				f.close()
		i_fold += 1
	return graph_labels_train, graph_labels_test


def parse_csv(filename):
	X = []
	f = open(filename, 'rt')
	try:
		reader = csv.reader(f)
		for row in reader:
			X.append([float(r) for r in row])
	finally:
		f.close()
	return np.array(X)


def preproscessing(database_train, class_index, labels_mapping, model):
	graph_id_to_list_id = dict()
	list_id_to_graph_id = dict()
	n_graphs = len(database_train)
	n_pos = 0
	pos_index = []
	L_hat = None
	H = None

	for i, graph in enumerate(database_train):
		graph_id_to_list_id[graph.id] = i
		list_id_to_graph_id[i] = graph.id
		if labels_mapping[graph.id][class_index] == 1:
			n_pos += 1
			pos_index.append(1)
		else:
			pos_index.append(0)

	n_neg = n_graphs - n_pos
	neg_index = np.array(1 - np.array(pos_index), dtype=bool)
	pos_index = np.array(pos_index, dtype=bool)

	if model == "gMLC":
		W = np.zeros((n_graphs, len(labels_mapping[1])))
		H = np.zeros((n_graphs, n_graphs))
		for i in xrange(0, n_graphs):
			for j in xrange(0, n_graphs):
				if i == j:
					H[i,j] = 1 - (1.0 / n_graphs)
				else:
					H[i,j] = - (1.0 / n_graphs)
		for i in xrange(0, n_graphs):
			graph_id = list_id_to_graph_id[i]
			labels_tmp = labels_mapping[graph_id]
			for j, label in enumerate(labels_tmp):
				W[i,j] = label
		L = pairwise_kernels(W, metric="linear")
	else:
		W = np.zeros((n_graphs, n_graphs))
		A = 0
		B = 0
		for i in xrange(0, n_graphs):
			graph_id_i = list_id_to_graph_id[i]
			for j in xrange(0, n_graphs):
				graph_id_j = list_id_to_graph_id[j]
				if labels_mapping[graph_id_i][class_index] != labels_mapping[graph_id_j][class_index]:
					A += 1
				if labels_mapping[graph_id_i][class_index] == labels_mapping[graph_id_j][class_index]:
					B += 1

		for i in xrange(0, n_graphs):
			graph_id_i = list_id_to_graph_id[i]
			for j in xrange(0, n_graphs):
				graph_id_j = list_id_to_graph_id[j]
				if labels_mapping[graph_id_i][class_index] == labels_mapping[graph_id_j][class_index]:
					W[i, j] = 1.0 / A
				else:
					W[i, j] = -1.0 / B

		D = np.zeros((n_graphs, n_graphs))
		for i in xrange(0, n_graphs):
			D[i,i] = sum(W[i, ])
		L = D - W

		L_hat = np.copy(L)
		L_hat[L_hat < 0] = 0

	return H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, neg_index, graph_id_to_list_id


def write_labels_mapping(labels_mapping, to_file):
	with open(to_file, "w") as f:
		writer = csv.writer(f)
		for k in labels_mapping:
			writer.writerow(list(labels_mapping[k]))