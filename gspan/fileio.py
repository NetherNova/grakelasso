import graph
import sys
import os
from rdflib import ConjunctiveGraph, URIRef, RDF
import numpy as np
from sklearn import cross_validation

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
				if num in pos_graphs:
					graph_labels_tmp.append(1)
				else:
					graph_labels_tmp.append(0)
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

	triples = graph.triples((subject, None, None))
	for (s,p,o) in triples:
		if p == RDF.type:
			continue
		try:
			s_id = entity_map[s]
		except KeyError:
			entity_map[s] = entity_counter
			s_id = entity_counter
			entity_counter += 1
			id_to_uri[s_id] = s
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
		if (s_local_id, p_id, o_local_id) in edge_set:
			return
		entity_set.add( (s_local_id, s_id) )
		entity_set.add( (o_local_id, o_id) )
		edge_set.append((s_local_id, p_id, o_local_id))
		dfs_triples(entity_set, entity_map, edge_set, relation_map, graph, o)