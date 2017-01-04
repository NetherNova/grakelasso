
import graph
import functools
import collections
import numpy as np
from rdflib import URIRef, ConjunctiveGraph, Literal
import sys
from sklearn.metrics.pairwise import pairwise_kernels
import logging

logging.basicConfig(level=logging.WARNING)

__subgraph_count = 0
"""
structs as named tuples takes care of the __eq__ functions that are needed for fds_code to be used as a ditionary key
"""
pre_dfs = collections.namedtuple('pre_dfs',['id','edge','prev'])
dfs_code = collections.namedtuple('dfs_code',
			['fromn','to','from_label','edge_label','to_label'])

"""
comparison functions for sorting the order the keys are traversed in the dictionary
"""
def dfs_code_compare(a):
	return (a.from_label, a.edge_label, a.to_label)

def dfs_code_backward_compare(a):
	return (a.to, a.edge_label)

def dfs_code_forward_compare(a):
	return (-a.fromn, a.edge_label, a.to_label)

class history():
	"""
 	class to maintain the history information.
 	"""
	def __init__(self):
		self.edges = []
		self.has_edges = set()
		self.has_node = set()

	def build(self, pdfs):
		ps = pdfs
		while ps != None:
			self.edges.append(ps.edge)
			self.has_edges.add(ps.edge.id)
			self.has_node.add(ps.edge.fromn)
			self.has_node.add(ps.edge.to)
			ps = ps.prev
		self.edges = list(reversed(self.edges))


def trim_infrequent_nodes(database, minsup):
	"""
	Calculates the freqent labels in *database* and removes infrequent ones
	:param database:
	:param minsup:
	:return:
	"""
	totrim = []
	frequent = []
	freq_labels = {}

	for g in database:
		nset = set()
		for n in g.nodes:
			nset.add(n.label)
		for l in list(nset):
			if l in freq_labels:
				freq_labels[l] += 1
			else:
				freq_labels[l] = 1
	for label in freq_labels:
		if freq_labels[label] < minsup:
			totrim.append(label)
		else:
			frequent.append(label)
	return database, frequent, totrim, freq_labels


def build_right_most_path(dfs_codes):
	"""
	Build the right most path through the DFS codes
	:param dfs_codes:
	:return:
	"""
	path = []
	prev_id = -1
	for idx,c in reversed(list(enumerate(dfs_codes))):
		if c.fromn < c.to and (len(path) == 0 or prev_id == c.to):
			prev_id = c.fromn
			path.append(idx)
	return path


def genumerate(projection, right_most_path, dfs_codes, min_label, db, mapper):
	"""
	Iterate through the projection to find potential next edges
	:param projection:
	:param right_most_path:
	:param dfs_codes:
	:param min_label:
	:param db:
	:param mapper:
	:return:
	"""
	pm_backward = {}
	pm_forward = {}
	p_graph = projection_to_graph(dfs_codes, mapper)

	for p in projection:	# holds number of graphs of this pattern f_gk = [0 0 1] |gk|
		h = history()
		h.build(p)
		pm_backward = get_backward(p, right_most_path, h, pm_backward, 
						dfs_codes, db)
		pm_forward = get_first_forward(p, right_most_path, h, pm_forward,
						dfs_codes, db, min_label)
		pm_forward = get_other_forward(p, right_most_path, h, pm_forward,
						dfs_codes, db, min_label)
	return pm_backward, pm_forward


def get_forward_init(node, graph):
	"""
	Get initial edges from the graph to grow.
	:param node:
	:param graph:
	:return:
	"""
	edges = []
	for e in node.edges:
		if node.label <= graph.nodes[e.to].label:
			edges.append(e)
	return edges


def get_backward(prev_dfs, right_most_path, hist, pm_backward, dfs_codes, db):
	"""
	Search to backward edges as potential next edges
	:param prev_dfs:
	:param right_most_path:
	:param hist:
	:param pm_backward:
	:param dfs_codes:
	:param db:
	:return:
	"""
	last_edge = hist.edges[right_most_path[0]]
	g = db[prev_dfs.id]
	last_node = g.nodes[last_edge.to]

	for idx,rmp in reversed(list(enumerate(right_most_path[1:]))):
		edge = hist.edges[rmp]
		for e in last_node.edges:
			if e.id in hist.has_edges:
				continue
			if e.to not in hist.has_node:
				continue
			from_node = g.nodes[edge.fromn]
			to_node = g.nodes[edge.to]
			if e.to == edge.fromn and (e.label > edge.label or (e.label == edge.label and last_node.label >= to_node.label)):
				from_id = dfs_codes[right_most_path[0]].to
				to_id = dfs_codes[rmp].fromn
				dfsc = dfs_code(from_id, to_id, last_node.label, e.label, from_node.label)
				pdfs = pre_dfs(g.id, e, prev_dfs)
				if dfsc in pm_backward:
					pm_backward[dfsc].append(pdfs)
				else:
					pm_backward[dfsc] = [pdfs,]
	
	return pm_backward


def get_first_forward(prev_dfs, right_most_path, hist, pm_forward, dfs_codes, db, min_label):
	"""
	Find the first forward edge as a next edge
	:param prev_dfs:
	:param right_most_path:
	:param hist:
	:param pm_forward:
	:param dfs_codes:
	:param db:
	:param min_label:
	:return:
	"""
	last_edge = hist.edges[right_most_path[0]]
	g = db[prev_dfs.id]
	last_node = g.nodes[last_edge.to]

	for e in last_node.edges:
		to_node = g.nodes[e.to]
		# if this node has already been explored or is further up in the min_label search
		# (reversed order as initial starting) --> continue
		if e.to in hist.has_node or to_node.label < min_label:
			continue
		to_id = dfs_codes[right_most_path[0]].to
		dfsc = dfs_code(to_id, to_id+1, last_node.label, e.label, to_node.label)
		pdfs = pre_dfs(g.id,e,prev_dfs)
		if dfsc in pm_forward:
			pm_forward[dfsc].append(pdfs)
		else:
			pm_forward[dfsc] = [pdfs,]

	return pm_forward


def get_other_forward(prev_dfs, right_most_path, hist, pm_forward, dfs_codes, db, min_label):
	"""
	Append any other forward edges as potential next edges
	:param prev_dfs:
	:param right_most_path:
	:param hist:
	:param pm_forward:
	:param dfs_codes:
	:param db:
	:param min_label:
	:return:
	"""
	g = db[prev_dfs.id]

	for rmp in right_most_path:
		cur_edge = hist.edges[rmp]
		cur_node = g.nodes[cur_edge.fromn]
		cur_to = g.nodes[cur_edge.to]

		for e in cur_node.edges:
			to_node = g.nodes[e.to]
			if to_node.id == cur_to.id or to_node.id in hist.has_node or to_node.label < min_label:
				continue
			if cur_edge.label < e.label or (cur_edge.label == e.label and cur_to.label <= to_node.label):
				from_id = dfs_codes[rmp].fromn
				to_id = dfs_codes[right_most_path[0]].to

				dfsc = dfs_code(from_id, to_id+1, cur_node.label, e.label, to_node.label)
				pdfs = pre_dfs(g.id,e,prev_dfs)

				if dfsc in pm_forward:
					pm_forward[dfsc].append(pdfs)
				else:
					pm_forward[dfsc] = [pdfs,]

	return pm_forward


def count_support(projection):
	"""
	Count how many graphs this projection shows up
	:param projection:
	:return:
	"""
	prev_id = -1
	size = 0
	for p in projection:
		if prev_id != p.id:
			prev_id = p.id
			size += 1
	return size


def build_graph(dfs_codes):
	"""
	Build a graph for a given set of dfs codes
	:param dfs_codes:
	:return:
	"""
	g = graph.Graph()
	numnodes = max([x[0] for x in dfs_codes] + [x[1] for x in dfs_codes])+1
	for i in range(numnodes):
		n = graph.Node()
		g.nodes.append(n)

	for idx,c in enumerate(dfs_codes):
		g.nodes[c.fromn].id = c.fromn
		g.nodes[c.fromn].label = c.from_label
		g.nodes[c.to].id = c.to
		g.nodes[c.to].label = c.to_label

		e = graph.Edge()
		e.id = g.nedges
		e.fromn = c.fromn
		e.to = c.to
		e.label = c.edge_label
		g.nodes[c.fromn].edges.append(e)
		"""
		double edges ?

		e2 = graph.Edge()
		e2.id = e.id
		e2.label = e.label
		e2.fromn = c.to
		e2.to = c.fromn
		g.nodes[c.to].edges.append(e2)
		"""
		g.nedges += 1

	return g


def is_min(dfs_codes):
	"""
	Check if a given DFS code is a minimum DFS code. Recursive.
	:param dfs_codes:
	:return:
	"""
	if len(dfs_codes) == 1:
		return True

	min_dfs_codes = []
	mingraph = build_graph(dfs_codes)
	
	projection_map = {}
	for n in mingraph.nodes:
		edges = []
		edges += get_forward_init(n, mingraph)
		if len(edges) > 0:
			 for e in edges:
				nf = mingraph.nodes[e.fromn]
				nt = mingraph.nodes[e.to]
				dfsc = dfs_code(0,1,nf.label,e.label,nt.label)

				pdfs = pre_dfs(0,e,None)

				if dfsc in projection_map:
					projection_map[dfsc].append(pdfs)
				else:
					projection_map[dfsc] = [pdfs,]

	pm = sorted(projection_map, key=dfs_code_compare)[0]
	min_dfs_codes.append(dfs_code(0,1,pm[2],pm[3],pm[4]))
	if dfs_codes[len(min_dfs_codes)-1] != min_dfs_codes[-1]:
		return False

	return projection_min(projection_map[pm], dfs_codes, min_dfs_codes, mingraph)


def judge_backwards(right_most_path, projection, min_dfs_codes, min_label, mingraph):
	"""
	Check for any backwards edges
	:param right_most_path:
	:param projection:
	:param min_dfs_codes:
	:param min_label:
	:param mingraph:
	:return:
	"""
	pm_backwards = {}

	for idx, c in reversed(list(enumerate(right_most_path[1:]))):
		for j in projection:
			h = history()
			h.build(j)
			
			last_edge = h.edges[right_most_path[0]]
			last_node = mingraph.nodes[last_edge.to]

			edge = h.edges[right_most_path[idx]]
			to_node = mingraph.nodes[edge.to]
			from_node = mingraph.nodes[edge.fromn]

			for e in last_node.edges:
				if e.id in h.has_edges:
					continue
				if e.to not in h.has_node:
					continue
				if e.to == edge.fromn and (e.label > edge.label or (e.label == edge.label and last_node.label > to_node.label)):
					from_id = min_dfs_codes[right_most_path[0]].to
					to_id = min_dfs_codes[right_most_path[idx]].fromn

					dfsc = dfs_code(from_id, to_id, last_node.label, e.label, from_node.label)
					pdfs = pre_dfs(0,e,j)

					if dfsc in pm_backwards:
						pm_backwards[dfsc].append(pdfs)
					else:
						pm_backwards[dfsc] = [pdfs,]

		if len(pm_backwards.keys()) != 0: 
			return True, pm_backwards

	return False, pm_backwards


def judge_forwards(right_most_path, projection, min_dfs_codes, min_label, mingraph):
	"""
	check for any valid forward edges
	:param right_most_path:
	:param projection:
	:param min_dfs_codes:
	:param min_label:
	:param mingraph:
	:return:
	"""
	pm_forward = {}

	for idx,p in enumerate(projection):
		h = history()
		h.build(p)

		last_edge = h.edges[right_most_path[0]]
		last_node = mingraph.nodes[last_edge.to]

		for e in last_node.edges:
			to_node = mingraph.nodes[e.to]

			if e.to in h.has_node or to_node.label < min_label:
				continue

			to_id = min_dfs_codes[right_most_path[0]].to
			dfsc = dfs_code(to_id, to_id+1, last_node.label, e.label, to_node.label)
			pdfs = pre_dfs(0,e,p)

			if dfsc in pm_forward:
				pm_forward[dfsc].append(pdfs)
			else:
				pm_forward[dfsc] = [pdfs,]
	
	if len(pm_forward.keys()) == 0:
		for rmp in right_most_path:
			for p in projection:
				h = history()
				h.build(p)

				cur_edge = h.edges[rmp]
				cur_node = mingraph.nodes[cur_edge.fromn]
				cur_to = mingraph.nodes[cur_edge.to]
				
				for e in cur_node.edges:
					to_node = mingraph.nodes[e.to]
					
					if cur_edge.to == to_node.id or to_node.id in h.has_node or to_node.label < min_label:
						continue

					if cur_edge.label < e.label or (cur_edge.label == e.label and cur_to.label <= to_node.label):
						from_id = min_dfs_codes[rmp].fromn
						to_id = min_dfs_codes[right_most_path[0]].to
						dfsc = dfs_code(from_id, to_id+1, cur_node.label, e.label, to_node.label)

						pdfs = pre_dfs(0,e,p)
						
						if dfsc in pm_forward:
							pm_forward[dfsc].append(pdfs)
						else:
							pm_forward[dfsc] = [pdfs,]
			
			if len(pm_forward.keys()) != 0:
				break

	if len(pm_forward.keys()) != 0:
		return True, pm_forward
	else:
		return False, pm_forward


def projection_min(projection, dfs_codes, min_dfs_codes, mingraph):
	"""
	Build a minimum projection
	:param projection:
	:param dfs_codes:
	:param min_dfs_codes:
	:param mingraph:
	:return:
	"""
	right_most_path = build_right_most_path(min_dfs_codes)
	min_label = min_dfs_codes[0].from_label

	ret, pm_backward = judge_backwards(right_most_path, projection, min_dfs_codes, min_label, mingraph)
	if ret:
		for pm in sorted(pm_backward, key=dfs_code_backward_compare):
			min_dfs_codes.append(pm)
			if dfs_codes[len(min_dfs_codes)-1] != min_dfs_codes[-1]:
				return False

			return projection_min(pm_backward[pm], dfs_codes, min_dfs_codes, mingraph)
	
	ret, pm_forward = judge_forwards(right_most_path, projection, min_dfs_codes, min_label, mingraph)
	if ret:
		for pm in sorted(pm_forward, key=dfs_code_forward_compare):
			min_dfs_codes.append(pm)
			if dfs_codes[len(min_dfs_codes)-1] != min_dfs_codes[-1]:
				return False

			return projection_min(pm_forward[pm], dfs_codes, min_dfs_codes,mingraph)
	return True


def show_subgraph(dfs_codes, nsupport, mapper):
	"""
	Draw a frequent subgraph with its support.
	:param dfs_codes:
	:param nsupport:
	:param mapper:
	:return:
	"""
	global __subgraph_count

	g = build_graph(dfs_codes)
	g.id = __subgraph_count
	__subgraph_count += 1
	g.gprint(nsupport, mapper)


def project(database, frequent_nodes, minsup, freq_labels, length, H, L, L_hat, n_graphs, n_pos, n_neg, pos_index, class_index, neg_index, graph_id_to_list_id, mapper, labels, model, constraints):
	"""
	Generate initial edges and start the mining process
	:param database:
	:param frequent_nodes:
	:param minsup:
	:param freq_labels:
	:param length:
	:param H:
	:param L:
	:param L_hat:
	:param n_graphs:
	:param n_pos:
	:param n_neg:
	:param pos_index:
	:param class_index:
	:param neg_index:
	:param graph_id_to_list_id:
	:param mapper:
	:param labels:
	:param model:
	:param constraints:
	:return:
	"""
	global __subgraph_count
	global __positive_index
	global __n_pos
	global __n_graphs
	global __H
	global __L
	global __L_hat
	global __dataset
	global __pattern_set
	global __cl_constraints
	global __ml_constraints
	global __negative_index
	global __graph_id_to_list_id
	global __neighbor_set

	__graph_id_to_list_id = graph_id_to_list_id

	__ml_constraints = [c for c in constraints[0] if c[0] < n_graphs and c[1] < n_graphs]
	__cl_constraints = [c for c in constraints[1] if c[0] < n_graphs and c[1] < n_graphs]

	# TODO: evaluate
	"""
	Only constraints for current binary split
	for con in __ml_constraints:
		if not labels[con[0]][class_index] == 1 and not labels[con[1]][class_index] == 1:
			__ml_constraints.remove((con[0], con[1]))

	for con in __cl_constraints:
		if not labels[con[0]][class_index] == 1 and not labels[con[1]][class_index] == 1:
			__cl_constraints.remove((con[0], con[1]))
	"""
	for i, con in enumerate(__ml_constraints):
		if con[0] >= n_graphs or con[1] >= n_graphs:
			__ml_constraints.remove(con)
			continue
		try:
			list_id1 = __graph_id_to_list_id[con[0]]
			list_id2 = __graph_id_to_list_id[con[1]]
			__ml_constraints[i] = (list_id1, list_id2)
		except KeyError:
			__ml_constraints.remove(con)

	for i, con in enumerate(__cl_constraints):
		if con[0] >= n_graphs or con[1] >= n_graphs:
			__cl_constraints.remove(con)
			continue
		try:
			list_id1 = __graph_id_to_list_id[con[0]]
			list_id2 = __graph_id_to_list_id[con[1]]
			__cl_constraints[i] = (list_id1, list_id2)
		except KeyError:
			__cl_constraints.remove(con)

	__positive_index = pos_index
	__negative_index = neg_index
	__n_pos = n_pos
	__n_graphs = n_graphs
	__H = H
	__L = L
	__L_hat = L_hat
	__dataset = []
	__pattern_set = []
	__subgraph_count = 0
	dfs_codes = []
	projection_map = {}

	for l in frequent_nodes:
		__subgraph_count += 1		

	for g in database:
		for n in g.nodes:
			#edges = []
			edges = get_forward_init(n, g)
			if len(edges) > 0:
				 for e in edges:
					nf = g.nodes[e.fromn]
					nt = g.nodes[e.to]
					dfsc = dfs_code(0,1,nf.label,e.label,nt.label)

					pdfs = pre_dfs(g.id,e,None)
					# because this is a root --> append the predecesspr dfs code (graph id, edge, None)
					if dfsc in projection_map:
						projection_map[dfsc].append(pdfs)
					else:
						projection_map[dfsc] = [pdfs,]

	# Start Subgraph Mining
	for pm in reversed(sorted(projection_map, key=dfs_code_compare)):	# sorted by highest fromnode label (order is important)
		# print pm
		# Partial pruning like apriori
		if len(projection_map[pm]) < minsup: # number of graphs, this initial pattern occurs
			continue
		
		dfs_codes.append(dfs_code(0,1,pm[2],pm[3],pm[4]))	# initial pattern for this projection is always local 0, 1)

		dfs_codes = mine_subgraph(database, projection_map[pm], 
							dfs_codes, minsup, length, 0, mapper, model)

		dfs_codes.pop()	# dfs_codes is a list of all projections for this initial pattern

	return __dataset, __pattern_set


def mine_subgraph(database, projection, dfs_codes, minsup, length, threshold, mapper, model):
	"""
	recursive subgraph mining routine
	:param database:
	:param projection:
	:param dfs_codes:
	:param minsup:
	:param length:
	:param threshold:
	:param mapper:
	:param model:
	:return:
	"""
	nsupport = count_support(projection)
	if nsupport < minsup:
		return dfs_codes
	if not is_min(dfs_codes):
		return dfs_codes
	stopping, threshold = evaluate_and_prune(dfs_codes, mapper, projection, threshold, length, model)
	if stopping:
		return dfs_codes

	# show_subgraph(dfs_codes, nsupport, mapper)
	right_most_path = build_right_most_path(dfs_codes)
	min_label = dfs_codes[0].from_label	# dfs_codes[0] is the starting pattern of this search (root),
	# it has the minimum node label (because reversed sorted before starting search)
	
	pm_backward, pm_forward = genumerate(projection, right_most_path, dfs_codes, min_label, database, mapper)

	for pm in sorted(pm_backward, key=dfs_code_backward_compare):
		dfs_codes.append(pm)
		dfs_codes = mine_subgraph(database, pm_backward[pm], dfs_codes, minsup, length, threshold, mapper, model)
		dfs_codes.pop()

	for pm in reversed(sorted(pm_forward, key=dfs_code_forward_compare)):
		dfs_codes.append(pm)
		dfs_codes = mine_subgraph(database, pm_forward[pm], dfs_codes, minsup, length, threshold, mapper, model)
		dfs_codes.pop()

	return dfs_codes

def q(vector, hat=False):
	"""
	Scoring function of GMGFL
	:param projection:
	:return: quality function
	"""
	global __positive_index
	global __n_pos
	global __n_graphs
	global __L
	global __L_hat
	global __dataset

	if hat:
		ret = vector.dot(__L_hat).dot(vector)
	else:
		ret = vector.dot(__L).dot(vector)
	return ret, vector

def projection_to_vector(projection):
	"""
	Numpy array from graph occurences
	:param projection:
	:return:
	"""
	global __n_graphs
	global __graph_id_to_list_id

	vector = np.zeros(__n_graphs)
	for p in projection:
		list_id = __graph_id_to_list_id[p.id]
		vector[list_id] = 1
	return vector


def get_min(fun):
	"""
	Return minimum index and value specified by function pointer *fun* over dataset
	:param fun:
	:return:
	"""
	global __dataset

	min_val = sys.maxint
	min_index = 0
	for i, vec in enumerate(__dataset):
		ret = fun(vec)
		if ret < min_val:
			min_val = ret
			min_index = i
	return min_index, min_val


def greedy_value(vector):
	"""
	Scoring function for greedy search
	:param vector:
	:return:
	"""
	global __positive_index
	global __n_graphs
	global __n_pos
	global __negative_index

	hits_pos = sum(vector[__positive_index])
	mis_pos = (__n_pos - hits_pos)
	hits_neg = sum(vector[__negative_index])
	mis_neg = (__n_graphs - __n_pos) - hits_neg
	ret = -(mis_pos * mis_neg + hits_pos * hits_neg)
	return ret


def gmlc(vector, hat=False):
	"""
	Scoring function for GMLC
	:param vector:
	:param hat:
	:return:
	"""
	global __dataset
	global __H
	global __L

	M = __H.dot(__L).dot(__H)
	M_hat = np.copy(M)
	M_hat[M_hat < 0] = 0
	if hat:
		ret = vector.dot(M_hat).dot(vector)
	else:
		ret = vector.dot(M).dot(vector)
	return ret


def check_ml_constraints(vector):
	"""
	True - if one of the *__ml_constraints* is violated
	False - else
	:param vector:
	:return:
	"""
	global __ml_constraints

	for con in __ml_constraints:
		# a vector must be either contained or missing for both instances *con* tuple
		if not (vector[con[0]] == vector[con[1]]):
			return False
	return True

def check_cl_constraints(vector):
	"""
	True - if one of the *__cl_constraints* is violated
	False - else
	:param vector:
	:return:
	"""
	global __cl_constraints

	for con in __cl_constraints:
		# a vector is not allowed to hold for both
		if vector[con[0]] == 1 and vector[con[1]] == 1:
			return False
	return True

def evaluate_and_prune(dfs_codes, mapper, projection, threshold, max_length, model):
	"""
	Apply scoring function of *model* to current pattern projection
	:param dfs_codes:
	:param mapper:
	:param projection:
	:param threshold:
	:param max_length:
	:param model:
	:return:
	"""
	global __pattern_set
	global __dataset
	# subgraph pattern and vector representation
	g = projection_to_graph(dfs_codes, mapper)
	vector = projection_to_vector(projection)
	# constraints checking!
	ml = check_ml_constraints(vector)
	cl = check_cl_constraints(vector)
	if not ml:
		return True, threshold
	if not cl:
		return False, threshold

	dataset_length = len(__dataset)
	add_eval = 0
	prune_eval = 0
	min_index = 0
	threshold = 0
	if model == "gMGFL":
		add_eval, _ = q(vector)
		prune_eval = q(vector, hat=True)
		min_index, threshold  = get_min(q)

	elif model == "top-k":
		add_eval = prune_eval = count_support(projection)
		min_index, threshold  = get_min(sum)

	elif model == "greedy":
		add_eval = prune_eval = greedy_value(vector)
		min_index, threshold  = get_min(greedy_value)

	elif model == "gMLC":
		add_eval = gmlc(vector)
		prune_eval = gmlc(vector, hat=True)
		min_index, threshold  = get_min(gmlc)
	else:
		logging.log(logging.ERROR, "Model %s not recognized" %(model))
		exit(0)
	# evaluate current pattern set
	if dataset_length < max_length or add_eval > threshold:
		__dataset.append(vector)
		__pattern_set.append(g)
	if dataset_length > max_length:
		__dataset.pop(min_index)
		__pattern_set.pop(min_index)
	if prune_eval <= threshold:
		return True, threshold
	return False, threshold


def projection_to_graph(dfs_codes, mapper):
	"""
	restore graph entities and relations
	:param dfs_codes:
	:param mapper:
	:return:
	"""
	g = build_graph(dfs_codes)
	local_dict = dict()
	edge_list = []
	for n in g.nodes:
		local_dict[n.id] = mapper[n.label]

	for n in g.nodes:
		for e in n.edges:
			# print(local_dict[e.fromn], local_dict[e.to], mapper[e.label])
			edge_list.append((local_dict[e.fromn], local_dict[e.to], mapper[e.label]))
	return edge_list


def database_to_matrix(database, pattern_set, mapper):
	"""
	Convert graph database to feature matrix given *pattern_set*
	:param database:
	:param pattern_set:
	:param mapper:
	:return:
	"""
	ret = np.zeros((len(database), len(pattern_set)))
	for i, g in enumerate(database):
		edge_list = []
		local_dict = dict()
		for n in g.nodes:
			local_dict[n.id] = mapper[n.label]

		for n in g.nodes:
			for e in n.edges:
				edge_list.append((local_dict[e.fromn], local_dict[e.to], mapper[e.label]))
		for j, p in enumerate(pattern_set):
			contained = True
			for triple in p:
				if not triple in edge_list:
					continue
				ret[i, j] = 1
	return ret

def database_statistics(database):
	"""
	Calculate average edge and nodes graph statistics
	:param database:
	:return:
	"""
	global_node_counter = 0
	global_edge_counter = 0
	n_graphs = 0
	for i, g in enumerate(database):
		for n in g.nodes:
			global_node_counter += 1
			for e in n.edges:
				global_edge_counter += 1
		n_graphs += 1
	avg_node = float(global_node_counter) / n_graphs
	avg_edge = float(global_edge_counter) / n_graphs

	return avg_node, avg_edge

