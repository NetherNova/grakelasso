
import graph
import functools
import collections
import numpy as np
from rdflib import URIRef
import sys

__subgraph_count = 0

# 
# I'm using a couple of structs as named tuples here. This takes care of the
# __eq__ functions that are needed for fds_code to be used as a ditionary key
#
pre_dfs = collections.namedtuple('pre_dfs',['id','edge','prev'])
dfs_code = collections.namedtuple('dfs_code',
			['fromn','to','from_label','edge_label','to_label'])

# 
# These are the comparison functions for sorting the order the keys are
# traversed in the dictionary
#
def dfs_code_compare(a):
	return (a.from_label, a.edge_label, a.to_label)

def dfs_code_backward_compare(a):
	return (a.to, a.edge_label)

def dfs_code_forward_compare(a):
	return (-a.fromn, a.edge_label, a.to_label)

# One class here to maintain the history information. 
class history():
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

# 
# Calculates the freqent labels
#
def trim_infrequent_nodes(database, minsup):
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
	print frequent
	print totrim
	return database, frequent, totrim, freq_labels

#
# Build the right most path through the DFS codes 
# 
def build_right_most_path(dfs_codes):
	path = []
	prev_id = -1
	#print list(reversed(list(enumerate(dfs_codes))))
	for idx,c in reversed(list(enumerate(dfs_codes))):
		if c.fromn < c.to and (len(path) == 0 or prev_id == c.to):
			prev_id = c.fromn
			path.append(idx)
	#print path
	return path

# 
# Iterate through the projection to find potential next edges (?)
#
def genumerate(projection, right_most_path, dfs_codes, min_label, db, mapper):
	global __cl_constraints
	global __ml_constraints
	#print min_label, len(projection)
	pm_backward = {}
	pm_forward = {}
	p_graph = projection_to_graph(dfs_codes, mapper)

	for p in projection:	# holds number of graphs of this pattern f_gk = [0 0 1} |gk|
		h = history()
		h.build(p)

		#print p.id, p.edge.fromn, p.edge.to, p.prev
		pm_backward = get_backward(p, right_most_path, h, pm_backward, 
						dfs_codes, db)
		pm_forward = get_first_forward(p, right_most_path, h, pm_forward,
						dfs_codes, db, min_label)
		pm_forward = get_other_forward(p, right_most_path, h, pm_forward,
						dfs_codes, db, min_label)
	if (len(pm_backward) > 0 or len(pm_forward) > 0):
		if len(__cl_constraints) > 0:
			pm_backward, pm_forward = check_cl_constraints(p_graph, pm_backward, pm_forward, mapper)
		if len(__ml_constraints) > 0:
			pm_backward, pm_forward = check_ml_constraints(p_graph, pm_backward, pm_forward, mapper)
	return pm_backward, pm_forward


def check_ml_constraints(p_graph, pm_backward, pm_forward, mapper):
	global __ml_constraints

	retain_list = []
	remove_list = []
	for c in __ml_constraints:
		if c[0] == "*":
			for k in pm_backward:
				tuple = ("*", mapper[k.to_label], "*")
				if tuple == c:
					retain_list.append(k)
			for k in pm_forward:
				tuple = ("*", mapper[k.to_label], "*")
				if tuple == c:
					retain_list.append(k)
		else:
			for k in pm_backward:
				tuple = (mapper[k.from_label], mapper[k.to_label], mapper[k.edge_label])
				if tuple == c:
					retain_list.append(k)
			for k in pm_forward:
				tuple = (mapper[k.from_label], mapper[k.to_label], mapper[k.edge_label])
				if tuple == c:
					retain_list.append(k)
	if len(retain_list) > 0:
		for k in pm_backward:
			if not k in retain_list:
				remove_list.append(k)
		for k in pm_forward:
			if not k in retain_list:
				remove_list.append(k)
		for r in remove_list:
			try:
				pm_forward.pop(r)
			except KeyError:
				pass
			try:
				pm_backward.pop(r)
			except KeyError:
				pass
	return pm_backward, pm_forward

def check_cl_constraints(p_graph, pm_backward, pm_forward, mapper):
	global __cl_constraints

	remove_list = []
	for c in __cl_constraints:
		if c[0] == "*":
			for k in pm_backward:
				tuple = ("*", mapper[k.to_label], "*")
				if tuple == c:
					remove_list.append(k)
			for k in pm_forward:
				tuple = ("*", mapper[k.to_label], "*")
				if tuple == c:
					remove_list.append(k)
		else:
			for k in pm_backward:
				tuple = (mapper[k.from_label], mapper[k.to_label], mapper[k.edge_label])
				if tuple == c:
					remove_list.append(k)
			for k in pm_forward:
				tuple = (mapper[k.from_label], mapper[k.to_label], mapper[k.edge_label])
				if tuple == c:
					remove_list.append(k)
	for r in remove_list:
		try:
			pm_forward.pop(r)
		except KeyError:
			pass
		try:
			pm_backward.pop(r)
		except KeyError:
			pass
	return pm_backward, pm_forward

#
# Get initial edges from the graph to grow.
#
def get_forward_init(node, graph):
	edges = []
	
	for e in node.edges:
		if node.label <= graph.nodes[e.to].label:
			edges.append(e)
	return edges

#
# Search to backward edges as potential next edges
#
def get_backward(prev_dfs, right_most_path, hist, pm_backward, dfs_codes, db):
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
			#print 'here3'
			from_node = g.nodes[edge.fromn]
			to_node = g.nodes[edge.to]
			#print 'here3',g.id,last_edge.fromn,last_edge.to,last_node.id, edge.fromn, edge.to, edge.label, idx, from_node.id, to_node.id
			if e.to == edge.fromn and (e.label > edge.label or (e.label == edge.label and last_node.label >= to_node.label)):
				#print 'here4'
				from_id = dfs_codes[right_most_path[0]].to
				to_id = dfs_codes[rmp].fromn
				
				dfsc = dfs_code(from_id, to_id, last_node.label, e.label, from_node.label)
				pdfs = pre_dfs(g.id, e, prev_dfs)
				
				if dfsc in pm_backward:
					pm_backward[dfsc].append(pdfs)
				else:
					pm_backward[dfsc] = [pdfs,]
	
	return pm_backward

#
# Find the first forward edge as a next edge
# 
def get_first_forward(prev_dfs, right_most_path, hist, pm_forward, dfs_codes, db, min_label):
	last_edge = hist.edges[right_most_path[0]]
	g = db[prev_dfs.id]
	last_node = g.nodes[last_edge.to]

	for e in last_node.edges:
		to_node = g.nodes[e.to]
		
		if e.to in hist.has_node or to_node.label < min_label:	# if this node has already been explored or is further up in the min_label search (reversed order as initial starting) --> continue
			continue

		to_id = dfs_codes[right_most_path[0]].to
		dfsc = dfs_code(to_id, to_id+1, last_node.label, e.label, to_node.label)

		pdfs = pre_dfs(g.id,e,prev_dfs)

		if dfsc in pm_forward:
			pm_forward[dfsc].append(pdfs)
		else:
			pm_forward[dfsc] = [pdfs,]

	return pm_forward
#
# Append any other forward edges as potential next edges
# 
def get_other_forward(prev_dfs, right_most_path, hist, pm_forward, dfs_codes, db, min_label):
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

#
# Count how many graphs this projection shows up (?)
#
def count_support(projection):
	prev_id = -1
	size = 0

	for p in projection:
		if prev_id != p.id:
			prev_id = p.id
			size += 1
	return size

#
# Build a graph for a given set of dfs codes. 
#
def build_graph(dfs_codes):
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
		e2 = graph.Edge()
		e2.id = e.id
		e2.label = e.label
		e2.fromn = c.to
		e2.to = c.fromn
		g.nodes[c.to].edges.append(e2)
		"""
		g.nedges += 1

	return g

#
# Check if a given DFS code is a minimum DFS code. Recursive.
#
def is_min(dfs_codes):

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

#
# Check for any backwards edges (?)
#
def judge_backwards(right_most_path, projection, min_dfs_codes, min_label, mingraph):
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

#
# check for any forward edges (?)
#
def judge_forwards(right_most_path, projection, min_dfs_codes, min_label, mingraph):
	
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

#
# Build a minimum projection (??) 
#
def projection_min(projection, dfs_codes, min_dfs_codes, mingraph):
	right_most_path = build_right_most_path(min_dfs_codes)
	min_label = min_dfs_codes[0].from_label

	ret, pm_backward = judge_backwards(right_most_path, projection, min_dfs_codes, min_label, mingraph)
	#print ret,pm_backward.keys()
	if ret:
		for pm in sorted(pm_backward, key=dfs_code_backward_compare):
			#print '--- ',pm
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

#
# Draw a frequent subgraph with its support.
# 
def show_subgraph(dfs_codes, nsupport, mapper):
	global __subgraph_count

	g = build_graph(dfs_codes)
	g.id = __subgraph_count
	__subgraph_count += 1
	g.gprint(nsupport, mapper)


#
# e: uri of entity
# pattern: graph representation of pattern
#
def check_entity_in_pattern(e, pattern):
	global __entity_dict
	try:
		val = __entity_dict[e]
	except KeyError:
		return False

	for s,o,p in pattern:
		pass
	return True




# 
# Generate initial edges and start the mining process
#
def project(database, frequent_nodes, minsup, freq_labels, length, mapper, labels, model, constraints, dependency_matrix, entity_dict, product_map):
	global __subgraph_count
	global __positive_index
	global __n_pos
	global __n_graphs
	global __L
	global __L_hat
	global __dataset
	global __pattern_set
	global __cl_constraints
	global __ml_constraints
	global __negative_index
	global __dependency_matrix
	global __entity_dict




	__entity_dict = entity_dict
	__dependency_matrix = dependency_matrix	# |entity| x |entity| 0-1 matrix

	__cl_constraints = constraints[0]
	__ml_constraints = constraints[1]

	__positive_index = labels
	__negative_index = np.array(1 - np.array(__positive_index), dtype=bool)
	__n_pos = sum(__positive_index)
	__n_graphs = len(__positive_index)

	__dataset = []
	__pattern_set = []
	__subgraph_count = 0

	W = np.zeros((__n_graphs, __n_graphs))
	A = 0
	B = 0
	for i in xrange(0, __n_graphs):
		for j in xrange(0, __n_graphs):
			if __positive_index[i] != __positive_index[j]:
				A += 1
			if __positive_index[i] == __positive_index[j]:
				B += 1

	for i in xrange(0, __n_graphs):
		for j in xrange(0, __n_graphs):
			if __positive_index[i] != __positive_index[j]:
				W[i, j] = 1.0 / A
			else:
				W[i, j] = -1.0 / B


	D = np.zeros((__n_graphs, __n_graphs))
	for i in xrange(0, __n_graphs):
		D[i,i] = sum(W[i, ])
	__L = D - W

	__L_hat = np.copy(__L)
	__L_hat[__L_hat < 0] = 0

	dfs_codes = []

	projection_map = {}
        
        # Print out all single-node graphs up front.	
	for l in frequent_nodes:
		#print 't # %d * %d' % (__subgraph_count, freq_labels[l])
		#print 'v 0 %d\n' % (l,)
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

	#for pm in sorted(projection_map, key=dfs_code_compare):
	#	print pm
	#print '----'
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

	if product_map is not None:
		num_features = len(__dataset[0])
		p1 = np.zeros(num_features)
		p2 = np.zeros(num_features)
		p3 = np.zeros(num_features)
		for i in xrange(num_features):
			if product_map[i] == 1:
				p1[i] = 1
			if product_map[i] == 2:
				p2[i] = 1
			if product_map[i] == 3:
				p3[i] = 1
		__pattern_set.append(p1)
		__pattern_set.append(p2)
		__pattern_set.append(p3)
	return __dataset, __pattern_set

# 
# recursive subgraph mining routine
# 
def mine_subgraph(database, projection, dfs_codes, minsup, length, threshold, mapper, model):
	nsupport = count_support(projection)
	if nsupport < minsup:
		return dfs_codes

	if not is_min(dfs_codes):
		return dfs_codes

	# print(nsupport)
	stopping, threshold = evaluate_and_prune(dfs_codes, mapper, projection, threshold, length, model)
	# projection_to_graph(dfs_codes, mapper)
	# print(stopping)
	if stopping:
		return dfs_codes

	# show_subgraph(dfs_codes, nsupport, mapper)

	right_most_path = build_right_most_path(dfs_codes)
	min_label = dfs_codes[0].from_label	# dfs_codes[0] is the starting pattern of this search (root), it has the minimum node label (because reversed sorted before starting search)
	
	pm_backward, pm_forward = genumerate(projection, right_most_path, dfs_codes, min_label, database, mapper)
	#print pm_backward.keys()
	
	#print '-----'
	#for pm in sorted(pm_backward, key=dfs_code_backward_compare):
	#	print pm
	#print '-'
	#for pm in reversed(sorted(pm_forward, key=dfs_code_forward_compare)):
	#	print pm
	#print '------'

	for pm in sorted(pm_backward, key=dfs_code_backward_compare):
		dfs_codes.append(pm)
		dfs_codes = mine_subgraph(database, pm_backward[pm], dfs_codes, minsup, length, threshold, mapper, model)
		dfs_codes.pop()

	for pm in reversed(sorted(pm_forward, key=dfs_code_forward_compare)):
		dfs_codes.append(pm)
		dfs_codes = mine_subgraph(database, pm_forward[pm], dfs_codes, minsup, length, threshold, mapper, model)
		dfs_codes.pop()


	return dfs_codes

def q(projection, vector=[], hat=False):
	"""
	for every graph in the projection assess if its class is positive or negative
	:param projection:
	:return: quality function
	"""
	global __positive_index
	global __n_pos
	global __n_graphs
	global __L
	global __L_hat
	global __dataset

	if len(vector) == 0:
		vector = projection_to_vector(projection)
	#print(vector)
	# structure weight according to BoM / Operation (Causing)

	if hat:
		ret = vector.dot(__L).dot(vector)
	else:
		ret = vector.dot(__L_hat).dot(vector)

	#print(ret)
	return ret, vector

def projection_to_vector(projection):
	global __n_graphs
	vector = np.zeros(__n_graphs)
	for p in projection:
		vector[p.id] = 1
	return vector

def get_min_q():
	global __dataset
	global __L
	min_q = sys.maxint
	min_index = 0
	for i, vec in enumerate(__dataset):
		ret = vec.dot(__L).dot(vec)
		if ret < min_q:
			min_q = ret
			min_index = i
	#if remove:
	#	__dataset.pop(min_index)
	#	__pattern_set.pop(min_index)
	return min_index, min_q

def get_min_freq():
	global __dataset
	min_freq = sys.maxint
	min_index = 0
	for i, vec in enumerate(__dataset):
		ret = sum(vec)
		if ret < min_freq:
			min_freq = ret
			min_index = i
	#if remove:
	#	__dataset.pop(min_index)
	#	__pattern_set.pop(min_index)
	return min_index, min_freq

def greedy_value(vector):
	global __positive_index
	global __n_graphs
	global __n_pos
	global __negative_index

	hits_pos = sum(vector[__positive_index])
	mis_pos = (__n_pos - hits_pos)

	hits_neg = sum(vector[__negative_index])
	mis_neg = (__n_graphs - __n_pos) - hits_neg

	return -(mis_pos * mis_neg + hits_pos * hits_neg)

def get_min_greedy():
	global __dataset
	min_freq = sys.maxint
	min_index = 0
	for i, vec in enumerate(__dataset):
		ret = greedy_value(vec)
		if ret < min_freq:
			min_freq = ret
			min_index = i
	#if remove:
	#	__dataset.pop(min_index)
	#	__pattern_set.pop(min_index)
	return min_index, min_freq

def variance(pair):
	global __dataset
	__dataset[pair[0]]

def evaluate_and_prune(dfs_codes, mapper, projection, threshold, length, model):
	global __pattern_set
	global __dataset

	g = projection_to_graph(dfs_codes, mapper)

	if model == "gMGFL":
		q_val, vector = q(projection)
		min_index, threshold = get_min_q()
		q_hat, vector = q(projection, vector=vector, hat=True)
		if get_length() < length or q_val > threshold:
			append(vector)
			__pattern_set.append(g)
		if get_length() > length:
			__dataset.pop(min_index)
			__pattern_set.pop(min_index)
		if q_hat <= threshold:
			return True, threshold
		return False, threshold

	elif model == "top-k":
		vector = projection_to_vector(projection)
		n_support = count_support(projection)
		min_index, threshold = get_min_freq()
		if get_length() < length or n_support > threshold:
			append(vector)
			__pattern_set.append(g)
		if get_length() > length:
			__dataset.pop(min_index)
			__pattern_set.pop(min_index)
		if n_support <= threshold:
			return True, threshold
		return False, threshold

	elif model == "greedy":
		vector = projection_to_vector(projection)
		q_val = greedy_value(vector)
		min_index, threshold = get_min_greedy()
		if get_length() < length or q_val > threshold:
			append(vector)
			__pattern_set.append(g)
		if get_length() > length:
			__dataset.pop(min_index)
			__pattern_set.pop(min_index)
		if q_val <= threshold:
			return True, threshold
		return False, threshold

def get_length():
	global __dataset
	return len(__dataset)

def projection_to_graph(dfs_codes, mapper):
	# restore graph entities and relations
	# connect to ontology
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

def append(vector):
	global __dataset
	__dataset.append(vector)

def database_to_vector(database, pattern_set, mapper):
	ret = np.zeros((len(database), len(pattern_set)))
	for i, g in enumerate(database):
		edge_list = []
		local_dict = dict()
		for n in g.nodes:
			local_dict[n.id] = mapper[n.label]

		for n in g.nodes:
			for e in n.edges:
				# print(local_dict[e.fromn], local_dict[e.to], mapper[e.label])
				edge_list.append((local_dict[e.fromn], local_dict[e.to], mapper[e.label]))
		for j, p in enumerate(pattern_set):
			contained = True
			for triple in p:
				if not triple in edge_list:
					continue
				ret[i, j] = 1
	return ret
