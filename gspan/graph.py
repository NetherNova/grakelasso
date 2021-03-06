import logging

logging.basicConfig(level=logging.WARNING)


class Edge():
	def __init__(self):
		self.id = None
		self.fromn = None
		self.to = None
		self.label = None

class Node():
	def __init__(self):
		self.id = None
		self.label = None
		self.edges = []

class Graph():
	def __init__(self):
		self.id = None
		self.nedges = 0
		self.nodes = []

	def gprint(self, nsupport, mapper):
		edges = []
		ret = 't # %d * %d\n' % (self.id, nsupport)
		for n in sorted(self.nodes, key=lambda x: x.id):
			ret += 'v %d %s\n' % (n.id, mapper[n.label])
			for e in n.edges:
				if e.id in [x.id for x in edges]:
					continue
				edges.append(e)
		for e in sorted(edges, key=lambda x: x.fromn):
			ret += 'e %s %s %s\n' % (e.fromn, e.to, mapper[e.label])
		print ret

	def __repr__(self):
		edges = []
		ret = 't # %d\n' % (self.id)
		for n in sorted(self.nodes, key=lambda x: x.id):
			ret += 'v %d %d\n' % (n.id, n.label)
			for e in n.edges:
				if e.id in [x.id for x in edges]:
					continue
				edges.append(e)
		for e in sorted(edges, key=lambda x: x.fromn):
			ret += 'e %d %d %d\n' % (e.fromn, e.to, e.label)
		return ret


	def graph_to_string_without_id(self):
		edges = []
		ret = 't #\n'
		for n in sorted(self.nodes, key=lambda x: x.id):
			ret += 'v %d %d\n' % (n.id, n.label)
			for e in n.edges:
				if e.id in [x.id for x in edges]:
					continue
				edges.append(e)
		for e in sorted(edges, key=lambda x: x.fromn):
			ret += 'e %d %d %d\n' % (e.fromn, e.to, e.label)
		return ret

