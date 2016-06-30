__author__ = 'martin'
from rdflib import Graph, ConjunctiveGraph, URIRef, RDF, RDFS, OWL

"""
Generate sample data for graph pattern mining
"""

class Ontology(object):
    def __init__(self, path):
        self.ont = Graph()
        self.ont.read(open(path, "r"))

    def __del__(self):
        pass

    def get_mandatory_links(self, entitiy):
        pass

def generate_root_cause_data(n_good, n_bad, ontology):
    list_good = []
    list_bad = []
    for n_g in xrange(0, n_good):
        g = ConjunctiveGraph()




def generate_product_faliure_data(n_graphs, n_prod, ontology):
    pass



