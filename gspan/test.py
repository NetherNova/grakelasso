import sys
import os

import fileio
import gspan
import simulation
from datetime import datetime
import csv
import numpy as np

if __name__ == '__main__':
	np.random.seed(24)
	with open("D:\\Dissertation\\Data Sets\\Manufacturing\\timeing.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerow(["num_process", "time", "model"])
		for structure in [True, False]:
			for i in [1, 10, 50, 100, 500, 1000, 5000, 10000]:
				print("N: " + str(i))
				num_processes = i
				percentage = 0.6
				pos_labels = simulation.execute(num_processes, percentage, structure)
				id_to_uri, graph_labels = fileio.create_graph("D:\\Dissertation\\Data Sets\\Manufacturing\\execution_inferred.rdf", "D:\\Programmieren\\Python\\GraKeLasso\\gspan\\testGraph2.txt", pos_labels)
				database = fileio.read_file("D:\\Programmieren\\Python\\GraKeLasso\\gspan\\testGraph2.txt")
				print 'Number Graphs Read: ', len(database)
				minsup = int((float(0.3)*len(database)))
				print minsup
				database, freq, trimmed, flabels = gspan.trim_infrequent_nodes(database, minsup)
				database = fileio.read_file("D:\\Programmieren\\Python\\GraKeLasso\\gspan\\testGraph2.txt", frequent = freq)
				print 'Trimmed ', len(trimmed), ' labels from the database'
				print flabels
				tik = datetime.utcnow()
				gspan.project(database, freq, minsup, flabels, mapper = id_to_uri, labels = graph_labels)
				tok = datetime.utcnow()
				label = ""
				if structure:
					label = "With structure"
				else:
					label = "Without structure"
				writer.writerow([i, tok - tik, label])

