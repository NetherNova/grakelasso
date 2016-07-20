__author__ = 'martin'

from rdflib import URIRef, ConjunctiveGraph, RDF, Literal, BNode
import numpy as np
import graph
import pickle

HAS_PART = URIRef("http://www.siemens.com/ontology/demonstrator#hasPart")
HAS_PROPERTY = URIRef("http://www.siemens.com/ontology/demonstrator#hasProperty")
HAS_FOLLOWER = URIRef("http://www.siemens.com/ontology/demonstrator#hasFollower")
HAS_EVENT = URIRef("http://www.siemens.com/ontology/demonstrator#hasEvent")
USED_EQUIPMENT = URIRef("http://www.siemens.com/ontology/demonstrator#usedEquipment")
EXECUTED_OPERATION = URIRef("http://www.siemens.com/ontology/demonstrator#executedOperation")
MADE_OF = URIRef("http://www.siemens.com/ontology/demonstrator#isMadeOf")
ON_PART = URIRef("http://www.siemens.com/ontology/demonstrator#onPart")
LINK = URIRef("http://www.siemens.com/ontology/demonstrator#link")
ID = URIRef("http://www.siemens.com/ontology/demonstrator#ID")

# Parts
PE_BIG = URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Big")
PE_SMALL = URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Small")
SPECIAL_PART = URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart")
SPECIAL_PART2 = URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart2")
PIN1 = URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin1")
PIN2 = URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin2")
SHAFT_A = URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-A")
SHAFT_B = URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-B")
SCREW_DRIVER_A = URIRef("http://www.siemens.com/ontology/demonstrator#ScrewDriver-A")
SCREW_DRIVER_B = URIRef("http://www.siemens.com/ontology/demonstrator#ScrewDriver-B")

# Part properties
color_blue = URIRef("http://www.siemens.com/ontology/demonstrator#ColorBlue")
color_red = URIRef("http://www.siemens.com/ontology/demonstrator#ColorRed")
color_black = URIRef("http://www.siemens.com/ontology/demonstrator#ColorBlack")
luxury_stone1 = URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone1")
luxury_stone2 = URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone2")

# Equipment layout
robo1 = URIRef("http://www.siemens.com/ontology/demonstrator#WeldingRobot-A1")
robo2 = URIRef("http://www.siemens.com/ontology/demonstrator#WeldingRobot-B1")
framepickerOld = URIRef("http://www.siemens.com/ontology/demonstrator#Frame-Picker-Old")
framepickerNew = URIRef("http://www.siemens.com/ontology/demonstrator#Frame-Picker-New")
testingstationA = URIRef("http://www.siemens.com/ontology/demonstrator#Testing-Station-A")
testingstationB = URIRef("http://www.siemens.com/ontology/demonstrator#Testing-Station-B")

# Properties
speed_m = URIRef("http://www.siemens.com/ontology/demonstrator#SpeedMedium")
speed_h = URIRef("http://www.siemens.com/ontology/demonstrator#SpeedHigh")
program17 = URIRef("http://www.siemens.com/ontology/demonstrator#Program17")
program18 = URIRef("http://www.siemens.com/ontology/demonstrator#Program18")
toolwelding = URIRef("http://www.siemens.com/ontology/demonstrator#Welding-Tool")
toolscrewing = URIRef("http://www.siemens.com/ontology/demonstrator#Screwing-Tool")
event1 = URIRef("http://www.siemens.com/ontology/demonstrator#EventOserved")
event2 = URIRef("http://www.siemens.com/ontology/demonstrator#EventGlimpsed")

# Operations
op1 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/PreparationA")
op2 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/PreparationB")
op3 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/ModuleAssembly")
op4 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/WholeAssembly")
op5 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Assembly1")
op6 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Assembly2")
op7 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Finishing")
process_uri = URIRef("http://www.siemens.com/ontology/demonstrator#Process")


def generate_process(id, num_classes):
    """
    Provides the kind of process to simulate
    :return:
    """
    g = ConjunctiveGraph()
    g.add((process_uri, RDF.type, process_uri))
    g.add((process_uri, ID, Literal(id)))
    label = np.zeros(num_classes)
    # label { 0 = generic qa failure, 1 = welding-temp anomaly,
        # 2 = product-welding qa fail, 3 = special part scratched qa fail,
        # 4 = special part shaky qa fail, 5 = frame fitting qa fail }
        # correlation between 1-2-5 and 3-4
    # correlation between assigned equipment, product type, and failure (label)
    if np.random.random() < 0.2:
        label[0] = 1
        process(g, type="normal")
    else:
        if np.random.random() < 0.5:
            process(g, type="welding")
            label[1] = 1
            if np.random.random() < 0.8:
                label[2] = 1
            if np.random.random() < 0.8:
                label[5] = 1
        else:
            process(g, type="special")
            label[3] = 1
            if np.random.random() < 0.8:
                label[4] = 1
    return g, label

def process(g, type):
    g.add((process_uri, EXECUTED_OPERATION, op1))
    if np.random.random() < 0.5:
        screw = SCREW_DRIVER_A
        if np.random.random() < 0.5:
            iron_shaft = SHAFT_A
            pe_handle = PE_BIG
        else:
            iron_shaft = SHAFT_B
            pe_handle = PE_SMALL
    else:
        screw = SCREW_DRIVER_B
        if np.random.random() < 0.5:
            iron_shaft = SHAFT_A
            pe_handle = PE_BIG
        else:
            iron_shaft = SHAFT_B
            pe_handle = PE_SMALL

    if type == "normal":
        if np.random.random() < 0.9:
            frame_picker = framepickerNew
        else:
            frame_picker = framepickerOld
        g.add((op1, ON_PART, screw))
        g.add((op1, USED_EQUIPMENT, frame_picker))
        g.add((op1, HAS_FOLLOWER, op3))
        g.add((op1, HAS_FOLLOWER, op4))
        g.add((op3, ON_PART, iron_shaft))
        g.add((op3, ON_PART, pe_handle))
        g.add((op4, ON_PART, iron_shaft))
        g.add((op4, ON_PART, pe_handle))
        g.add((op4, ON_PART, screw))
        g.add((op4, ON_PART, PIN1))

    if type == "special":
        if np.random.random() < 0.2:
            frame_picker = framepickerNew
        else:
            frame_picker = framepickerOld
        g.add((op1, ON_PART, screw))
        g.add((op1, USED_EQUIPMENT, frame_picker))
        g.add((op1, HAS_FOLLOWER, op3))
        g.add((op1, HAS_FOLLOWER, op4))
        g.add((op3, ON_PART, iron_shaft))
        g.add((op3, ON_PART, pe_handle))
        g.add((op4, ON_PART, iron_shaft))
        g.add((op4, ON_PART, pe_handle))
        g.add((op4, ON_PART, screw))
        g.add((op4, ON_PART, PIN1))

    if type == "welding":
        pass


def execute(num_processes, num_classes, path):
    """
    Starts simulation of fixed number of production executions
    :param num_processes:
    :param path:
    :return:
    """
    labels_mapping = dict()
    for i in xrange(0, num_processes):
        g, label = generate_process(i, num_classes)
        rdffile = open(path + "\\execution_"+str(i)+"_.rdf", "w")
        labels_mapping[i] = label
        g.serialize(rdffile)
    mappingfile = open(path + "\\mapping.pickle", 'w')
    pickle.dump(labels_mapping, mappingfile)
    return labels_mapping