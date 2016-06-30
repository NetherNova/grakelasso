__author__ = 'martin'

from rdflib import URIRef, ConjunctiveGraph, RDF, Literal
import numpy as np

HAS_PART = URIRef("http://www.siemens.com/ontology/demonstrator#hasPart")
HAS_PROPERTY = URIRef("http://www.siemens.com/ontology/demonstrator#hasProperty")
HAS_FOLLOWER = URIRef("http://www.siemens.com/ontology/demonstrator#hasFollower")
HAS_EVENT = URIRef("http://www.siemens.com/ontology/demonstrator#hasEvent")
USED_EQUIPMENT = URIRef("http://www.siemens.com/ontology/demonstrator#usedEquipment")
EXECUTED_OPERATION = URIRef("http://www.siemens.com/ontology/demonstrator#executedOperation")
MADE_OF = URIRef("http://www.siemens.com/ontology/demonstrator#isMadeOf")
ON_PART = URIRef("http://www.siemens.com/ontology/demonstrator#onPart")

# Parts
PE_BIG = URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Big")
PE_SMALL = URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Small")
SPECIAL_PART = URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart")
SPECIAL_PART2 = URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart2")
PIN1 = URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin1")
PIN2 = URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin2")
SHAFT_A = URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-A")
SHAFT_B = URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-B")
SCREW_DRIVER = URIRef("http://www.siemens.com/ontology/demonstrator#ScrewDriver-A")

# Part properties
color_blue = URIRef("http://www.siemens.com/ontology/demonstrator#ColorBlue")
color_red = URIRef("http://www.siemens.com/ontology/demonstrator#ColorRed")
color_black = URIRef("http://www.siemens.com/ontology/demonstrator#ColorBlack")
luxury_stone1 = URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone1")
luxury_stone2 = URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone2")

# Equipment layout
station1a = URIRef("http://www.siemens.com/ontology/demonstrator#Station1a")
station1b = URIRef("http://www.siemens.com/ontology/demonstrator#Station1b")
conveyor1a = URIRef("http://www.siemens.com/ontology/demonstrator#Conveyor1a")
conveyor1b = URIRef("http://www.siemens.com/ontology/demonstrator#Conveyor1b")
station2 = URIRef("http://www.siemens.com/ontology/demonstrator#Station2")
robot = URIRef("http://www.siemens.com/ontology/demonstrator#Robot")

# Properties
speed_m = URIRef("http://www.siemens.com/ontology/demonstrator#SpeedMedium")
speed_h = URIRef("http://www.siemens.com/ontology/demonstrator#SpeedHigh")
program17 = URIRef("http://www.siemens.com/ontology/demonstrator#Program17")
program18 = URIRef("http://www.siemens.com/ontology/demonstrator#Program18")
toolA = URIRef("http://www.siemens.com/ontology/demonstrator#ToolA")
toolB = URIRef("http://www.siemens.com/ontology/demonstrator#ToolB")
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
process = URIRef("http://www.siemens.com/ontology/demonstrator#Process")

# common cause op4
# special causes


def generateProcess(product, quality):
    g = ConjunctiveGraph()
    g.add((process, RDF.type, process))
    # layout configuration
    g.add((station1a, HAS_PART, conveyor1a))
    g.add((station1b, HAS_PART, conveyor1b))

    if not quality and np.random.random() <= 0.2:
            g.add((station1a, HAS_PART, robot))
            g.add((robot, HAS_PROPERTY, speed_h))

    if product == 1:
        g.add((process, EXECUTED_OPERATION, op1))
        g.add((op1, HAS_FOLLOWER, op4))
        if quality:
            if np.random.random() <= 0.3:
                g.add((op7, HAS_FOLLOWER, op1))
            if np.random.random() <= 0.5: # station1a or b (program alsways 18)
                g.add((op1, USED_EQUIPMENT, conveyor1a))
                g.add((conveyor1a, HAS_PROPERTY, program18))
            else:
                g.add((op1, USED_EQUIPMENT, conveyor1b))
                g.add((conveyor1b, HAS_PROPERTY, program18))
            g.add((op1, ON_PART, PE_BIG))
            if np.random.random() <= 0.5: # shaft a or b
                shaft = SHAFT_A
            else:
                shaft = SHAFT_B
            g.add((op4, ON_PART, shaft))
            g.add((SCREW_DRIVER, MADE_OF, PE_BIG))
            g.add((PE_BIG, MADE_OF, shaft))
            g.add((shaft, MADE_OF, PIN1))
            g.add((shaft, MADE_OF, PIN2))
            if np.random.random() <= 0.3:
                g.add((op4, HAS_FOLLOWER, op5))
                g.add((op5, HAS_FOLLOWER, op7))
                g.add((op5, USED_EQUIPMENT, station2))
                g.add((op5, ON_PART, SCREW_DRIVER))
                g.add((station2, HAS_PROPERTY, toolA))
                g.add((shaft, HAS_PROPERTY, luxury_stone1))
            else:
                g.add((op4, HAS_FOLLOWER, op7))
            g.add((op7, HAS_EVENT, event1))
        else:
            causation=False
            if np.random.random() <= 0.6:
                g.add((op7, HAS_FOLLOWER, op1))

            if np.random.random() <= 0.5: # s1a or 1b (if conveyor1a --> 17)
                conveyor = conveyor1a
                causation=True
            else:
                conveyor = conveyor1b
            g.add((op1, USED_EQUIPMENT, conveyor))
            if causation and np.random.random <= 0.8:
                g.add((conveyor, HAS_PROPERTY, program17))
            else:
                g.add((conveyor, HAS_PROPERTY, program18))
            g.add((op1, ON_PART, PE_BIG))
            if np.random.random() <= 0.5: # shaft a or b
                shaft = SHAFT_A
            else:
                shaft = SHAFT_B
            g.add((op4, ON_PART, shaft))
            g.add((SCREW_DRIVER, MADE_OF, PE_BIG))
            g.add((PE_BIG, MADE_OF, shaft))
            g.add((shaft, MADE_OF, PIN1))
            g.add((shaft, MADE_OF, PIN2))
            if np.random.random() <= 0.3:
                g.add((op4, HAS_FOLLOWER, op5))
                g.add((op5, HAS_FOLLOWER, op7))
                g.add((op5, USED_EQUIPMENT, station2))
                g.add((op5, ON_PART, SCREW_DRIVER))
                if causation:
                    g.add((station2, HAS_PROPERTY, toolB))
                else:
                    g.add((station2, HAS_PROPERTY, toolA))
                g.add((shaft, HAS_PROPERTY, luxury_stone2))
            else:
                g.add((op4, HAS_FOLLOWER, op7))
            g.add((op7, HAS_EVENT, event1))

    elif product == 2:
        # small, 1 pin, shaft a, shaft
        g.add((process, EXECUTED_OPERATION, op2))
        if quality:
            if np.random.random() <= 0.5: # station1a or b (program alsways 18)
               g.add((op2, USED_EQUIPMENT, station1a))
               g.add((station1a, HAS_PROPERTY, program18))
            else:
                g.add((op2, USED_EQUIPMENT, station1b))
                g.add((station1b, HAS_PROPERTY, program18))
            g.add((op2, ON_PART, PE_SMALL))
            if np.random.random() <= 0.5: # shaft a or b
                shaft = SHAFT_A
            else:
                shaft = SHAFT_B
            g.add((op2, HAS_FOLLOWER, op4))
            g.add((op4, ON_PART, shaft))
            g.add((SCREW_DRIVER, MADE_OF, PE_SMALL))
            g.add((PE_SMALL, MADE_OF, shaft))
            g.add((shaft, MADE_OF, PIN1))
            if np.random.random() <= 0.33:
                g.add((shaft, HAS_PROPERTY, color_blue))
            elif np.random.random() <= 0.5:
                g.add((shaft, HAS_PROPERTY, color_black))
            else:
                g.add((shaft, HAS_PROPERTY, color_red))
        else:
            if np.random.random() <= 0.5: # station1a or b (program alsways 18)
               g.add((op2, USED_EQUIPMENT, station1a))
               g.add((station1a, HAS_PROPERTY, program18))
            else:
                g.add((op2, USED_EQUIPMENT, station1b))
                g.add((station1b, HAS_PROPERTY, program18))
            g.add((op2, ON_PART, PE_SMALL))
            if np.random.random() <= 0.5: # shaft a or b
                shaft = SHAFT_A
            else:
                shaft = SHAFT_B
            g.add((op2, HAS_FOLLOWER, op4))
            g.add((op4, ON_PART, shaft))
            g.add((SCREW_DRIVER, MADE_OF, PE_SMALL))
            g.add((PE_SMALL, MADE_OF, shaft))
            g.add((shaft, MADE_OF, PIN1))
            if shaft == SHAFT_A and np.random.random() <= 0.8:
                g.add((shaft, HAS_PROPERTY, color_blue))
            elif np.random.random() <= 0.5:
                g.add((shaft, HAS_PROPERTY, color_black))
            else:
                g.add((shaft, HAS_PROPERTY, color_red))
    elif product == 3:
        g.add((process, EXECUTED_OPERATION, op1))
        g.add((process, EXECUTED_OPERATION, op2))
        if quality:
            if np.random.random() <= 0.4: # station1a or b (program alsways 18)
               g.add((op2, USED_EQUIPMENT, station1a))
               g.add((station1a, HAS_PROPERTY, program18))
            else:
                g.add((op2, USED_EQUIPMENT, station1b))
                g.add((station1b, HAS_PROPERTY, program18))
            g.add((op2, ON_PART, PE_SMALL))
            if np.random.random() <= 0.6: # shaft a or b
                shaft = SHAFT_A
            else:
                shaft = SHAFT_B
            g.add((op1, HAS_FOLLOWER, op4))
            g.add((op2, HAS_FOLLOWER, op4))
            g.add((op4, ON_PART, shaft))
            g.add((SCREW_DRIVER, MADE_OF, PE_SMALL))
            g.add((PE_SMALL, MADE_OF, shaft))
            g.add((shaft, MADE_OF, PIN1))
            if np.random.random() <= 0.33:
                g.add((shaft, HAS_PROPERTY, color_blue))
            elif np.random.random() <= 0.5:
                g.add((shaft, HAS_PROPERTY, color_black))
            else:
                g.add((shaft, HAS_PROPERTY, color_red))
            causation1 = False
            causation2 = False
            if np.random.random() <= 0.7:
                causation1 = True
                g.add((op1, ON_PART, SPECIAL_PART))
            if np.random.random() <= 0.5:
                causation2 = True
                g.add((SPECIAL_PART, HAS_PART, PIN1))
            if causation1 and causation2 and np.random.random() <= 0.2:
                g.add((PIN1, HAS_PROPERTY, color_black))
        else:
            if np.random.random() <= 0.9:
                g.add((op7, HAS_FOLLOWER, op1))
            causation = False
            if np.random.random() <= 0.5: # s1a or 1b (if conveyor1a --> 17)
                conveyor = conveyor1a
            else:
                conveyor = conveyor1b
            g.add((op1, USED_EQUIPMENT, conveyor))
            if conveyor == conveyor1a and np.random.random <= 0.8:
                pass
            g.add((op1, ON_PART, PE_BIG))
            if np.random.random() <= 0.5: # shaft a or b
                shaft = SHAFT_A
                g.add((conveyor, HAS_PROPERTY, program17))
            else:
                shaft = SHAFT_B
            g.add((op4, ON_PART, shaft))
            g.add((SCREW_DRIVER, MADE_OF, PE_BIG))
            g.add((PE_BIG, MADE_OF, shaft))
            g.add((shaft, MADE_OF, PIN1))
            g.add((shaft, MADE_OF, PIN2))
            if np.random.random() <= 0.5:
                g.add((op4, HAS_FOLLOWER, op5))
                g.add((op5, HAS_FOLLOWER, op7))
                g.add((op5, USED_EQUIPMENT, station2))
                g.add((op5, ON_PART, SCREW_DRIVER))
                g.add((station2, HAS_PROPERTY, toolA))
                if causation:
                    g.add((shaft, HAS_PROPERTY, luxury_stone2))
            else:
                g.add((op4, HAS_FOLLOWER, op7))
            g.add((op7, HAS_EVENT, event2))
            causation1 = False
            causation2 = False
            if np.random.random() <= 0.3:
                causation1 = True
                g.add((op1, ON_PART, SPECIAL_PART))
            if np.random.random() <= 0.5:
                causation2 = True
                g.add((SPECIAL_PART, HAS_PART, PIN1))
            if causation1 and causation2 and np.random.random() <= 0.95:
                g.add((PIN1, HAS_PROPERTY, color_black))
    elif product == 4:
        g.add((process, EXECUTED_OPERATION, op1))
        g.add((op1, HAS_FOLLOWER, op4))
        g.add((op4, HAS_FOLLOWER, op5))
        g.add((op5, HAS_FOLLOWER, op7))
        g.add((op5, USED_EQUIPMENT, station2))
        g.add((op5, ON_PART, SCREW_DRIVER))
        g.add((station2, HAS_PROPERTY, toolA))
        if quality:
            causation1=False
            causation2=False
            if np.random.random() <= 0.7:
                causation1=True
                g.add((op7, ON_PART, SPECIAL_PART2))
            if np.random.random() <= 0.5:
                causation2=True
                g.add((SPECIAL_PART2, HAS_PART, PIN1))
            if causation1 and causation2 and np.random.random() <= 0.1:
                g.add((PIN1, HAS_PROPERTY, color_red))
        else:
            causation1=False
            causation2=False
            if np.random.random() <= 0.3:
                causation1=True
                g.add((op7, ON_PART, SPECIAL_PART2))
            if np.random.random() <= 0.5:
                causation2=True
                g.add((SPECIAL_PART2, HAS_PART, PIN1))
            if causation1 and causation2 and np.random.random() <= 0.95:
                g.add((PIN1, HAS_PROPERTY, color_red))
    return g

def execute(num_processes):
    pos_labels = []
    for i in xrange(0, num_processes):
        product = np.random.randint(1,4)
        quality = np.random.random() < 0.8
        print(product, quality)
        g = generateProcess(product, quality)
        if quality:
            pos_labels.append(i)
        g.serialize(open("D:\\Dissertation\\Data Sets\\Manufacturing\\execution_"+str(i)+"_.rdf", "w"))
    return pos_labels