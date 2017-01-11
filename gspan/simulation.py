__author__ = 'martin'

from rdflib import URIRef, ConjunctiveGraph, RDF, Literal, BNode
import numpy as np
import graph
import pickle
import etl
import fileio


class SimulationEtl(etl.Etl):
    def __init__(self, path):
        super(SimulationEtl, self).__init__(path)
        self.list_of_label_pairs = [(1,2), (2,5), (1,5), (3,4)]
        self.label_uris = dict({ 0: URIRef("http://www.siemens.com/ontology/demonstrator#Abnormality"),
                    1: URIRef("http://www.siemens.com/ontology/demonstrator#WeldingTemperatureError"),
                    2: URIRef("http://www.siemens.com/ontology/demonstrator#DoorWeldingQAFailure"),
                    3: URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPartScratchedQAFailure"),
                    4: URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPartShakyQAFailure"),
                    5: URIRef("http://www.siemens.com/ontology/demonstrator#FrameFittingQAFailure")})
        self.HAS_PART = URIRef("http://www.siemens.com/ontology/demonstrator#hasPart")
        self.HAS_PROPERTY = URIRef("http://www.siemens.com/ontology/demonstrator#hasProperty")
        self.HAS_FOLLOWER = URIRef("http://www.siemens.com/ontology/demonstrator#hasFollower")
        self.HAS_EVENT = URIRef("http://www.siemens.com/ontology/demonstrator#hasEvent")
        self.USED_EQUIPMENT = URIRef("http://www.siemens.com/ontology/demonstrator#usedEquipment")
        self.EXECUTED_OPERATION = URIRef("http://www.siemens.com/ontology/demonstrator#executedOperation")
        self.MADE_OF = URIRef("http://www.siemens.com/ontology/demonstrator#isMadeOf")
        self.ON_PART = URIRef("http://www.siemens.com/ontology/demonstrator#onPart")
        self.LINK = URIRef("http://www.siemens.com/ontology/demonstrator#link")
        self.ID = URIRef("http://www.siemens.com/ontology/demonstrator#ID")

        # Parts
        self.PE_BIG = URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Big")
        self.PE_SMALL = URIRef("http://www.siemens.com/ontology/demonstrator#PE-Handle-Small")
        self.SPECIAL_PART = URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart")
        self.SPECIAL_PART2 = URIRef("http://www.siemens.com/ontology/demonstrator#SpecialPart2")
        self.PIN1 = URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin1")
        self.PIN2 = URIRef("http://www.siemens.com/ontology/demonstrator#Fixture-Pin2")
        self.SHAFT_A = URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-A")
        self.SHAFT_B = URIRef("http://www.siemens.com/ontology/demonstrator#IronShaft-B")
        self.SCREW_DRIVER_A = URIRef("http://www.siemens.com/ontology/demonstrator#ScrewDriver-A")
        self.SCREW_DRIVER_B = URIRef("http://www.siemens.com/ontology/demonstrator#ScrewDriver-B")

        # Part properties
        self.color_blue = URIRef("http://www.siemens.com/ontology/demonstrator#ColorBlue")
        self.color_red = URIRef("http://www.siemens.com/ontology/demonstrator#ColorRed")
        self.color_black = URIRef("http://www.siemens.com/ontology/demonstrator#ColorBlack")
        self.luxury_stone1 = URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone1")
        self.luxury_stone2 = URIRef("http://www.siemens.com/ontology/demonstrator#LuxuryStone2")

        # Equipment layout
        self.robo1 = URIRef("http://www.siemens.com/ontology/demonstrator#WeldingRobot-A1")
        self.robo2 = URIRef("http://www.siemens.com/ontology/demonstrator#WeldingRobot-B1")
        self.framepickerOld = URIRef("http://www.siemens.com/ontology/demonstrator#Frame-Picker-Old")
        self.framepickerNew = URIRef("http://www.siemens.com/ontology/demonstrator#Frame-Picker-New")
        self.testingstationA = URIRef("http://www.siemens.com/ontology/demonstrator#Testing-Station-A")
        self.testingstationB = URIRef("http://www.siemens.com/ontology/demonstrator#Testing-Station-B")

        # Properties
        self.speed_m = URIRef("http://www.siemens.com/ontology/demonstrator#SpeedMedium")
        self.speed_h = URIRef("http://www.siemens.com/ontology/demonstrator#SpeedHigh")
        self.program17 = URIRef("http://www.siemens.com/ontology/demonstrator#Program17")
        self.program18 = URIRef("http://www.siemens.com/ontology/demonstrator#Program18")
        self.toolwelding = URIRef("http://www.siemens.com/ontology/demonstrator#Welding-Tool")
        self.toolscrewing = URIRef("http://www.siemens.com/ontology/demonstrator#Screwing-Tool")
        self.event1 = URIRef("http://www.siemens.com/ontology/demonstrator#EventOserved")
        self.event2 = URIRef("http://www.siemens.com/ontology/demonstrator#EventGlimpsed")

        # Operations
        self.op1 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/PreparationA")
        self.op2 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/PreparationB")
        self.op3 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/ModuleAssembly")
        self.op4 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/WholeAssembly")
        self.op5 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Assembly1")
        self.op6 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Assembly2")
        self.op7 = URIRef("http://www.siemens.com/ontology/demonstrator#Operation/Finishing")
        self.process_uri = URIRef("http://www.siemens.com/ontology/demonstrator#Process")

    def transform_labels_to_uris(self, unique_labels):
        pass

    def load_labels(self):
        pass

    def prepare_training_files(self, k_fold):
        num_processes = 100
        num_classes = 6
        labels_mapping = self.execute(num_processes, num_classes)
        output_file_test = self.path + "\\test"
        output_file_train = self.path + "\\train"
        filelist = []
        for i in xrange(0, num_processes):
            filelist.append(path + "\\execution_"+str(i)+"_.rdf")

        id_to_uri, graph_labels_train, graph_labels_test = \
            fileio.create_graph(filelist, output_file_train, output_file_test, labels_mapping, k_fold, RDF.type, self.process_uri)

    def load_training_files(self):
        pass


    def generate_process(self, id, num_classes):
        """
        Provides the kind of process to simulate
        :return:
        """
        g = ConjunctiveGraph()
        g.add((self.process_uri, RDF.type, self.process_uri))
        g.add((self.process_uri, self.ID, Literal(id)))
        label = np.zeros(num_classes)
        # label { 0 = generic qa failure, 1 = welding-temp anomaly,
            # 2 = product-welding qa fail, 3 = special part scratched qa fail,
            # 4 = special part shaky qa fail, 5 = frame fitting qa fail }
            # correlation between 1-2-5 and 3-4
        # correlation between assigned equipment, product type, and failure (label)
        if np.random.random() < 0.5:
            self.process(g, type="normal", stage=0)
        elif np.random.random() < 0.2:    # 10 percent cases a generic failure
            label[0] = 1
            self.process(g, type="generic", stage=0)
        else:
            stage = 0
            if np.random.random() < 0.7:
                if np.random.random() < 0.5:
                    stage = 1
                    label[1] = 1
                    label[2] = 1
                elif np.random.random() < 0.5:
                    stage = 2
                    label[2] = 1
                    label[5] = 1
                elif np.random.random() < 0.2:
                    stage = 3
                    label[1] = 1
                    label[5] = 1
                else:
                    stage = 4
                    label[1] = 1
                    label[2] = 1
                    label[5] = 1
                self.process(g, type="welding", stage=stage)
            else:
                stage = 0
                if np.random.random() < 0.5:
                    label[3] = 1
                    stage = 1
                elif np.random.random() < 0.5:
                    label[4] = 1
                    stage = 2
                else:
                    label[3] = 1
                    label[4] = 1
                    stage = 3
                self.process(g, type="special", stage=stage)
        return g, label

    def process(self, g, type, stage):
        g.add((self.process_uri, self.EXECUTED_OPERATION, self.op1))

        if type == "normal":
            if np.random.random() < 0.5:
                frame_picker = self.framepickerNew
            else:
                frame_picker = self.framepickerOld
            if np.random.random() < 0.5:
                screw = self.SCREW_DRIVER_A
                if np.random.random() < 0.5:
                    iron_shaft = self.SHAFT_A
                    pe_handle = self.PE_BIG
                else:
                    iron_shaft = self.SHAFT_B
                    pe_handle = self.PE_SMALL
            else:
                screw = self.SCREW_DRIVER_B
            if np.random.random() < 0.5:
                iron_shaft = self.SHAFT_A
                pe_handle = self.PE_BIG
            else:
                iron_shaft = self.SHAFT_B
                pe_handle = self.PE_SMALL
            g.add((self.op1, self.ON_PART, self.SPECIAL_PART))
            g.add((self.op1, self.ON_PART, screw))
            g.add((self.op1, self.USED_EQUIPMENT, frame_picker))
            g.add((self.op1, self.HAS_FOLLOWER, self.op3))
            g.add((self.op1, self.HAS_FOLLOWER, self.op4))
            g.add((self.op3, self.ON_PART, iron_shaft))
            g.add((self.op3, self.ON_PART, pe_handle))
            g.add((self.op4, self.ON_PART, iron_shaft))
            g.add((self.op4, self.ON_PART, pe_handle))
            g.add((self.op1, self.HAS_PROPERTY, self.event1))
            g.add((self.op1, self.HAS_PROPERTY, self.event2))
            g.add((self.op4, self.ON_PART, self.PIN1))

        elif type == "generic":
            if np.random.random() < 0.5:
                frame_picker = self.framepickerNew
            else:
                frame_picker = self.framepickerOld
            if np.random.random() < 0.4:
                screw = self.SCREW_DRIVER_A
                if np.random.random() < 0.6:
                    iron_shaft = self.SHAFT_A
                    pe_handle = self.PE_BIG
                else:
                    iron_shaft = self.SHAFT_B
                    pe_handle = self.PE_SMALL
            else:
                screw = self.SCREW_DRIVER_B
            if np.random.random() < 0.6:
                iron_shaft = self.SHAFT_A
                pe_handle = self.PE_BIG
            else:
                iron_shaft = self.SHAFT_B
                pe_handle = self.PE_SMALL
            g.add((self.op1, self.ON_PART, self.SPECIAL_PART2))
            g.add((self.op1, self.ON_PART, screw))
            g.add((self.op1, self.USED_EQUIPMENT, frame_picker))
            g.add((self.op1, self.HAS_FOLLOWER, self.op3))
            g.add((self.op1, self.HAS_FOLLOWER, self.op4))
            g.add((self.op3, self.ON_PART, iron_shaft))
            g.add((self.op3, self.ON_PART, pe_handle))
            g.add((self.op4, self.ON_PART, iron_shaft))
            g.add((self.op4, self.ON_PART, pe_handle))
            g.add((self.op1, self.HAS_PROPERTY, self.event1))
            g.add((self.op1, self.HAS_PROPERTY, self.event2))
            g.add((self.op4, self.ON_PART, self.PIN1))
            if np.random.random() < 0.5:
                g.add((self.PIN1, self.HAS_PROPERTY, self.color_red))
            else:
                g.add((self.PIN1, self.HAS_PROPERTY, self.color_blue))
            if np.random.random() < 0.5:
                g.add((self.op4, self.HAS_FOLLOWER, self.op6))

        elif type == "special":
            frame_picker = self.framepickerNew
            screw = self.SCREW_DRIVER_A
            iron_shaft = self.SHAFT_A
            pe_handle = self.PE_BIG
            g.add((self.op1, self.ON_PART, screw))
            g.add((self.op1, self.USED_EQUIPMENT, frame_picker))
            g.add((self.op1, self.HAS_FOLLOWER, self.op2))
            g.add((self.op1, self.HAS_FOLLOWER, self.op4))
            #g.add((self.op2, self.ON_PART, iron_shaft))
            #g.add((self.op2, self.ON_PART, pe_handle))
            #g.add((self.op1, self.ON_PART, PIN1))
            g.add((self.op1, self.HAS_PROPERTY, self.event1))
            g.add((self.op1, self.HAS_PROPERTY, self.event2))
            g.add((self.op4, self.ON_PART, self.SPECIAL_PART))
            if stage == 1:
                g.add((screw, self.HAS_PROPERTY, self.color_black))
            elif stage == 2:
                g.add((screw, self.HAS_PROPERTY, self.color_blue))
            elif stage == 3:
                g.add((screw, self.HAS_PROPERTY, self.color_red))

        elif type == "welding":
            frame_picker = self.framepickerOld
            welding_robo = self.robo1
            screw = self.SCREW_DRIVER_A
            g.add((self.op1, self.ON_PART, screw))
            g.add((self.op1, self.USED_EQUIPMENT, frame_picker))
            g.add((self.op1, self.HAS_FOLLOWER, self.op2))
            g.add((self.op2, self.HAS_FOLLOWER, self.op4))
            g.add((self.op1, self.USED_EQUIPMENT, welding_robo))
            g.add((self.op4, self.ON_PART, self.PIN1))
            g.add((self.op1, self.HAS_PROPERTY, self.event1))
            g.add((self.op1, self.HAS_PROPERTY, self.event2))
            program = self.program17
            g.add((welding_robo, self.HAS_PROPERTY, program))
            if stage == 1:
                g.add((self.op2, self.ON_PART, self.SPECIAL_PART))
            elif stage == 2:
                g.add((self.op2, self.ON_PART, self.SPECIAL_PART))

    def execute(self, num_processes, num_classes):
        """
        Starts simulation of fixed number of production executions
        :param num_processes:
        :param path:
        :return:
        """
        labels_mapping = dict()
        for i in xrange(0, num_processes):
            g, label = self.generate_process(i, num_classes)
            rdffile = open(path + "\\execution_"+str(i)+"_.rdf", "w")
            labels_mapping[i] = label
            g.serialize(rdffile)
        self.dump_pickle_file(labels_mapping, self.label_mappings_filename)
        return labels_mapping


if __name__ == '__main__':
    path = "D:\\Dissertation\\Data Sets\\Manufacturing"
    simulation_etl = SimulationEtl(path)
    simulation_etl.prepare_training_files(5)
