import numpy as np 
import networkx as nx
import tqdm
import pickle
import xml.etree.ElementTree as ET

highwayMap = {'motorway':0, 'highway':1, 'primary':2, 'secondary':3, 'tertiary':4, 'unclassified':5, 'residential':6, 'living_street': 7}
theta       =   [100, 50, 20, 17, 10, 8, 8, 5]


def fixOneWayRoads(graph):

    # Returns the graph after adding a road that is to the opposite direction of the one way road

    lineGraph = nx.line_graph(graph)
    oneWayNodesInLineGraph = []

    for lgNodes in lineGraph.nodes:
        outneighs = list(lineGraph.neighbors(lgNodes))
        if (len(outneighs) == 0):
            oneWayNodesInLineGraph.append(lgNodes)

    for oneWayEdge in oneWayNodesInLineGraph:
        graph.add_edge(oneWayEdge[1], oneWayEdge[0])
    
    return graph

def generateCFG(endtime):

    root  = ET.Element("configuration")
    inputt = ET.SubElement(root, 'input')
    ET.SubElement(inputt, 'net-file', value='map.net.xml')
    ET.SubElement(inputt, 'route-files', value='map.rou.xml')
    time = ET.SubElement(root, 'time')

    ET.SubElement(time, 'begin', value='0')
    ET.SubElement(time, 'end', value=str(endtime))

    tree = ET.ElementTree(root)
    tree.write('map.sumo.cfg')

def generateAdditionalFile():
    root  = ET.Element("additional")
    inputt = ET.SubElement(root, 'edgeData', id="1997", file="edgeaddout.xml", freq='1')
    tree = ET.ElementTree(root)
    tree.write('additional.xml')

def sumoNetToNetworkx(netPath):
    tree = ET.parse(netPath)
    root = tree.getroot()
    sumoEdges = []
    sumoJunctions = {}

    for child in root:
        if child.tag == "edge":
            if not child.attrib["id"].__contains__(":"):
                sumoEdges.append(child)

        elif child.tag == "junction":
            if not child.attrib["id"].__contains__(":"):
                sumoJunctions[child.attrib["id"]] = child
    G = nx.MultiDiGraph()

    for edge in sumoEdges:

        try:
            highwayName = edge.attrib["type"].split(".")[1].split("_")[0]

            if highwayName not in highwayMap.keys():
                highwayName = "unclassified"

        except:
            highwayName = "unclassified"

        G.add_edge(edge.attrib["from"], edge.attrib["to"], id=edge.attrib["id"], highway=highwayName, length=float(edge[0].attrib["length"]), load=0)

        for edge in G.edges:
            G.edges[edge]["capacity"] = G.edges[edge]["length"] * theta[highwayMap[G.edges[edge]["highway"]]]

    idToEdgeMap = {}

    for edge in G.edges:
        idToEdgeMap[G.edges[edge]["id"]] = edge

    for node in G.nodes:
        G.nodes[node]['x'] = sumoJunctions[node].attrib["x"]
        G.nodes[node]['y'] = sumoJunctions[node].attrib["y"]

    return G, idToEdgeMap

def edgeDataForNetworkx(edgeDataOutPath, T0, idToEdgeMap, G):
    tree = ET.parse(edgeDataOutPath)
    root = tree.getroot()

    intervals = []
    for child in root:
        if child.tag == "interval":
            intervals.append(child)

    intervals = intervals[T0:]

    sumoSimulations = []

    t = 0
    for interval in intervals:
        Gr = G.copy()
        for edge in interval:
            try:
                edgeId = idToEdgeMap[edge.attrib["id"]]
                load = float(edge.attrib["occupancy"])/100 * Gr.edges[edgeId]["capacity"]
                Gr.edges[edgeId]["load"] = load

            except:
                pass
        t += 1
        sumoSimulations.append(Gr.copy())

    return sumoSimulations