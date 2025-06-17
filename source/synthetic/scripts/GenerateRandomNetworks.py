import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
import os
import shutil;

def generate_and_visualize_network(algorithm='erdos_renyi', num_nodes=10, param=0.5, filePath='network'):
    # Generate the graph using the specified algorithm
    if algorithm == 'erdos_renyi':
        G = nx.erdos_renyi_graph(num_nodes, param)
    elif algorithm == 'barabasi_albert':
        G = nx.barabasi_albert_graph(num_nodes, int(param))
    elif algorithm == 'watts_strogatz':
        G = nx.watts_strogatz_graph(num_nodes, int(param), 0.1)
    else:
        raise ValueError("Unsupported algorithm. Use 'erdos_renyi', 'barabasi_albert', or 'watts_strogatz'.")
    
    # Ensure the graph is directed and has no dead ends
    G = G.to_directed()
    for node in list(G.nodes):
        if G.out_degree(node) == 0:
            neighbors = list(G.nodes)
            neighbors.remove(node)
            target = random.choice(neighbors)
            G.add_edge(node, target)
        if G.in_degree(node) == 0:
            neighbors = list(G.nodes)
            neighbors.remove(node)
            source = random.choice(neighbors)
            G.add_edge(source, node)
    
    networkx_to_sumo(G, sumo_file=filePath + ".net.xml", node_file=filePath + ".nod.xml", edge_file=filePath + ".edg.xml")
    
def networkx_to_sumo(G, sumo_file="myNetwork.net.xml", node_file='nodes.nod.xml', edge_file='edges.edg.xml'):
    scaling_factor = 700

    # Assign random positions to nodes for SUMO visualization
    pos = {node: (random.random() * scaling_factor, random.random() * scaling_factor) for node in G.nodes}

    with open(node_file, 'w') as nodes:
        nodes.write('<nodes>\n')
        for node, (x, y) in pos.items():
            nodes.write(f'    <node id="{node}" x="{x}" y="{y}" type="priority"/>\n')
        nodes.write('</nodes>\n')

    with open(edge_file, 'w') as edges:
        edges.write('<edges>\n')
        for edge in G.edges():
            edges.write(f'    <edge id="{edge[0]}_{edge[1]}" from="{edge[0]}" to="{edge[1]}" priority="1"/>\n')
        edges.write('</edges>\n')

    os.system(f'netconvert --node-files={node_file} --edge-files={edge_file} --output-file={sumo_file}')

    os.remove(node_file)
    os.remove(edge_file)


if (not os.path.exists("../_networks/inputNetworks/")):
    os.mkdir()

nodeCounts = [7,8,10,15]

algoName = 'erdos_renyi'
paramList = [0.3, 0.5, 0.8]

for nodeCount in nodeCounts:
    for paramVal in paramList:
        generate_and_visualize_network(algorithm=algoName, num_nodes=nodeCount, param=paramVal, filePath= f"../_networks/inputNetworks/{algoName}_{nodeCount}_{paramVal}")

algoName = 'barabasi_albert'
paramList = [1, 2, 3]
for nodeCount in nodeCounts:
    for paramVal in paramList:
        generate_and_visualize_network(algorithm=algoName, num_nodes=nodeCount, param=paramVal, filePath= f"../_networks/inputNetworks/{algoName}_{nodeCount}_{paramVal}")

algoName = 'watts_strogatz'
paramList = [2, 4, 7]
for nodeCount in nodeCounts:
    for paramVal in paramList:
        generate_and_visualize_network(algorithm=algoName, num_nodes=nodeCount, param=paramVal, filePath= f"../_networks/inputNetworks/{algoName}_{nodeCount}_{paramVal}")