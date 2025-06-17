import osmnx as ox
import os
import networkx as nx
import random

import scripts.SumoScripts as SS
import numpy as np
import pickle
import bisect
import concurrent.futures as cf
import tqdm
import gc
import time

import argparse
import multiprocessing

#############################################################################################################
#                                                                                                           #
#############################################################################################################
#SETUP#

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--num-workers", type=int, default=os.getenv('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
args = parser.parse_args()

simulationName = args.input
cpu_workers = args.num_workers

GENERATION          = 10
POPULATION          = 5
STARTING_POINT      = 500
SIM_STEPS           = 50
CHECKPOINT          = 20

RANDOMCHILD         = 5
maxWorkers          = cpu_workers

DELTA_T             = 1
DATA                = os.path.join("sumomaps", simulationName)
OUTPUT_PATH         = os.path.join(DATA, "ctmModel")
# Get G, C and W
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

outputFileName      = os.path.join(OUTPUT_PATH, "ctmModel.pkl")
outputFinalFileName = os.path.join(OUTPUT_PATH, "ctmModelFinal.pkl")

if os.path.exists(outputFinalFileName):
    print(f"[INFO] Output file '{outputFinalFileName}' already exists. Skipping simulation.")
    exit(0)

with open(DATA + "/G.pkl", "rb") as f:
    G = pickle.load(f)

with open(DATA + "/W.pkl", "rb") as f:
    WCollection = pickle.load(f)
WT = WCollection[STARTING_POINT::SIM_STEPS]
TIME_STEPS  = len(WT) 

#############################################################################################################
#                                                                                                           #
#############################################################################################################

highwayMap = {'motorway':0, 'highway':1, 'primary':2, 'secondary':3, 'tertiary':4, 'unclassified':5, 'residential':6, 'living_street': 7}

def generate_ctm_graph(original_graph, cell_length, WT):
    ctm_graph = nx.MultiDiGraph()  # MultiDiGraph for directed graph with parallel edges

    edgeNumber = 0
    for edge in original_graph.edges(keys=True):  # Explicitly get edge tuples (u, v, k)
        u, v, k = edge
        road_length = original_graph.edges[edge]['length']  # Access edge attributes
        capacity = original_graph.edges[edge]['capacity']

        originalLoads = []
        for loadVector in WT:
            originalLoads.append(loadVector[edgeNumber])
        originalLoads = np.array(originalLoads)

        highway = original_graph.edges[edge].get('highway', 'unknown')  # Default if highway not provided
     
        num_cells = max(1, int(road_length / cell_length))  # Ensure at least one cell
        remaining_length = road_length % cell_length
        
        # Create intermediate nodes
        intermediate_nodes = [f"{u}_{v}_cell_{i}_key_{k}" for i in range(num_cells)]
        nodes = [u] + intermediate_nodes + [v]
        
        # Add nodes to the new graph
        for node in nodes:
            ctm_graph.add_node(node)
        
        # Add edges (cells) to the new graph
        for i in range(len(nodes) - 1):
            edge_length = cell_length if i < num_cells - 1 else remaining_length
            if edge_length > 0:
                ctm_graph.add_edge(
                    nodes[i], nodes[i + 1],
                    length=edge_length,
                    capacity=capacity / num_cells,
                    loads = originalLoads / num_cells,
                    highway=highway
                )
    
        edgeNumber += 1
    return ctm_graph

class candidate:
    def __init__(self, roadTypeLength):
        
        self.roadTypeLength = roadTypeLength
        self.maxFlowPerRoad  = np.random.rand(roadTypeLength) * 100

        vfreePerRoad = []
        wfreePerRoad = []
        for r in range(roadTypeLength):
            vfreePerRoad.append(random.random() * self.maxFlowPerRoad[r])
            wfreePerRoad.append(random.random() * self.maxFlowPerRoad[r])
        
        self.vfreePerRoad = np.array(vfreePerRoad)
        self.wfreePerRoad = np.array(wfreePerRoad)

        self.capacity = []
        self.error = 0 #Fitness
        self.D = [] #Data
        self.WT = []

    def __lt__(self, other):
        return self.error < other.error

    def __gt__(self, other):
         return self.error > other.error

    def mutate(self):
        """
        Mutates the candidate's parameters with adaptive control and boundary enforcement.

        Args:
            mutationCoefficient (float): Base mutation strength.
            min_value (float): Minimum allowed value for the parameters.
            max_value (float): Maximum allowed value for the parameters.
            adaptive (bool): Whether to adapt mutation strength dynamically.
            generation (int): Current generation, required if adaptive=True.
            max_generation (int): Maximum number of generations, required if adaptive=True.
        """

        # # Mutate `junctionPreferenceCoefficient`
        # mutation_vector = np.random.uniform(-1, 1, self.roadTypeLength) * 1
        # self.junctionPreferenceCoefficient += mutation_vector
        # self.junctionPreferenceCoefficient = np.clip(self.junctionPreferenceCoefficient, 0.1, 1)  # Enforce bounds

        # # Mutate `capacityCoefficients`
        # mutation_vector = np.random.uniform(-1, 1, self.roadTypeLength) * random.random() * 10
        # self.capacityCoefficients += mutation_vector
        # self.capacityCoefficients = np.clip(self.capacityCoefficients, 0.1, 100)  # Enforce bounds

    def checkError(W, WT, C = None, MAXCOSTVALUE = 9999999):
        # W is the simulation data, WT is the real data, and C is the capacities
        # Normalize the error by the square of the capacities for each road
        
        # Check if W has zero size
        if W.size == 0:
            return MAXCOSTVALUE
        
        # Calculate the difference between the two arrays
        diff = W - WT
        # WT Yanlis, cunku WT G uzerinden bakiliyor, bizim WT'yi GG uzerinden cevirmemiz gerekiyor buraya hic girmeden.

        # If capacities are provided, normalize by the square of the capacities for each road
        if C is not None:
            # Ensure the capacity array has the right shape
            C = C.reshape(1, -1)  # Reshape to ensure it broadcasts correctly across roads
            normalization_factor = C ** 2
            error = np.sum((diff ** 2) / normalization_factor) / (W.shape[0] * W.shape[1])
        else:
            # If no capacities are provided, calculate regular MSE
            error = np.sum(diff ** 2) / (W.shape[0] * W.shape[1])

        # Check for invalid or large errors
        if error > MAXCOSTVALUE or np.isnan(error):
            return MAXCOSTVALUE

        return error

def calculateFlow(edge, edge_id, succ, sendings, receivings, flows):
    sending = sendings[edge_id]
    receiving = receivings[edge_id]
    flows[(edge, succ)] = min(sending, receiving)
    return min(sending, receiving)

def getCandidate(GG, WT, edgeToIdDictionary, edgePreds, edgeSuccs, capacities, initialLoads):
    potentialCandidate = candidate(8)

    # Initialize candidate properties
    potentialCandidate.D.append(initialLoads)
    potentialCandidate.capacity = capacities

    # Convert to NumPy arrays for faster computations
    currentLoads = np.array(initialLoads)
    capacities = np.array(capacities)
    edge_ids = {edge: edgeToIdDictionary[edge] for edge in GG.edges}

    edge_ids        =   {}
    vFrees          =   []
    wFrees          =   []
    maxFlowPerRoads =   []

    for edge in GG.edges:
        edge_ids[edge] = edgeToIdDictionary[edge]
        vFrees.append(potentialCandidate.vfreePerRoad[highwayMap[GG.edges[edge]['highway']]])
        wFrees.append(potentialCandidate.wfreePerRoad[highwayMap[GG.edges[edge]['highway']]])
        maxFlowPerRoads.append(potentialCandidate.maxFlowPerRoad[highwayMap[GG.edges[edge]['highway']]])

    vFrees          = np.array(vFrees)
    wFrees          = np.array(wFrees)
    maxFlowPerRoads = np.array(maxFlowPerRoads)

    for _ in range(TIME_STEPS - 1):
    # for _ in tqdm.tqdm(range(TIME_STEPS - 1)):
        nextLoads = currentLoads.copy()  # Prepare for next time step
        flows = {}  # Store flows for this time step

        # Precompute sending and receiving capacities
        sendings    = np.minimum(vFrees * currentLoads * DELTA_T                , maxFlowPerRoads)
        receivings  = np.minimum(wFrees * (capacities - currentLoads) * DELTA_T , maxFlowPerRoads)

        # Compute flows for all edges
        for edge in GG.edges:
            edge_id = edge_ids[edge]
            for succ in edgeSuccs[edge]:
                succ_id = edge_ids[succ]
                sending = sendings[edge_id]
                receiving = receivings[succ_id]

                # Flow is constrained by sending and receiving capacities
                flow = min(sending, receiving)
                flows[(edge, succ)] = flow

        # Update loads based on computed flows
        for edge in GG.edges:
            edge_id = edge_ids[edge]

            # Calculate inflow and outflow
            inflow = sum(flows[(pred, edge)] for pred in edgePreds[edge] if (pred, edge) in flows)
            outflow = sum(flows[(edge, succ)] for succ in edgeSuccs[edge] if (edge, succ) in flows)

            # Update load with conservation of flow
            nextLoads[edge_id] = np.clip(currentLoads[edge_id] + inflow - outflow, 0, capacities[edge_id])

        # Move to the next time step
        currentLoads = nextLoads
        potentialCandidate.D.append(currentLoads.tolist())

    # Calculate error
    error = candidate.checkError(
        np.array(potentialCandidate.D),
        np.array(WT),
        np.array(potentialCandidate.capacity)
    )

    potentialCandidate.error = error 
    return potentialCandidate

#############################################################################################################
#                                                                                                           #
#############################################################################################################

if __name__ == "__main__":
    
    startTime = time.time()
    
    ## Preperation ##
    print(">>>\tPreparing network")
    GG = generate_ctm_graph(G, 10, WT)
    lineGraph = nx.line_graph(GG)

    edgeToIdDictionary  = {}
    edgePreds           = {}
    edgeSuccs           = {}
    
    t = 0

    capacities = []
    initialLoads = []
    dividedWT = []

    for edge in GG.edges:
        
        loads = GG.edges[edge]['loads']
        if t == 0: 
            for load in loads:
                dividedWT.append([load])
        else:
            for loadStep in range(len(dividedWT)):
                dividedWT[loadStep].append(loads[loadStep])

        preds = list(lineGraph.predecessors(edge))
        succs = list(lineGraph.successors(edge))

        edgePreds[edge] = preds
        edgeSuccs[edge] = succs
        edgeToIdDictionary[edge] = t

        capacity = GG.edges[edge]['capacity']

        initialLoads.append(loads[0])
        capacities.append(capacity)

        t += 1

    population = []
    print(">>>\tInitating initial population.")
    with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        futures = [executor.submit(getCandidate, GG, dividedWT, edgeToIdDictionary, edgePreds, edgeSuccs, capacities, initialLoads) for _ in range(POPULATION)]
        # retrieve the results as they are completed with tqdm progress bar
        results = [future.result() for future in tqdm.tqdm(cf.as_completed(futures), total= POPULATION)]

    executor.shutdown(wait=True)
    
    for candida in results:
        bisect.insort_left(population, candida)

    bestFitness = []
    bestFitness.append(population[0].error)
    
    print(">>>\tInitial population generated")
    print(">>>\tStarting optimization")

    gc.collect()

    for simGen in tqdm.tqdm(range(GENERATION)):

        print("BEST ERROR:" + str(bestFitness[0]))

        population = population[:POPULATION]

        with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [executor.submit(getCandidate, GG, dividedWT, edgeToIdDictionary, edgePreds, edgeSuccs, capacities, initialLoads) for _ in range(RANDOMCHILD)]
            # retrieve the results as they are completed with tqdm progress bar
            results = [future.result() for future in tqdm.tqdm(cf.as_completed(futures), total=RANDOMCHILD)]

        for candida in results:
            bisect.insort_left(population, candida)

        executor.shutdown(wait=True)

        if (simGen % CHECKPOINT == 0):
            endTime = time.time()

            executionTime = endTime - startTime
        
            with open(outputFileName, 'wb') as f:
                pickle.dump((population, bestFitness, dividedWT, executionTime), f)

    endTime = time.time()
    executionTime = endTime - startTime
    
    with open(outputFinalFileName, 'wb') as f:
        pickle.dump((population, bestFitness, dividedWT, executionTime), f)
