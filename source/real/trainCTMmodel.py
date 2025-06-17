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

#############################################################################################################
#                                                                                                           #
#############################################################################################################
#SETUP#

# Argument parser setup
parser = argparse.ArgumentParser(description="Run simulation for a specific city and time section.")
parser.add_argument("PATH", type=str, help="City name")
parser.add_argument("timeSection", type=str, choices=["morning", "night"], help="Time section")

# Parse arguments
args = parser.parse_args()

# Assign variables
PATH = args.PATH
timeSection = args.timeSection

CTM_MODEL_PATH = os.path.join(PATH, "ctm_real_model_withStepWait")
if not os.path.exists(CTM_MODEL_PATH):
    os.makedirs(CTM_MODEL_PATH)

CTM_DATA_PATH = os.path.join(PATH, "ctmData")
if not os.path.exists(CTM_DATA_PATH):
    os.makedirs(CTM_DATA_PATH)

GENERATION          = 2
POPULATION          = 1
STARTING_POINT      = 0
SIM_STEPS           = 1
CHECKPOINT          = 5

RANDOMCHILD         = 2
maxWorkers          = 5

DELTA_T             = 1
STEP_WAIT           = 5

morning_files_collection = [] 

HOWMANYDAYS = 2

#############################################################################################################
#                                                                                                           #
#############################################################################################################

highwayMap = {'primary':0, 'secondary':1, 'tertiary':2, 'residential':3}

def generate_ctm_graph(original_graph, cell_length, WT, dataAvailableIndices):
    ctm_graph = nx.MultiDiGraph()
    newDataAvIndices = []
    newCapacities = []
    ctm_to_original = {}

    ctmEdgeNumer = 0
    edgeNumber = 0

    edgeToIndex = {}
    index = 0
    for edge in original_graph.edges(keys=True):
        
        edgeToIndex[edge] = index

        u, v, k = edge
        
        road_length = original_graph.edges[edge]['length']
        capacity = original_graph.edges[edge]['capacity']
        highway = original_graph.edges[edge].get('highway', 'unknown')

        originalLoads = []
        for loadVector in WT:
            originalLoads.append(loadVector[edgeNumber])
        originalLoads = np.array(originalLoads)
     
        num_cells = max(1, int(road_length / cell_length))
        remaining_length = road_length % cell_length

        intermediate_nodes = [f"{u}_{v}_cell_{i}_key_{k}" for i in range(num_cells - 1)]
        nodes = [u] + intermediate_nodes + [v]

        for node in nodes:
            ctm_graph.add_node(node)

        for i in range(len(nodes) - 1):
            edge_length = cell_length if i < num_cells - 1 else remaining_length
            if edge_length > 0:
                if edgeNumber in dataAvailableIndices:
                    newDataAvIndices.append(ctmEdgeNumer)

                ctm_graph.add_edge(
                    nodes[i], nodes[i + 1],
                    length=edge_length,
                    capacity=capacity / num_cells,
                    loads=originalLoads / num_cells,
                    highway=highway
                )

                newCapacities.append(capacity / num_cells)
                ctm_to_original[(nodes[i], nodes[i + 1])] = (u, v, k)
                ctmEdgeNumer += 1

        edgeNumber += 1

        index += 1
    return ctm_graph, newDataAvIndices, newCapacities, ctm_to_original, edgeToIndex

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

    def checkError(W, WT, C = None, MAXCOSTVALUE = 9999999, dataAvailableIndices = None):
        # W is the simulation data, WT is the real data, and C is the capacities
        # Normalize the error by the square of the capacities for each road
        
        if dataAvailableIndices is not None:
    
            filtered_W = W[:, dataAvailableIndices]  # Simulation data for the selected edges
            filtered_WT = WT[:, dataAvailableIndices]  # Real-world data for the selected edges

            if C is not None:
                    filtered_C = C[dataAvailableIndices]  # Extract capacities for selected edges
                    normalization_factor = filtered_C ** 2
                    error = np.sum((filtered_W - filtered_WT) ** 2 / normalization_factor) / filtered_W.size
            else:
                error = np.sum((filtered_W - filtered_WT) ** 2) / filtered_W.size
        
        else:
            # Check if W has zero size
            if W.size == 0:
                return MAXCOSTVALUE
            
            # Calculate the difference between the two arrays
            diff = W - WT

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
        print(error)
        if error > MAXCOSTVALUE or np.isnan(error):
            return MAXCOSTVALUE

        return error
    
    def checkErrorCTM(W, WT, C=None, MAXCOSTVALUE=9999999, dataAvailableIndices = None):
    # W is the simulation data, WT is the real data, and C is the capacities
    # Normalize the error by the square of the capacities for each road
    
        if dataAvailableIndices is not None:
            if C is not None:
                error = np.sum((W[:, dataAvailableIndices] - WT[:, dataAvailableIndices]) ** 2 / (np.maximum(C[dataAvailableIndices], W[:, dataAvailableIndices] , WT[:, dataAvailableIndices] ) ** 2)) / W[:, dataAvailableIndices].size
            else:
                error = np.sum((W[:, dataAvailableIndices]  - WT[:, dataAvailableIndices]) ** 2) / W[:, dataAvailableIndices].size
        
        else:
            # Check if W has zero size
            if W.size == 0:
                return MAXCOSTVALUE
            
            # Calculate the difference between the two arrays
            diff = W - WT

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
        print(error)
        if error > MAXCOSTVALUE or np.isnan(error):
            return MAXCOSTVALUE

        return error

def calculateFlow(edge, edge_id, succ, sendings, receivings, flows):
    sending = sendings[edge_id]
    receiving = receivings[edge_id]
    flows[(edge, succ)] = min(sending, receiving)
    return min(sending, receiving)

def adjust_load_to_match_jam(L_hat, C, J_target):
    L_hat = np.array(L_hat)
    C = np.array(C)
    n = len(C)
    J_hat = np.sum(L_hat / C) / n

    if np.isclose(J_target, J_hat, atol=1e-5):
        return L_hat.copy()
    elif J_target > J_hat:
        alpha = (J_target - J_hat) / (1 - J_hat)
        alpha = min(alpha, 1.0)
        return L_hat + alpha * (C - L_hat)
    else:
        beta = 1 - (J_target / J_hat)
        beta = min(beta, 1.0)
        return L_hat - beta * L_hat


def getCandidate(GG, divWTsForDifTimes, edgeToIdDictionary, edgePreds, edgeSuccs, capacities, initLoadsForDifTimes, ctm_to_original, WTsForDifTimes, Coriginal, edgeToIndex, dataAvailableIndices, GedgeLength, jam_model=None, start_hour=9, step_minutes=5):
    """
    Computes the CTM while ensuring strict mass conservation and keeping error and candidate generation unchanged.
    """
    print('AAAAA')
    potentialCandidate = candidate(8)
    error = 0
    okDataFileCounter = 0
    
    print(">>>\tGenerating candidate")
    for i in tqdm.tqdm(range(len(divWTsForDifTimes))):
    
        WT = divWTsForDifTimes[i]
        initialLoads = initLoadsForDifTimes[i]
        candidateD = [initialLoads]
        potentialCandidate.capacity = capacities

        currentLoads = np.array(initialLoads, dtype=float)
        capacities = np.array(capacities, dtype=float)
        edge_ids = {edge: edgeToIdDictionary[edge] for edge in GG.edges}
        
        maxFlowPerRoads = np.array([potentialCandidate.maxFlowPerRoad[highwayMap[GG.edges[edge]['highway']]] for edge in GG.edges])
        
        TIME_STEPS = len(WT)
        for step in tqdm.tqdm(range(TIME_STEPS - 1)):
            for s in range(STEP_WAIT):
                nextLoads = currentLoads.copy()
                flows = {}

                # === STEP 2: JAM FACTOR ADJUSTMENT ===
                if jam_model:
                    # Compute current simulation time in hours (with fractional steps)
                    sim_minutes = step * step_minutes
                    hour = start_hour + sim_minutes / 60.0
                    J_target = jam_model(hour % 24)  # wrap into [0, 24)
                    currentLoads = adjust_load_to_match_jam(currentLoads, capacities, J_target)

                sendings = np.minimum(currentLoads * DELTA_T, maxFlowPerRoads).copy()
                receivings = np.minimum((capacities - currentLoads) * DELTA_T, maxFlowPerRoads).copy()

                inflow = np.zeros_like(currentLoads)
                outflow = np.zeros_like(currentLoads)

                # 1. Compute actual flows with budget tracking
                for edge in GG.edges:
                    edge_id = edge_ids[edge]
                    for succ in edgeSuccs[edge]:
                        succ_id = edge_ids[succ]

                        flow = min(sendings[edge_id], receivings[succ_id], currentLoads[edge_id])
                        flow = round(flow, 6)  # Optional: limit precision

                        sendings[edge_id] -= flow
                        receivings[succ_id] -= flow

                        flows[(edge, succ)] = flow
                        outflow[edge_id] += flow
                        inflow[succ_id] += flow

                # 2. Apply inflow and outflow
                nextLoads = currentLoads - outflow + inflow
                nextLoads = np.clip(nextLoads, 0, capacities)

                # 3. Mass check
                total_before = np.sum(currentLoads)
                total_after = np.sum(nextLoads)
                if not np.isclose(total_before, total_after, atol=1e-6):
                    print(f"⚠️ Mass imbalance at step {step}: Δ={total_after - total_before:.6f}")

                currentLoads = nextLoads

            candidateD.append(currentLoads.tolist())

        try:
            error += compute_original_edge_errors(GG, WTsForDifTimes[i], np.array(candidateD), edgeToIndex, edgeToIdDictionary, Coriginal, ctm_to_original, GedgeLength, dataAvailableIndices=dataAvailableIndices)
            # error += candidate.checkErrorCTM(
            #     np.array(candidateD),
            #     np.array(WT),
            #     np.array(potentialCandidate.capacity),
            #     dataAvailableIndices=dataAvailableIndices
            # )
            print(error)
            okDataFileCounter += 1
            potentialCandidate.D.append(candidateD)
        except Exception as e:
            # print(f"Error in checkError: {e}")
            print(e)
    
    potentialCandidate.error = error / max(okDataFileCounter, 1)
    #potentialCandidate.error = error
    return potentialCandidate

import numpy as np

def compute_original_edge_errors(GG, WT, What, edgeToIndexOriginal, edgeToIdDictionary, Coriginal, ctm_to_original, GedgeLength, dataAvailableIndices= None, MAXCOSTVALUE=9999999,):

    Whatoriginal = np.zeros((WT.shape[0], WT.shape[1]))

    for t in range(WT.shape[0]):

        loadVector = np.zeros((GedgeLength, 1))

        for edge in GG.edges:
            loadVector[edgeToIndexOriginal[ctm_to_original[(edge[0], edge[1])]]] += What[t][edgeToIdDictionary[edge]]
        
        Whatoriginal[t] = loadVector.reshape(-1)

    if dataAvailableIndices is not None:
        WhatoriginalWithAvailable = Whatoriginal[:, dataAvailableIndices]
        WTWithAvaiable = WT[:, dataAvailableIndices]
        if Coriginal is not None:
            C = np.array(Coriginal)[dataAvailableIndices]
            normalization = np.maximum(C, WhatoriginalWithAvailable, WTWithAvaiable) ** 2
            error = np.sum((WhatoriginalWithAvailable - WTWithAvaiable) ** 2 / normalization) / WhatoriginalWithAvailable.size
        else:
            error = np.sum((WhatoriginalWithAvailable - WTWithAvaiable) ** 2) / WhatoriginalWithAvailable.size
    else:
        if Whatoriginal.size == 0:
            return MAXCOSTVALUE
        if Coriginal is not None:
            C = np.array(Coriginal).reshape(1, -1)
            normalization = C ** 2
            error = np.sum((Whatoriginal - WT) ** 2 / normalization) / (Whatoriginal.shape[0] * Whatoriginal.shape[1])
        else:
            error = np.sum((Whatoriginal - WT) ** 2) / (Whatoriginal.shape[0] * Whatoriginal.shape[1])

    if error > MAXCOSTVALUE or np.isnan(error):
        return MAXCOSTVALUE
    return error



#############################################################################################################
#                                                                                                           #
#############################################################################################################

if __name__ == "__main__":

    startTime = time.time()

    with open(os.path.join(PATH, "G.pkl"), "rb") as f:
        G = pickle.load(f)

    with open(os.path.join(PATH, "C.pkl"), "rb") as f:
        C = pickle.load(f)
    
    with open(os.path.join(PATH, "interpolated_WTs_morning.pkl"), "rb") as f:
        WTS = pickle.load(f)

    WTsForDifTimes = WTS[:HOWMANYDAYS]

    with open(os.path.join(PATH, "dataAvailableIndices.pkl"), "rb") as f:
        dataAvailableIndices = pickle.load(f)

    # Load the coefficients array from the .npy file
    coeffs = np.load(PATH + "/jam_model_poly.npy")
    jam_model = np.poly1d(coeffs)

    divWTsForDifTimes      = []
    initLoadsForDifTimes   = []
    
    for day in range(HOWMANYDAYS):

        # G,C,L,W, dataAvailableIndices = file
        
        WT = WTsForDifTimes[day]

        ## Preperation ##
        print(">>>\tPreparing network")
        GG, newDataAvIndices, newCapacities, ctm_to_original, edgeToIndex = generate_ctm_graph(G, 10, WT, dataAvailableIndices)

        with open(os.path.join(CTM_DATA_PATH,"dataAvailableIndicesCTM"), 'wb') as f:
            pickle.dump(newDataAvIndices, f)

        with open(os.path.join(CTM_DATA_PATH,"capacitiesCTM"), 'wb') as f:
            pickle.dump(newCapacities, f)

        lineGraph = nx.line_graph(GG)

        edgeToIdDictionary  = {}
        edgePreds           = {}
        edgeSuccs           = {}
        
        t = 0

        capacities = []
        initialLoads = []
        dividedWT = []

        for edge in tqdm.tqdm(GG.edges):
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

        divWTsForDifTimes.append(dividedWT)
        initLoadsForDifTimes.append(initialLoads)

    GedgeLength = len(G.edges) 
    population = []
    print(">>>\tInitating initial population.")
    with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        futures = [executor.submit(getCandidate, GG, divWTsForDifTimes, edgeToIdDictionary, edgePreds, edgeSuccs, capacities, initLoadsForDifTimes, ctm_to_original, WTsForDifTimes, C, edgeToIndex, dataAvailableIndices, GedgeLength, jam_model=jam_model, start_hour=9, step_minutes=5) for _ in range(POPULATION)]

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

    with open(os.path.join(CTM_MODEL_PATH,"GG.pkl"), 'wb') as f:
        pickle.dump((GG, capacities, population[0].WT), f)

    with open(os.path.join(CTM_DATA_PATH,"divWTsForDifTimes.pkl"), 'wb') as f:
            pickle.dump(divWTsForDifTimes, f)

    for simGen in tqdm.tqdm(range(GENERATION)):

        print("BEST ERROR:" + str(bestFitness[0]))

        population = population[:RANDOMCHILD]

        with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [executor.submit(getCandidate, GG, divWTsForDifTimes, edgeToIdDictionary, edgePreds, edgeSuccs, capacities, initLoadsForDifTimes, ctm_to_original, WTsForDifTimes, C, edgeToIndex, dataAvailableIndices, GedgeLength, jam_model=jam_model, start_hour=9, step_minutes=5) for _ in range(RANDOMCHILD)]
            # retrieve the results as they are completed with tqdm progress bar
            results = [future.result() for future in tqdm.tqdm(cf.as_completed(futures), total=RANDOMCHILD)]

        for candida in results:
            bisect.insort_left(population, candida)

        executor.shutdown(wait=True)

        if (simGen % CHECKPOINT == 0):
            endTime = time.time()
        executionTime = endTime - startTime

        bestFitness.append(population[0].error)
    
        if (simGen % CHECKPOINT == 0):
            with open(os.path.join(CTM_MODEL_PATH,"ctmModel" + timeSection + "_" + str(simGen) + ".pkl"), 'wb') as f:
                pickle.dump((population[0], bestFitness, dividedWT, executionTime), f)

    endTime = time.time()
    executionTime = endTime - startTime
    
    with open(os.path.join(CTM_MODEL_PATH,"ctmModel" + timeSection + "_" + str(simGen) + ".pkl"), 'wb') as f:
        pickle.dump((population[0], bestFitness, dividedWT, executionTime), f)