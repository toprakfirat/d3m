import xml.etree.ElementTree as ET
import numpy as np 
import networkx as nx
import tqdm
import pickle
import os
import shutil
import matplotlib.pyplot as plt
import concurrent.futures as cf
import numpy as np 
import networkx as nx
import tqdm
import pickle
import os
import bisect
import shutil
import gc
import time
import re
import sys

import scripts.TrafficModelWithJunctionRespect as tmjr
import scripts.EvolutionaryOptimizationParallel as evo
import scripts.SumoScripts as SS
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Run simulation for a specific city and time section.")
parser.add_argument("PATH", type=str, help="City name")
parser.add_argument("timeSection", type=str, choices=["morning", "night"], help="Time section")
parser.add_argument("variation", type=float, default=0.2)
parser.add_argument("matrix_id", type=int, choices=range(10))
# Parse arguments
args = parser.parse_args()

# Assign variables
PATH = args.PATH
timeSection = args.timeSection

GENERATION   = 2
POPULATION   = 3 # MIN 3 DIFF CHILD OBUR TURLU LOOPTA KALIYOR

AVERAGECHILD = 2
RANDOMCHILD  = 5
DIFFCHILD    = 0
BEST_CANDIDATE_OFFSPRING = 2
BEST_AROUND_WANDER = 4

SIMTIME      = -1

MAXCOSTVALUE = 9999999
constantFlow = 1
STEPSIZE     = 1
CHECKPOINT   = 1

HOwMANYDAYS  = 5

highwayMap = {'primary':0, 'secondary':1, 'tertiary':2, 'residential':3}

roadTypeLength = len(highwayMap.keys())

all_matrices = [

    # 🔻 Bottleneck Matrix 1
    np.array([[91.2, 81.1, 36.1, 18.2],
              [20.7, 62.8, 35.5, 17.5],
              [10.2, 37.3, 52.4, 23.5],
              [ 5.4, 10.2, 20.4, 31.0]]),

    # 🔻 Bottleneck Matrix 2
    np.array([[84.5, 69.6, 45.2, 21.0],
              [20.1, 59.6, 46.0, 19.4],
              [ 8.8, 29.8, 60.6, 16.1],
              [ 5.0, 10.7, 19.7, 30.9]]),

    # 🔻 Bottleneck Matrix 3
    np.array([[96.0, 77.7, 43.5, 19.7],
              [21.2, 56.4, 37.4, 23.5],
              [ 7.9, 33.1, 39.1, 20.0],
              [ 5.2, 10.7, 19.0, 31.6]]),

    # 🔻 Bottleneck Matrix 4
    np.array([[100.0, 76.9, 44.1, 19.7],
              [15.7, 52.3, 45.1, 21.2],
              [10.2, 22.7, 54.3, 20.9],
              [ 5.4,  8.8, 20.8, 31.2]]),

    # 🔻 Bottleneck Matrix 5
    np.array([[100.0, 67.5, 37.7, 19.3],
              [19.5, 64.2, 42.6, 18.8],
              [ 9.3, 24.8, 49.2, 20.9],
              [ 5.5, 10.0, 22.3, 24.8]]),
    
        # Matrix 1
    np.array([[95.2, 70.6, 48.2, 26.1],
              [60.3, 44.8, 31.6, 15.7],
              [76.7, 56.2, 33.9, 22.3],
              [40.1, 28.9, 20.2, 13.4]]),

    # Matrix 2
    np.array([[93.1, 69.4, 52.3, 30.5],
              [78.2, 59.8, 41.1, 26.6],
              [61.4, 47.3, 33.5, 17.2],
              [39.6, 30.8, 21.5, 14.3]]),

    # Matrix 3
    np.array([[97.4, 71.5, 51.0, 27.9],
              [80.0, 60.2, 39.5, 21.3],
              [65.8, 49.4, 34.1, 18.7],
              [43.2, 31.1, 22.6, 15.2]]),

    # Matrix 4
    np.array([[92.6, 68.8, 45.2, 24.4],
              [75.9, 58.7, 37.0, 20.8],
              [63.1, 47.2, 32.9, 17.3],
              [42.5, 29.9, 20.7, 13.8]]),

    # Matrix 5
    np.array([[94.3, 70.1, 49.8, 28.6],
              [77.5, 59.9, 40.3, 24.5],
              [64.7, 48.2, 35.0, 19.1],
              [41.3, 30.6, 21.8, 14.7]]),
]

median_M = all_matrices[args.matrix_id]
variation = args.variation


def getRandomThetaWithConstraints(roadTypeLength):
    """
    Generate a random theta (lambdaVec) with descending hierarchy constraints:
    primary > secondary > tertiary > residential.
    Ensures each value is between 10 and 100.
    """
    base = np.sort(np.random.uniform(10, 100, size=roadTypeLength))[::-1]  # strictly decreasing
    noise = np.random.normal(loc=0, scale=5, size=roadTypeLength)  # small perturbations
    theta = np.clip(base + noise, 10, 100)

    # Optionally sort again to enforce ordering after noise
    theta = np.sort(theta)[::-1]

    return theta


def getRandomMatrixWithMedianMatrix(median_M, variation):
    # Apply small variation
    noise = np.random.normal(loc=0, scale=variation, size=(roadTypeLength, roadTypeLength))
    perturbed = median_M * (1 + noise)
    perturbed = np.clip(perturbed, 10, 100)

    # Enforce decreasing hierarchy: primary ≥ secondary ≥ tertiary ≥ residential
    hierarchy = [0, 1, 2, 3]  # index map: primary → secondary → tertiary → residential

    for i in range(roadTypeLength):
        # Extract the row in highway type order
        row = [perturbed[i][j] for j in hierarchy]

        # Sort it in decreasing order
        sorted_row = np.sort(row)[::-1]

        # Assign back while respecting original hierarchy positions
        for idx, j in enumerate(hierarchy):
            perturbed[i][j] = sorted_row[idx]

    return perturbed.copy()

def getCandidate(G, WTS, C, dataAvailableIndices, polyModel = None, median_M = None, variation = None):

    error = 0
    
    candidate = evo.candidate(roadTypeLength)
    candidate.D = []
    candidate.WT = []
      
    if median_M is not None:
        candidate.M = getRandomMatrixWithMedianMatrix(median_M, variation)
        candidate.theta = getRandomThetaWithConstraints(roadTypeLength)
    
    candidate.M = enforce_matrix_constraints(candidate.M)

    goodDataCounter = 0
    for i in range(len(WTS)):
        WT  = WTS[i]

        sim   = tmjr.simulator(G, candidate.theta, highwayMap, L0 = WT[0])
        sim.simulate(candidate.M, len(WT) - 1, True, polyModel)
        
        candidate.D.append(sim.W)
        candidate.WT.append(WT)

        try:
            error += evo.checkError(np.array(candidate.D)[0], np.array(WT), np.array(C), dataAvailableIndices = dataAvailableIndices)
            goodDataCounter += 1
        except:
            continue
        
    candidate.error = error / goodDataCounter
    
    print(f"Candidate error: {candidate.error}")
    return candidate

def enforce_matrix_constraints(M):
    """
    Ensures each row of the matrix is sorted in descending order 
    (i.e., Primary ≥ Secondary ≥ Tertiary ≥ Residential).
    """
    constrained = M.copy()
    for i in range(M.shape[0]):
        sorted_row = np.sort(constrained[i])[::-1]
        constrained[i] = sorted_row
    return np.clip(constrained, 10, 100)

def getOffspringCandidate(candidate, G, WTS, C, dataAvailableIndices, polyModel = None):

    error = 0
    
    candidate.D = []
    candidate.WT = []

    goodDataCounter = 0
    for i in range(len(WTS)):

        WT  = WTS[i]

        sim   = tmjr.simulator(G, candidate.theta, highwayMap, L0 = WT[0])
        sim.simulate(candidate.M, len(WT) - 1, True, polyModel)
        
        candidate.D.append(sim.W)
        candidate.WT.append(WT)

        try:
            error += evo.checkError(np.array(candidate.D)[0], np.array(WT), np.array(C), dataAvailableIndices = dataAvailableIndices)
            goodDataCounter += 1
        except:
            continue
        
    candidate.error = error / goodDataCounter
    
    print(f"Candidate error: {candidate.error}")
    return candidate

def find_latest_checkpoint(folder, base_name):
    checkpoint_files = [
        f for f in os.listdir(folder) if re.match(rf"{base_name}_Gen-\d+\.pkl", f)
    ]
    if not checkpoint_files:
        return None, 0  # No checkpoint found

    # Extract generation numbers from filenames
    checkpoint_files.sort(key=lambda x: int(re.search(r"Gen-(\d+)", x).group(1)))
    latest_checkpoint = checkpoint_files[-1]
    latest_generation = int(re.search(r"Gen-(\d+)", latest_checkpoint).group(1))
    return os.path.join(folder, latest_checkpoint), latest_generation

def load_latest_checkpoint(folder, base_name):
    checkpoint_files = [
        f for f in os.listdir(folder) if re.match(rf"{base_name}_Gen-(\d+)\.pkl", f)
    ]
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        return None, None, 0  # No checkpoint found, return default values

    # Extract generation numbers and sort files by generation
    checkpoint_files.sort(key=lambda f: int(re.search(r"Gen-(\d+)", f).group(1)), reverse=True)

    latest_checkpoint = os.path.join(folder, checkpoint_files[0])
    
    with open(latest_checkpoint, 'rb') as f:
        population, bestFitness = pickle.load(f)
    
    latest_generation = int(re.search(r"Gen-(\d+)", checkpoint_files[0]).group(1))
    
    print(f"Loaded checkpoint from: {latest_checkpoint}")
    return population, bestFitness, latest_generation

def save_generation_checkpoint(folder, base_name, generation, population, bestFitness):
    checkpoint_file = os.path.join(folder, f"{base_name}_Gen-{generation}.pkl")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump((population, bestFitness), f)
    print(f"Checkpoint saved: {checkpoint_file}")

def optimize(mutations = False, population = [], generation = 0, timeSection = "morning", dataAvailableIndices = None, WTS = None, bestFitness = [], polyModel = None):

    maxWorkers = 5

    thePath = os.path.join(PATH, "ourmodel_real_interpolated" + timeSection + str("_") + str(variation) + str("_") + str(args.matrix_id))
    
    if len(population) < POPULATION:
        print("Generating initial population")
        with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [executor.submit(getCandidate, G, WTS, C, dataAvailableIndices, polyModel) for _ in range(POPULATION - len(population))]
            # retrieve the results as they are completed with tqdm progress bar
            results = [future.result() for future in tqdm.tqdm(cf.as_completed(futures), total=POPULATION)]

            for candidate in results:
                bisect.insort_left(population, candidate)

    population.sort(key=lambda x: x.error, reverse=False)
    
    print("Initial population generated")
    print("Starting optimization")

    population = population[:POPULATION]
    gc.collect()
    
    for t in tqdm.tqdm(range(generation + 1, GENERATION)):
        # RUN SIMULATION UNTIL ONE-STEP IS GOOD ENOUGH

        if not os.path.exists(thePath):
                os.makedirs(thePath)

        if t%CHECKPOINT == 0:
            save_generation_checkpoint(thePath, PATH, t, population, bestFitness)

        totalFitness = 0
        for p in population:
            totalFitness += p.error

        parentDeck = []
        for p in population:
            parentDeck.extend([p] * int(1 + (1 - (p.error / totalFitness)) * 50))
        deckLength = len(parentDeck)
        
        #############################################################################################################
        # GENERATE OFFSPRINGS
        #############################################################################################################
        offSprings = []

        print("DIFFCHILD")
        for i in tqdm.tqdm(range(DIFFCHILD)):
            # Select three distinct parents randomly from the parent deck
            p1 = parentDeck[np.random.randint(0, deckLength)]
            p2 = parentDeck[np.random.randint(0, deckLength)]
            p3 = parentDeck[np.random.randint(0, deckLength)]

            # Ensure p1, p2, and p3 are distinct
            while p1 == p2 or p2 == p3 or p1 == p3:
                p1 = parentDeck[np.random.randint(0, deckLength)]
                p2 = parentDeck[np.random.randint(0, deckLength)]
                p3 = parentDeck[np.random.randint(0, deckLength)]

            # Adaptive F and crossover rate calculation
            # F = np.random.uniform(0.4, 0.9) * (1 - t / GENERATION)
            # F = max(F, 0.1)  # Prevent F from being too small

            # crossover_rate = np.random.uniform(0.7, 1.0) * (t / GENERATION)
            # crossover_rate = min(crossover_rate, 0.9)  # Prevent excessive crossover

            F = np.random.uniform(0.5, 0.9) * (1 - t / GENERATION)
            crossover_rate = np.random.uniform(0.8, 1.0) * (t / GENERATION)

            # Create offspring using DE mutation
            child = evo.candidate(roadTypeLength)
            child.M = p1.M + F * (p2.M - p3.M)
            child.M = np.clip(child.M, 0, 100)  # Ensure values remain within [0, 100]
            
            # Apply crossover
            crossover_mask = np.random.rand(*child.M.shape) < crossover_rate
            child.M = np.where(crossover_mask, child.M, p1.M)

            # Apply the same process to theta values
            child.theta = p1.theta + F * (p2.theta - p3.theta)
            child.theta = np.clip(child.theta, 0, 100)
            crossover_mask_theta = np.random.rand(len(child.theta)) < crossover_rate
            child.theta = np.where(crossover_mask_theta, child.theta, p1.theta)

            offSprings.append(child)

        print("AVG CHILD")
        for i in tqdm.tqdm(range(AVERAGECHILD)):
            p1 = parentDeck[np.random.randint(0, deckLength)]
            p2 = parentDeck[np.random.randint(0, deckLength)]
            child = evo.candidate(roadTypeLength)
            child.setOffspring(p1.M, p2.M, p1.theta, p2.theta)

            if mutations:
                child.mutate(
                    mutationCoefficient=5, 
                    min_value=10, 
                    max_value=100, 
                    adaptive=True, 
                    generation=t, 
                    max_generation=GENERATION
                )
            
            
            
            offSprings.append(child)

        print("RANDOMCHILD")
        for i in tqdm.tqdm(range(RANDOMCHILD)):
            child = evo.candidate(roadTypeLength)

            if median_M is not None and variation is not None:
                child.M = getRandomMatrixWithMedianMatrix(median_M, variation)
                child.theta = getRandomThetaWithConstraints(roadTypeLength)

            offSprings.append(child)

        # Retrieve the current best candidate
        best_candidate = population[0]  # Assuming the population is sorted in ascending order by error

        print("Getting best candidate.")
        print("Best Candidate M values:")
        print(best_candidate.M)
        print("Best Candidate theta values:")
        print(best_candidate.theta)
        #############################################################################################################

        # Generate offspring around the best candidate with values bounded between 0 and 100
        for _ in range(BEST_CANDIDATE_OFFSPRING):
            child = evo.candidate(roadTypeLength)
            
            # Choose a perturbation factor F_local, as before
            F_local = np.random.uniform(0.05, 0.3) if np.random.rand() < 0.8 else np.random.uniform(0.3, 0.7)
            
            # Apply perturbation and ensure values stay within [0, 100]
            perturbation = F_local * (np.random.rand(*best_candidate.M.shape) - 0.5)
            child.M = best_candidate.M + perturbation
            child.M = np.clip(child.M, 10, 100)  # Ensure all values are within [0, 100]

            # Repeat for theta if necessary
            perturbation_theta = F_local * (np.random.rand(len(best_candidate.theta)) - 0.5)
            child.theta = best_candidate.theta + perturbation_theta
            child.theta = np.clip(child.theta, 0, 100)  # Ensure values are within [0, 100] for theta if needed

            offSprings.append(child)

        #############################################################################################################
        #############################################################################################################
        
        for candidate in offSprings:
            candidate.M = enforce_matrix_constraints(candidate.M)

        print("Getting offsprings.")
        with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [executor.submit(getOffspringCandidate, cand, G, WTS, C, dataAvailableIndices, polyModel) for cand in offSprings]

            # retrieve the results as they are completed with tqdm progress bar
            results = [future.result() for future in tqdm.tqdm(cf.as_completed(futures), total=len(offSprings))]
            
            for candidate in results:
                bisect.insort_left(population, candidate)

        bestFitness.append(population[0].error)
        print("Generation: " + str(t) + " Best fitness: " + str(population[0].error))

        population = population[:POPULATION]
        gc.collect()

    return population[0], bestFitness

if __name__ == "__main__":
    
    with open(os.path.join(PATH, "G.pkl"), "rb") as f:
        G = pickle.load(f)

    with open(os.path.join(PATH, "C.pkl"), "rb") as f:
        C = pickle.load(f)
    
    with open(os.path.join(PATH, "interpolated_WTs_morning.pkl"), "rb") as f:
        WTS = pickle.load(f)

    WTS = WTS[:HOwMANYDAYS]

    with open(os.path.join(PATH, "dataAvailableIndices.pkl"), "rb") as f:
        dataAvailableIndices = pickle.load(f)

    # Load the coefficients array from the .npy file
    coeffs = np.load(PATH + "/jam_model_poly.npy")
    poly_model = np.poly1d(coeffs)

    population  = []
    bestFitness = []

    thePath = os.path.join(PATH, "ourmodel_real_interpolated" + timeSection + str("_") + str(variation) + str("_") + str(args.matrix_id))

    latest_generation = 0
    try:
        population, bestFitness, latest_generation = load_latest_checkpoint(thePath, PATH)
    except:
        pass

    if latest_generation > 0:
        print(f"Resuming from generation {latest_generation}")
    else:
        print("Starting fresh")

    if type(population) == type(None):
        population  = []
        bestFitness = [] 

    if type(population) != list:
        population = [population]

    if len(population) < POPULATION:
        maxWorkers = 5
        
        print("Generating initial population")
        with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [executor.submit(getCandidate, G, WTS, C, dataAvailableIndices, poly_model) for _ in range(POPULATION - len(population))]
            results = [future.result() for future in tqdm.tqdm(cf.as_completed(futures), total=POPULATION)]

            for candidate in results:
                bisect.insort_left(population, candidate)   

    population, bestFictness = optimize(WTS = WTS, mutations=True, population=population, generation=latest_generation, timeSection=timeSection, dataAvailableIndices = dataAvailableIndices, bestFitness = bestFitness, polyModel=poly_model)

    save_generation_checkpoint(thePath, PATH, "f", population, bestFitness)
        