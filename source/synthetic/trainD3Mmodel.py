import xml.etree.ElementTree as ET
import numpy as np 
import networkx as nx
import tqdm
import pickle
import os
import shutil
import scripts.SumoScripts as SS
import matplotlib.pyplot as plt
import scripts.TrafficModelWithJunctionRespect as tmjr
import scripts.EvolutionaryOptimizationParallel as evo
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

GENERATION   = 5
POPULATION   = 5

AVERAGECHILD = 1
RANDOMCHILD  = 1
DIFFCHILD    = 1
BEST_CANDIDATE_OFFSPRING = 1
BEST_AROUND_WANDER = 1

SIMTIME      = -1

MAXCOSTVALUE = 9999999
constantFlow = 1
STEPSIZE     = 10
CHECKPOINT   = 20
SIMSTART     = 500

highwayMap = {'motorway':0, 'highway':1, 'primary':2, 'secondary':3, 'tertiary':4, 'unclassified':5, 'residential':6, 'living_street': 7}

M   =       [[100, 50, 20, 17, 10, 8, 8, 5],
            [55, 55, 22, 20, 15, 10, 10, 5],
            [25, 25, 25, 23, 20, 15, 12, 5],
            [25, 25, 25, 25, 23, 20, 13, 5],
            [22, 22, 22, 22, 22, 21, 15, 5],
            [21, 21, 21, 21, 21, 21, 17, 6],
            [21, 21, 21, 21, 21, 21, 17, 6],
            [18, 18, 18, 18, 18, 18, 18, 18]]

theta       =   [100, 50, 20, 17, 10, 8, 8, 5]

M     = np.array(M)
theta = np.array(theta)

roadTypeLength = len(theta)

def getCandidate(G, WT, C):

    candidate = evo.candidate(roadTypeLength)
    sim   = tmjr.simulator(G, candidate.theta, highwayMap)
    sim.simulate(candidate.M, len(WT) - 1, True)
    candidate.D = sim.W
    error = evo.checkError(np.array(candidate.D), np.array(WT), np.array(C))
    candidate.error = error
    
    return candidate

def getOffspringCandidate(candidate, G, WT, C):
    
    sim   = tmjr.simulator(G, candidate.theta, highwayMap)
    sim.simulate(candidate.M, len(WT) - 1, True)

    candidate.D = sim.W
    error = evo.checkError(np.array(candidate.D), np.array(WT), np.array(C))
    candidate.error = error

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

def load_checkpoint_from_file(file_path):
    with open(file_path, 'rb') as f:
        population, bestFitness = pickle.load(f)
    print(f"Checkpoint loaded: {file_path}")
    return population, bestFitness

def save_generation_checkpoint(folder, base_name, generation, population, bestFitness):
    checkpoint_file = os.path.join(folder, f"{base_name}_Gen-{generation}.pkl")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump((population, bestFitness), f)
    print(f"Checkpoint saved: {checkpoint_file}")


def optimize(G,C,W, mutations = False, population = [], generation = GENERATION):

    maxWorkers = cpu_workers
    # SETUP INITIAL VALUES
    WT = W[:SIMTIME]
    L = WT[0]
    
    t = 0
    for edge in G.edges:
        G.edges[edge]['load'] = L[t]
        t+=1
    
    if len(population) < POPULATION:
        print("Generating initial population")
        with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [executor.submit(getCandidate, G, WT, C) for _ in range(POPULATION - len(population))]
            # retrieve the results as they are completed with tqdm progress bar
            results = [future.result() for future in tqdm.tqdm(cf.as_completed(futures), total=POPULATION)]

            for candidate in results:
                bisect.insort_left(population, candidate)

    population.sort(key=lambda x: x.error, reverse=False)
    bestFitness = []
    print("Initial population generated")
    print("Starting optimization")

    population = population[:POPULATION]
    gc.collect()
    
    for t in tqdm.tqdm(range(GENERATION - generation)):
        # RUN SIMULATION UNTIL ONE-STEP IS GOOD ENOUGH

        t += generation
        
        if t%CHECKPOINT == 0:
            save_generation_checkpoint(mainFolder, baseName, t, population, bestFitness)

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

        for i in range(DIFFCHILD):
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

        for i in range(AVERAGECHILD):
            p1 = parentDeck[np.random.randint(0, deckLength)]
            p2 = parentDeck[np.random.randint(0, deckLength)]
            child = evo.candidate(roadTypeLength)
            child.setOffspring(p1.M, p2.M, p1.theta, p2.theta)
            offSprings.append(child)

            if mutations:
                child.mutate(
                    mutationCoefficient=5, 
                    min_value=10, 
                    max_value=100, 
                    adaptive=True, 
                    generation=t, 
                    max_generation=GENERATION
                )

        for i in range(RANDOMCHILD):
            child = evo.candidate(roadTypeLength)
            offSprings.append(child)

        # Retrieve the current best candidate
        best_candidate = population[0]  # Assuming the population is sorted in ascending order by error

        # Generate offspring around the best candidate with values bounded between 0 and 100
        for _ in range(BEST_CANDIDATE_OFFSPRING):
            child = evo.candidate(roadTypeLength)
            
            # Choose a perturbation factor F_local, as before
            F_local = np.random.uniform(0.05, 0.3) if np.random.rand() < 0.8 else np.random.uniform(0.3, 0.7)
            
            # Apply perturbation and ensure values stay within [0, 100]
            perturbation = F_local * (np.random.rand(*best_candidate.M.shape) - 0.5)
            child.M = best_candidate.M + perturbation
            child.M = np.clip(child.M, 0, 100)  # Ensure all values are within [0, 100]

            # Repeat for theta if necessary
            perturbation_theta = F_local * (np.random.rand(len(best_candidate.theta)) - 0.5)
            child.theta = best_candidate.theta + perturbation_theta
            child.theta = np.clip(child.theta, 0, 100)  # Ensure values are within [0, 100] for theta if needed

            offSprings.append(child)

        #############################################################################################################
        #############################################################################################################

        with cf.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [executor.submit(getOffspringCandidate, cand, G, WT, C) for cand in offSprings]

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
    DATA_SUMOAVGS = os.path.join("sumomaps", simulationName)

    networkCollection = []

    with open(DATA_SUMOAVGS +"/G.pkl", 'rb') as f:
        G = pickle.load(f)
    with open(DATA_SUMOAVGS + "/C.pkl", 'rb') as f:
        C = pickle.load(f)
    with open(DATA_SUMOAVGS + "/W.pkl", "rb") as f:
        WCollecetion = pickle.load(f)

    networkCollection.append((G, C, WCollecetion))

    print("Data prepared")
    print("Starting optimization")
    for network in networkCollection:

        if (network == '.DS_Store'):
            continue

        G, C, WCollecetion = network

        if (len(WCollecetion) % STEPSIZE != 0):
            print("ERROR: WCollecetion is not a multiple of STEPSIZE")
            print(len(WCollecetion) % STEPSIZE)
            print(len(WCollecetion))
            continue

        simulationTime = len(WCollecetion[::STEPSIZE])
        print(simulationTime)

        WAverageCollections = [WCollecetion[::STEPSIZE]]
        
        # WAVerageCollectionsNames = ["W0", "W10", "W50", "W100"]
        # WAVerageCollectionsNumbers = [1, 10, 50, 100]

        WAVerageCollectionsNames = ["W0"]
        WAVerageCollectionsNumbers = [1]

        print("Optimizing for network: " + simulationName)

        for t in tqdm.tqdm(range(len(WAVerageCollectionsNames))):
            
            mainFolder = os.path.join(DATA_SUMOAVGS, "model")
            
            if not os.path.exists(mainFolder):
                os.makedirs(mainFolder)

            baseName = WAVerageCollectionsNames[t]

            #check point
            checkPointFile, lastGeneration = find_latest_checkpoint(mainFolder, baseName)

            if checkPointFile:
                print("\t\n")
                print('LOADED CHECKPOINT')
                print('CONTINUE FROM GENERATION: ' + str(lastGeneration))
                population, bestFitness = load_checkpoint_from_file(checkPointFile)
            else:
                print('COULDNT FIND CHECKPOINT')
                lastGeneration = 0
                population, bestFitness = [], []  # Or initialize a fresh population as needed

            sum = np.zeros((simulationTime, len(WCollecetion[0])))

            startTime = time.time()
            
            print("Optimizing for W: " + WAVerageCollectionsNames[t])
            population, bestFitness = optimize(G,C,WAverageCollections[t], mutations=True, population=population, generation=lastGeneration)
            print("Optimization done for W: " + WAVerageCollectionsNames[t])
            
            endTime = time.time()
            executionTime = endTime - startTime

            with open(mainFolder + "/" + baseName + ".pkl", 'wb') as f:
                pickle.dump((population, bestFitness, executionTime), f)

    print("DONE")

