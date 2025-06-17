import networkx as nx
import numpy as np
import random
from tqdm import tqdm

class simulator():
    
    def __init__(self, G, theta, highwayMap, constantFlow = 5, L0 = []):

    ############################################################################################################
    #    # G         : Graph        (NetworkX Graph)                                
    #    # LINE      : Line graph of G                                                      
    #    # L0        : Initial Loads
    #    # C         : Capacities
    #    # W         : Simulation holder, each index holds the load distribution
    #    # edgeIndex : Dictionary of edge index
    #    # neighbors : Dictionary of neighbors of each edge
    #    # TIJ       : Tij gives a list by time that is calculated values of Tij
    #    # MIJ       : MIJ gives a list by time that is calculated values of MIJ
    #    # thetas    : Dictionary of thetas
    #    # highwayMap: Dictionary of highwayMap
    ############################################################################################################

        self.G          = G
        self.LINE       = nx.line_graph(G)
        self.L0         = []
        self.C          = dict()
        self.W          = []
        self.edgeIndex  = dict()
        self.inneighs   = dict()
        self.outneighs  = dict()
        self.TIJ        = dict()
        self.MIJ        = dict()
        self.theta     = theta
        self.thetas     = dict()
        self.highwayMap = highwayMap
        self.thetaCoefficientMap = dict()
        self.constantFlow = constantFlow
        self.simCost = 0

        REVLINE = nx.reverse_view(self.LINE)

        self.FLOATPRECISION = 5 

        t = 0
        for i in self.G.edges:
            
            if len(L0) == 0:
                self.G.edges[i]['load'] = round(self.G.edges[i]['load'], self.FLOATPRECISION)
                #Set load and capacity and theta
                self.L0.append(self.G.edges[i]["load"])
            else:
                self.L0 = L0

            self.C[i] = self.G.edges[i]["capacity"]
            self.thetas[i] = theta[self.highwayMap[self.G.edges[i]["highway"]]]
            
            #Neighbors
            outneighs = list(self.LINE.neighbors(i))
            inneighs  = list(REVLINE.neighbors(i))
            
            self.outneighs[i] = outneighs
            self.inneighs[i]  = inneighs

            for neighbor in outneighs:
                self.TIJ[(i, neighbor)] = []
                self.MIJ[(i, neighbor)] = (self.highwayMap[self.G.edges[i]["highway"]], self.highwayMap[self.G.edges[neighbor]["highway"]])
                 
            self.edgeIndex[i] = t
            t += 1
        
        #Set theta coefficient map
        for i in self.G.edges:
            totalTheta = 0

            for neighbor in self.outneighs[i]:
                totalTheta += self.thetas[neighbor]

            for neighbor in self.outneighs[i]:
                self.thetaCoefficientMap[(i, neighbor)] = self.thetas[neighbor] / totalTheta
                
        self.W.append(self.L0)

        junctionCoefficientDictionary = dict()
        for i in G.nodes:
            neighboringEdges =  G.in_edges(i, keys=True)
            neighboringEdgesSize = len(neighboringEdges)
            for j in neighboringEdges:
                junctionCoefficientDictionary[j] = 1/neighboringEdgesSize
        self.junctionCoefficientDictionary = junctionCoefficientDictionary

    def simulate(self, M, SIMTIME=100, randomShare=False, jamModel=None, startTime = 9, stepsInMinutes = 5):

        self.M = M

        for t in tqdm(range(SIMTIME)):

            Lt = self.W[t]

            if jamModel is not None:
                hour = startTime + (t * stepsInMinutes) / 60
                J_target = jamModel(hour % 24)  # ensure within [0, 24) for daily cycle
                Lt = self.adjust_load_to_match_jam(Lt, list(self.C.values()), J_target)

            
            for tij in self.TIJ.keys():
                val = self.Tij(tij[0], tij[1], t, Lt, M, randomShare = randomShare)
                self.TIJ[tij].append(val)

            Ltp1 = []
            for edge in self.G.edges:
                inFlow = 0
                outflow = 0

                for inneigh in self.inneighs[edge]:
                    inFlow += self.TIJ[(inneigh, edge)][t]

                for outneigh in self.outneighs[edge]:
                    outflow += self.TIJ[(edge, outneigh)][t]

                newLoad = Lt[self.edgeIndex[edge]] + inFlow - outflow

                if self.closeToZero(newLoad):
                    newLoad = 0

                if newLoad < 0:
                    print("Negative load detected")
                    print("Time: ", t)
                    print("Edge: ", edge)
                    raise ValueError("Negative load detected")

                Ltp1.append(newLoad)

            if randomShare:
                self.distributeRandomLoad(Ltp1)
                
            self.W.append(Ltp1)

    def roundedDirichlet(self, a, decimalPlaces = 5):
        a = np.array([round(element, decimalPlaces) for element in a])
        leftOut = round(1 - a.sum(), decimalPlaces)
        a[random.randint(0, len(a) - 1)] += leftOut
        return a

    def distributeRandomLoad(self, loadDist):
        #Randomly select roads and distribute a small part of load to neighbors
        for i in self.G.edges:
            index = self.edgeIndex[i]
            load = loadDist[index]
            #Select the road if random 
            if load < 0.01 or np.random.uniform(0, 1) > 0.4:
                continue
            else:
                outNeighs = self.outneighs[i]
                
                #Select a load to distribute to neighbors
                selectedLoad = round(np.random.uniform(0, 0.5) * load, self.FLOATPRECISION)
                selectedDistribution = np.random.dirichlet(np.ones(len(outNeighs)),size=1)[0]

                t = 0
                sentLoads = []
                for neighbor in self.outneighs[i]:
                    potentialSentLoad = np.round(selectedDistribution[t] * selectedLoad, self.FLOATPRECISION)
                    neighborIndex = self.edgeIndex[neighbor]

                    neighborCapacity = self.C[neighbor]
                    neighborLoad = loadDist[neighborIndex]
                    
                    
                    if potentialSentLoad <= neighborCapacity - neighborLoad:
                        sentLoad = potentialSentLoad * selectedDistribution[t]
                        loadDist[neighborIndex] = loadDist[neighborIndex] + sentLoad
                        sentLoads.append(sentLoad)

                    t +=1

                for sent in sentLoads:
                    loadDist[index] = loadDist[index] - sent
                    

    def Tij(self, i, j, t, L, M, randomShare=False):

        l = L[self.edgeIndex[i]]
        if (l <= 1e-5):
            return 0
        
        # Sigma
        sigmaOut = l * self.thetaCoefficientMap[(i, j)]

        # F
        epower =  ( - (l / self.C[i])) + 1
            
        if (self.closeToZero(epower)):
            fOut = self.constantFlow
        else: 
            fOut = M[self.MIJ[(i, j)]]  * (np.exp(epower) - 1) / (np.e - 1) + self.constantFlow        

        # if (randomShare and np.random.uniform(0, 1) > 0.4):
        #     fOut = min(fOut + np.random.uniform(-0.5, 0.5), 0)
        
        i = (i[0], i[1], 0)
        pijOut = min(sigmaOut, fOut * self.junctionCoefficientDictionary[i])       

        # S
        loadSum = 0
        for inneigh in self.inneighs[j]:
            loadSum += L[self.edgeIndex[inneigh]]
        if loadSum == 0:
            return 0
        loadCoeff = l / loadSum
        sOut = (self.C[j] - L[self.edgeIndex[j]]) * loadCoeff

        # Tij
        tijout = min(pijOut, sOut)

        #tijout = 0
        return tijout
    
    def closeToZero(self, val):
        return val <= 1e-5
    
    def adjust_load_to_match_jam(self, L_hat, C, J_target, adjustStep=0.5):
        """
        Adjusts the current load vector L_hat to match the desired average jam factor J_target.
        Can increase or decrease load. Capacity-aware. Safe.
        """
        L_hat = np.array(L_hat)
        C = np.array(C)
        n = len(C)

        J_hat = np.sum(L_hat / C) / n

        if np.isclose(J_target, J_hat, atol=1e-5):
            return L_hat.copy()

        elif J_target > J_hat:
            # Increase load proportionally to remaining capacity
            alpha = (J_target - J_hat) / (1 - J_hat)
            alpha = min(alpha, 1.0)
            headroom = C - L_hat
            delta_L = alpha * headroom
            return L_hat + delta_L * adjustStep

        else:
            # Decrease load proportionally to current load
            beta = 1 - (J_target / J_hat)
            beta = min(beta, 1.0)
            delta_L = beta * L_hat
            return L_hat - delta_L * adjustStep
