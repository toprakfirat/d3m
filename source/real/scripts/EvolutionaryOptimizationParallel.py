import numpy as np

class candidate:
    def __init__(self, roadTypeLength):
        self.M =  np.random.rand(roadTypeLength, roadTypeLength) * 100 #M Matrix
        self.theta = np.random.rand(roadTypeLength) * 100 #Theta Vector
        self.lambdaVec = np.random.rand(roadTypeLength) * 100 #Lambda Vector
        self.error = 0 #Fitness
        self.D = [] #Data
        self.constantFlow = 5
        self.WT = []

    def __lt__(self, other):
        return self.error < other.error

    def __gt__(self, other):
         return self.error > other.error

    def setOffspring(self, M1, M2, theta1, theta2):
        self.M = np.maximum((M1 + M2) / 2 + ( np.random.uniform(-1, 1, (self.M.shape[0], self.M.shape[1])) * 10), 10)
        self.theta = theta1 + theta2 / 2 + np.random.uniform(-1, 1, (self.theta.shape)) * 10

    def mutate(self, mutationCoefficient=3, min_value=10, max_value=100, adaptive=False, generation=None, max_generation=None):
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
        if adaptive and generation is not None and max_generation is not None:
            # Scale mutation coefficient adaptively
            progress = generation / max_generation
            mutationCoefficient *= (1 - progress)  # Decrease mutations as evolution progresses

        # Mutate `M`
        mutation_matrix = np.random.uniform(-1, 1, self.M.shape) * mutationCoefficient
        self.M += mutation_matrix
        self.M = np.clip(self.M, min_value, max_value)  # Enforce bounds

        # Mutate `theta`
        mutation_vector = np.random.uniform(-1, 1, self.theta.shape) * mutationCoefficient
        self.theta += mutation_vector
        self.theta = np.clip(self.theta, min_value, max_value)  # Enforce bounds


def checkError(W, WT, C=None, MAXCOSTVALUE=9999999, dataAvailableIndices = None):
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
            
    if error > MAXCOSTVALUE or np.isnan(error):
        return MAXCOSTVALUE

    return error