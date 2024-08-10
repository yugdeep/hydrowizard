import numpy as np


class RBFNetwork:
    def __init__(self, input_dim, output_dim, centers, betas, weights):
        self.num_rbfs = centers.shape[0]
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.centers = np.array(centers, dtype=np.float32)
        self.betas = np.array(betas, dtype=np.float32)
        self.weights = np.array(weights, dtype=np.float32)

    def rbf(self, x, center, beta):
        diff = x - center
        diff_squared = np.sum(diff**2)
        beta_squared = np.sum(beta**2)
        return np.exp(-diff_squared / beta_squared)

    def evaluate(self, x):
        G = np.zeros((self.num_rbfs))
        for j in range(self.num_rbfs):
            center_j = self.centers[j, :]
            beta_j = self.betas[j, :]
            G[j] = self.rbf(x, center_j, beta_j)
        output = G.dot(self.weights)
        return output


# # Example usage:
# input_dim = 4
# output_dim = 2
# num_rbfs = input_dim + output_dim
# centers = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 1,0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 1,0, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0]).reshape(num_rbfs, input_dim) # Shape (num_rbfs, input_dim)
# print("\nCenters\n",centers)
# betas = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]).reshape(num_rbfs, input_dim)  # Shape (num_rbfs, input_dim)
# print("\nBetas\n",betas)
# weights = np.array([0.5, 0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(num_rbfs, output_dim)  # Shape (num_rbfs, output_dim)
# print("\nWeights\n",weights)
# alphas = np.array([0.1, 0.2]) # Shape (output_dim)
# print("\nAlphas\n",alphas)
# rbf_net = RBFNetwork(input_dim, output_dim, centers, betas, weights, alphas)

# # Test with a sample input
# x = np.array([1.5, 2.5, 3.0, 4.0]) #, [2.5, 3.5, 4.1, 4.5]])
# output = rbf_net.evaluate(x)
# print("\nOutput\n",output)
