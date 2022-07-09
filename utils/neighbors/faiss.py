"""Class used for implementing faster nearest neighbor lookup. Written in C++,
FAISS is used for efficient similarity searches. For interpolated_replay experience
replay, FAISS is used to query nearest neighbors."""
# External Python packages
import faiss
import torch


class FaissKNeighbors:
    """An implementation of FAISS using L2 norms.

    Parameters:
        k (int): The number of neighbors we consider for the FAISS tree.
        gpu_id (str): The string ID corresponding to the GPU to be used with
            FAISS. Defaults to None.
    """
    def __init__(self, k=50, gpu_id=None):
        self.index = None
        self.k = k
        self.gpu_id = gpu_id

    def fit(self, X):
        """Function to fit the FAISS index.

        This method fits the Index corresponding to the FAISS tree with
        neighbor data used for querying.

        Parameters:
            X (np.array):  Array of shape (N, d), where N is the number of
                samples and d is the dimension of the data. Note that the
                array must be of type np.float32.
        """
        if self.gpu_id is None:  # Use CPU only
            self.index = faiss.IndexFlatL2(X.shape[1])
        else:  # Use GPU
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = self.gpu_id

            flat_config = [cfg]
            resources = [faiss.StandardGpuResources()]
            self.index = faiss.GpuIndexFlatL2(
                resources[0], X.shape[1], flat_config[0])

        # Add to the Flat Index
        self.index.add(X)

    def query(self, X, k=None):
        """Function to query the neighbors of the FAISS tree.

        Parameters:
            X (np.array):  Array of shape (N, D), where N is the number of
                samples, and D is the dimension of the features.
            k (int):  If provided, the number of neighbors to compute. Defaults
                to None, in which case self.k is used as the number of neighbors.

        Returns:
            indices (np.array): Array of shape (N, K), where N is the number of
                samples, and K is the number of nearest neighbors to be computed.
                The ith row corresponds to the k-nearest neighbors of the ith
                sample.
        """
        # Set number of neighbors
        if k is None:  # Use default number of neighbors
            k = self.k

        # Query and return nearest neighbors
        _, indices = self.index.search(X, k=k)
        return indices
