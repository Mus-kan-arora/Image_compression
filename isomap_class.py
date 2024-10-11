import numpy as np
from sklearn.neighbors import NearestNeighbors

class Isomap:
    def __init__(self, n_neighbors=5, n_components=2):
        """
        Initialize the Isomap model.
        
        Parameters:
        - n_neighbors: Number of nearest neighbors to use for graph construction.
        - n_components: The number of dimensions to reduce to.
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit_transform(self, X):
        """
        Fit the model to the data and apply Isomap for dimensionality reduction.

        Parameters:
        - X: The input data, shape (n_samples, n_features)

        Returns:
        - Y: The reduced dimensional data, shape (n_samples, n_components)
        """
        k_matrix = self._k_neighbors_graph(X)
        g_matrix = self._geodesic_distance(k_matrix)
        self.embedding_ = self._mds(g_matrix)
        return self.embedding_


    def _k_neighbors_graph(self, X):
        """
        Compute the k-nearest neighbors graph for the dataset.

        Parameters:
        - X: The input data, shape (n_samples, n_features)

        Returns:
        - k_matrix: K-nearest neighbor distance matrix, shape (n_samples, n_samples)
        """
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        distances, indices = nbrs.fit(X).kneighbors(X)
        
        # Initialize the k-nearest neighbor graph with infinite distances
        n_samples = X.shape[0]
        k_matrix = np.full((n_samples, n_samples), np.inf)  # Full distance matrix
        
        # Populate the k-nearest neighbor distances
        for i in range(n_samples):
            k_matrix[i, indices[i]] = distances[i]
        
        return k_matrix

    def _geodesic_distance(self, k_matrix):
        """
        Compute the geodesic distance matrix using the Floyd-Warshall algorithm.

        Parameters:
        - k_matrix: K-nearest neighbor distance matrix, shape (n_samples, n_samples)

        Returns:
        - g_matrix: Geodesic distance matrix, shape (n_samples, n_samples)
        """
        n_samples = k_matrix.shape[0]
        g_matrix = k_matrix.copy()

        # Floyd-Warshall algorithm for shortest paths
        for k in range(n_samples):
            for i in range(n_samples):
                for j in range(n_samples):
                    g_matrix[i, j] = min(g_matrix[i, j], g_matrix[i, k] + g_matrix[k, j])
        
        return g_matrix

    def _mds(self, g_matrix):
        
        # if np.any(np.isnan(g_matrix)):
        #     raise ValueError("Geodesic distance matrix contains NaN values.")
        # if np.any(np.isinf(g_matrix)):
        #     raise ValueError("Geodesic distance matrix contains infinite values.")
        
        """
        Perform classical Multidimensional Scaling (MDS) on the geodesic distance matrix.

        Parameters:
        - g_matrix: Geodesic distance matrix, shape (n_samples, n_samples)

        Returns:
        - Y: Reduced dimensional data, shape (n_samples, n_components)
        """
        n_samples = g_matrix.shape[0]
        
        # Centering matrix H
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        
        # Double centering the squared distances matrix B = -0.5 * H * (D^2) * H
        B = -0.5 * H @ (g_matrix ** 2) @ H

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(B)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvals)[::-1]
        sorted_eigenvals = eigenvals[sorted_indices]
        sorted_eigenvecs = eigenvecs[:, sorted_indices]

        # Select top components
        Y = sorted_eigenvecs[:, :self.n_components] @ np.diag(np.sqrt(sorted_eigenvals[:self.n_components]))
        
        return Y



# import numpy as np
# from sklearn.neighbors import NearestNeighbors

# class Isomap:
#     def __init__(self, n_neighbors=5, n_components=2):
#         """
#         Initialize the Isomap model.
        
#         Parameters:
#         - n_neighbors: Number of nearest neighbors to use for graph construction.
#         - n_components: The number of dimensions to reduce to.
#         """
#         self.n_neighbors = n_neighbors
#         self.n_components = n_components

#     def fit_transform(self, X):
#         """
#         Fit the model to the data and apply Isomap for dimensionality reduction.

#         Parameters:
#         - X: The input data, shape (n_samples, n_features)

#         Returns:
#         - Y: The reduced dimensional data, shape (n_samples, n_components)
#         """
#         k_matrix = self._k_neighbors_graph(X)
#         g_matrix = self._geodesic_distance(k_matrix)
#         return self._mds(g_matrix)

#     def _k_neighbors_graph(self, X):
#         """
#         Compute the k-nearest neighbors graph for the dataset.

#         Parameters:
#         - X: The input data, shape (n_samples, n_features)

#         Returns:
#         - k_matrix: K-nearest neighbor distance matrix, shape (n_samples, n_samples)
#         """
#         nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
#         distances, indices = nbrs.fit(X).kneighbors(X)
        
#         # Initialize the k-nearest neighbor graph with infinite distances
#         n_samples = X.shape[0]
#         k_matrix = np.full((n_samples, n_samples), np.inf)  # Full distance matrix
        
#         # Populate the k-nearest neighbor distances
#         for i in range(n_samples):
#             k_matrix[i, indices[i]] = distances[i]
        
#         return k_matrix

#     def _geodesic_distance(self, k_matrix):
#         """
#         Compute the geodesic distance matrix using the Floyd-Warshall algorithm.

#         Parameters:
#         - k_matrix: K-nearest neighbor distance matrix, shape (n_samples, n_samples)

#         Returns:
#         - g_matrix: Geodesic distance matrix, shape (n_samples, n_samples)
#         """
#         n_samples = k_matrix.shape[0]
#         g_matrix = k_matrix.copy()

#         # Floyd-Warshall algorithm for shortest paths
#         for k in range(n_samples):
#             for i in range(n_samples):
#                 for j in range(n_samples):
#                     g_matrix[i, j] = min(g_matrix[i, j], g_matrix[i, k] + g_matrix[k, j])
        
#         return g_matrix

#     def _mds(self, g_matrix):
#         """
#         Perform classical Multidimensional Scaling (MDS) on the geodesic distance matrix.

#         Parameters:
#         - g_matrix: Geodesic distance matrix, shape (n_samples, n_samples)

#         Returns:
#         - Y: Reduced dimensional data, shape (n_samples, n_components)
#         """
#         n_samples = g_matrix.shape[0]
        
#         # Centering matrix H
#         H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        
#         # Double centering the squared distances matrix B = -0.5 * H * (D^2) * H
#         B = -0.5 * H @ (g_matrix ** 2) @ H

#         # Eigenvalue decomposition
#         eigenvals, eigenvecs = np.linalg.eigh(B)

#         # Sort eigenvalues and corresponding eigenvectors in descending order
#         sorted_indices = np.argsort(eigenvals)[::-1]
#         sorted_eigenvals = eigenvals[sorted_indices]
#         sorted_eigenvecs = eigenvecs[:, sorted_indices]

#         # Select top components
#         Y = sorted_eigenvecs[:, :self.n_components] @ np.diag(np.sqrt(sorted_eigenvals[:self.n_components]))
        
#         return Y
