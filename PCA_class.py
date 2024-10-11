import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        PCA Class Constructor
        :param n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
    
    def fit_transform(self, data):
        """
        Fit the PCA model to the data and apply the transformation.
        :param data: The input data to be reduced (n_samples, n_features)
        :return: The reduced data (n_samples, n_components)
        """
        # Step 1: Center the data (subtract the mean of each feature)
        m = data.shape
        for i in range(m[1]):  
            feature = data[:, i]
            mean_i = sum(feature) / len(feature)
            data[:, i] = feature - mean_i
        
        self.mean_ = np.mean(data, axis=0)

        # Step 2: Compute the covariance matrix
        transpose_data = np.transpose(data)
        covariance = np.dot(transpose_data, data) / (m[0] - 1)

        # Step 3: Compute eigenvalues and eigenvectors
        eigen_value, eigen_vector = np.linalg.eig(covariance)

        # Step 4: Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigen_value)[::-1]
        eigen_value_sorted = eigen_value[sorted_indices]
        eigen_vector_sorted = eigen_vector[:, sorted_indices]

        # Step 5: Select the top 'k' eigenvectors (k = n_components)
        top_k_eigen = eigen_vector_sorted[:, :self.n_components]

        # Step 6: Transform the data to the new space
        reduced_data = np.dot(data, top_k_eigen)

        # Store the components
        self.components_ = top_k_eigen

        return reduced_data
    
    def inverse_transform(self, Z):
        return np.dot(Z, self.components_.T) + self.mean_  # Add the mean back