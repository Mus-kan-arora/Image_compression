import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# Load the Olivetti Faces dataset
faces = fetch_olivetti_faces()
X = faces.data  # Original data
Y = faces.target
print(Y)
n_samples, n_features = X.shape
n_components = 2  # Reduce to 2 dimensions for visualization
n_neighbors = 10  # Number of neighbors for ISOMAP

# Perform PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)  # Reduce dimensions

# Perform ISOMAP
isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
X_iso = isomap.fit_transform(X)  # Reduce dimensions

# Function to plot original images and reduced dimensions
def plot_all_dimensions(X, X_pca, X_iso, labels):
    plt.figure(figsize=(15, 10))

    # Original data scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', edgecolor='k', s=30)
    plt.title("Original Data Scatter Plot")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Face Class')

    # PCA scatter plot
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Spectral', edgecolor='k', s=30)
    plt.title("PCA Reduced Dimensions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Face Class')

    # ISOMAP scatter plot
    plt.subplot(1, 3, 3)
    plt.scatter(X_iso[:, 0], X_iso[:, 1], c=labels, cmap='Spectral', edgecolor='k', s=30)
    plt.title("ISOMAP Reduced Dimensions")
    plt.xlabel("ISOMAP Dimension 1")
    plt.ylabel("ISOMAP Dimension 2")
    plt.colorbar(label='Face Class')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the title position
    plt.show()

# Plotting all dimensions
plot_all_dimensions(X, X_pca, X_iso, faces.target)





