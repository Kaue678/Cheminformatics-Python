import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the CSV file
data = pd.read_csv('output.csv')

# Extract the features for PCA
features = data[['MolWt', 'logP', 'HBA', 'HBD', 'NumRotBonds', 'NumAromaticRings', 'NumOxygenAtoms', 'NumNitrogenAtoms', 'TopologicalSurfaceArea', 'FractionSP3Carbons']]

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Perform PCA with three components
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features_standardized)

# Standardize the data for clustering algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_standardized)

# Clustering with DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Clustering with K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Clustering with Agglomerative Hierarchical Clustering
agg = AgglomerativeClustering(n_clusters=4)
agg_labels = agg.fit_predict(X_scaled)

# Plotting in 3D
fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2])
ax1.set_title('PCA')

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.scatter(features['MolWt'], features['logP'], features['HBA'], c=dbscan_labels, cmap='viridis')
ax2.set_title('DBSCAN')

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.scatter(features['MolWt'], features['logP'], features['HBA'], c=kmeans_labels, cmap='viridis')
ax3.set_title('K-means')

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(features['MolWt'], features['logP'], features['HBA'], c=agg_labels, cmap='viridis')
ax4.set_title('Agglomerative Clustering')

plt.tight_layout()
plt.show()

