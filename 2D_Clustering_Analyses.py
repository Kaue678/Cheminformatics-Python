import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('output.csv')

# Extract the features for PCA
features = data[['MolWt', 'logP', 'HBA', 'HBD', 'NumRotBonds', 'NumAromaticRings', 'NumOxygenAtoms', 'NumNitrogenAtoms', 'TopologicalSurfaceArea', 'FractionSP3Carbons']]

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Perform PCA
pca = PCA(n_components=2)
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

# Plotting
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.scatter(features_pca[:, 0], features_pca[:, 1])
plt.title('PCA')

plt.subplot(2, 2, 2)
plt.scatter(features['MolWt'], features['logP'], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN')

plt.subplot(2, 2, 3)
plt.scatter(features['MolWt'], features['logP'], c=kmeans_labels, cmap='viridis')
plt.title('K-means')

plt.subplot(2, 2, 4)
plt.scatter(features['MolWt'], features['logP'], c=agg_labels, cmap='viridis')
plt.title('Agglomerative Clustering')

plt.tight_layout()
plt.show()

