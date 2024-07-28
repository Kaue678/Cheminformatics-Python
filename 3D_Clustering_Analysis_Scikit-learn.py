import numpy as np  
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA  
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  

# Load the data from the CSV file  
data = pd.read_csv('molecular_descriptors.csv')  

# Extract the features for PCA  
features = data[['MolWt', 'logP', 'HBA', 'HBD', 'RTB', 'AR', 'O', 'N', 'tPSA', 'FSp3']]  

# Standardize the features  
scaler = StandardScaler()  
features_standardized = scaler.fit_transform(features)  

# Perform PCA with three components  
pca = PCA(n_components=3)  
features_pca = pca.fit_transform(features_standardized)  

# Clustering with DBSCAN using PCA features  
dbscan = DBSCAN(eps=0.5, min_samples=5)  
dbscan_labels = dbscan.fit_predict(features_pca)  

# Clustering with K-means using PCA features  
kmeans = KMeans(n_clusters=4, random_state=42)  
kmeans_labels = kmeans.fit_predict(features_pca)  

# Clustering with Agglomerative Hierarchical Clustering using PCA features  
agg = AgglomerativeClustering(n_clusters=4)  
agg_labels = agg.fit_predict(features_pca)  

# Plotting in 3D  
fig = plt.figure(figsize=(14, 10))  

# Subplot for PCA  
ax1 = fig.add_subplot(2, 2, 1, projection='3d')  
ax1.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2])  
ax1.set_title('PCA')  

# Subplot for DBSCAN  
ax2 = fig.add_subplot(2, 2, 2, projection='3d')  
ax2.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], c=dbscan_labels, cmap='viridis')  
ax2.set_title('DBSCAN')  

# Subplot for K-means  
ax3 = fig.add_subplot(2, 2, 3, projection='3d')  
ax3.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], c=kmeans_labels, cmap='viridis')  
ax3.set_title('K-means')  

# Subplot for Agglomerative Clustering  
ax4 = fig.add_subplot(2, 2, 4, projection='3d')  
ax4.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], c=agg_labels, cmap='viridis')  
ax4.set_title('Agglomerative Clustering')  

# Adjust layout and save as PNG  
plt.tight_layout()  

# Save the figure as a PNG file  
plt.savefig('clustering_visualization.png')  

# Show the plot (optional, you can remove this line if you only want to save)  
plt.show()
