import numpy as np  
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA  
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering  
import matplotlib.pyplot as plt  

# Load the data from the CSV file  
data = pd.read_csv('molecular_descriptors.csv')  

# Extract the features for PCA  
features = data[['MolWt', 'logP', 'HBA', 'HBD', 'RTB', 'AR',   
                 'O', 'N', 'tPSA',   
                 'FSp3']]  

# Standardize the features  
scaler = StandardScaler()  
features_standardized = scaler.fit_transform(features)  

# Perform PCA  
pca = PCA(n_components=2)  
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

# Save PCA Plot  
plt.figure(figsize=(10, 8))  
plt.scatter(features_pca[:, 0], features_pca[:, 1])  
plt.title('PCA')  
plt.xlabel('PCA Component 1')  
plt.ylabel('PCA Component 2')  
plt.savefig('pca_plot.png')  
plt.close()  

# Save DBSCAN Plot using PCA components  
plt.figure(figsize=(10, 8))  
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=dbscan_labels, cmap='viridis')  
plt.title('DBSCAN')  
plt.xlabel('PCA Component 1')  
plt.ylabel('PCA Component 2')  
plt.savefig('dbscan_plot.png')  
plt.close()  

# Save K-means Plot using PCA components  
plt.figure(figsize=(10, 8))  
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=kmeans_labels, cmap='viridis')  
plt.title('K-means')  
plt.xlabel('PCA Component 1')  
plt.ylabel('PCA Component 2')  
plt.savefig('kmeans_plot.png')  
plt.close()  

# Save Agglomerative Clustering Plot using PCA components  
plt.figure(figsize=(10, 8))  
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=agg_labels, cmap='viridis')  
plt.title('Agglomerative Clustering')  
plt.xlabel('PCA Component 1')  
plt.ylabel('PCA Component 2')  
plt.savefig('agglomerative_clustering_plot.png')  
plt.close()  

print("Plots saved successfully to PNG files.")
