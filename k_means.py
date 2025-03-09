import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("Mall_Customers.csv" )

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 1: Finding the optimal number of clusters using the Elbow Method
sse = []
for i in range(1, 11):  # Checking cluster numbers from 1 to 10
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8,6))
plt.plot(range(1, 11),sse, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 2: Apply K-Means with the optimal number of clusters
optimal_clusters = 5  # Based on the elbow method
kmeans = KMeans(n_clusters=optimal_clusters)
df['Cluster'] = kmeans.fit_predict(X)

# Step 3: Plot both clusters and centroids in the same graph
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']  # Colors for clusters

# Plot each cluster
for i in range(optimal_clusters):
    plt.scatter(X[df['Cluster'] == i]['Annual Income (k$)'],
                X[df['Cluster'] == i]['Spending Score (1-100)'],
                s=100, c=colors[i], label=f'Cluster {i}', edgecolors='black', alpha=0.8)

# Plot centroids (on the same figure)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=400, c='yellow', marker='X', label='Centroids', edgecolors='black')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
