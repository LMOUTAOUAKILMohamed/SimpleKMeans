import numpy as np
from scipy.spatial.distance import euclidean
import random

class SimpleKMeans:
    
    def __init__(self, k=2, max_iterations=10, tolerance=0.01):
        self.k = k
        self.max_iter = max_iterations
        self.tol = tolerance
        
    def init_random_centroids(self, X):
        centroids = np.empty([self.k, X.shape[1]])
        for cluster_index in range(self.k):
            centroids[cluster_index] = X[cluster_index].copy()
        return centroids
    
    def init_clusters(self):
        clusters = {}
        for cluster_index in range(self.k):
            clusters[cluster_index] = []
        return clusters
    
    def nearest_centroid(self, point):
        min_dist = euclidean(point, self.centroids[0])
        nearest_centroid = 0
        for centroid_index in range(self.k):
            dist = euclidean(point, self.centroids[centroid_index])
            if dist > min_dist:
                continue
            min_dist = dist
            nearest_centroid = centroid_index
        return nearest_centroid
            
    def cluster_points(self, X):
        clusters = self.init_clusters()
        for point in X:
            nearest_centroid_index = self.nearest_centroid(point)
            clusters[nearest_centroid_index].append(point)
        return clusters
    
    def calculate_new_centroids(self, clusters, x_dim):
        new_centroids = np.empty([self.k, x_dim])
        for cluster_index in range(self.k):
            if len(clusters[cluster_index]) == 0:
                new_centroids[cluster_index] = self.centroids[cluster_index].copy()
            else:
                new_centroids[cluster_index] = list(np.mean(clusters[cluster_index], axis=0))
        return new_centroids
        
    def fit(self, X):
        # Centroids initialization using random data points from X
        self.centroids = self.init_random_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign data points to the closest centroids (cluster)
            clusters = self.cluster_points(X)
            # Keep old centroids to compare with them
            old_centroids = self.centroids.copy()
            # Calculate the mean of data points of each cluster (the new centroids)
            self.centroids = self.calculate_new_centroids(clusters, X.shape[1])
            # Compare the centroids to check if we reached the optimal centroids
            self.optimum_reached = True
            for cluster_index in range(self.k):
                if euclidean(self.centroids[cluster_index], old_centroids[cluster_index]) > self.tol:
                    self.optimum_reached = False

            if self.optimum_reached == True:
                break
        self.n_iterations = iteration + 1
        return self
    
    def predict(self, X):
        classified_points = []
        for point in X:
            classified_points.append(self.nearest_centroid(point))
        return classified_points