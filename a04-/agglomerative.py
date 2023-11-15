import numpy as np
import time as time
import numpy as np
def euclidean_distance(point1, point2):
    """
    Computes euclidean distance of point1 and point2.
    
    point1 and point2 are lists.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def clusters_distance(cluster1, cluster2):
    """
    Computes max distance between two clusters. 
    
    we use single linkage as per: \
    https://medium.com/@codingpilot25/hierarchical-clustering-and-linkage-explained-in-simplest-way-eef1216f30c5
    
    cluster1 and cluster2 are lists of lists of points
    """
    return min([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
def clusters_distance_2(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters
    
    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)

class AgglomerativeClustering:
    
    def __init__(self, n=2, initial_n=25):
        # initialzing no. of initial and final clusters 
        self.n = n
        self.initial_n= initial_n
        
    def initial_clusters(self, points):
        """
        partition pixels into self.initial_k groups based on color similarity
        """
        initialclusters = {}
        d = int(256 / (self.initial_n))
        for i in range(self.initial_n):
            j = i * d
            initialclusters[(j, j, j)] = []
        for i, p in enumerate(points):
            # keep track
            if i%100000 == 0:
                print('processing pixel:', i)
            most_similar = min(initialclusters.keys(), key=lambda c: euclidean_distance(p, c))  
            initialclusters[most_similar].append(p)
        return [g for g in initialclusters.values() if len(g) > 0]
        
    def fit(self, points):

        # initially, assign each set of similar points to a distinct cluster 
        self.clusters_list = self.initial_clusters(points)


        while len(self.clusters_list) > self.n:

            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                 key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

        
        self.cluster = {}
        for cluster_number, cluster in enumerate(self.clusters_list):
            for point in cluster:
                self.cluster[tuple(point)] = cluster_number
                

        self.centers = {}
        for cluster_number, cluster in enumerate(self.clusters_list):
            self.centers[cluster_number] = np.average(cluster, axis=0)
                    


    def cluster_pred(self, point):
        """
        Find cluster number of point
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def center_pred(self, point):
        """
        Find center of the cluster that point belongs to
        """
        point_cluster_num = self.cluster_pred(point)
        center = self.centers[point_cluster_num]
        return center