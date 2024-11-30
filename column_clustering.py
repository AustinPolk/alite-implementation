import numpy as np

class Point:
    def __init__(self, point: np.ndarray, index: int):
        self.Point: np.ndarray = point
        self.Index: int = index

class ColumnCluster:
    def __init__(self, first_point: np.ndarray, from_table: int, index: int):
        self.Points: list[Point] = [Point(first_point, index)]
        self.Center: np.ndarray = first_point
        self.Tables: set[int] = {from_table}
        self.ClusterIndex: int = index
    def distance_from(self, other):
        # check if these clusters contain columns from the same table
        if self.Tables & other.Tables:
            # disallow these clusters to be close by setting the distance to infinite
            return np.inf
        else:
            # compute euclidean distance
            t = self.Center - other.Center
            squared = np.dot(t.T, t)
            return np.sqrt(squared)
    def combine_with(self, other):
        # recalculate the cluster mean
        sum = self.Center * len(self.Points) + other.Center * len(other.Points)
        n = len(self.Points) + len(other.Points)
        self.Center = sum / n

        # add other points to this cluster
        self.Points.extend(other.Points)

        # record all tables that are now in this cluster
        self.Tables |= other.Tables

class ColumnClustering:
    def __init__(self, min_clusters: int):
        self.min_clusters_ = min_clusters
        self.labels: dict[int, list[int]] = {}
    def fit(self, column_embeddings: list[np.ndarray], from_table: list[int]):
        cluster_tuples = zip(column_embeddings, from_table, range(len(column_embeddings)))
        clusters = [ColumnCluster(embedding, table, idx) for embedding, table, idx in cluster_tuples]

        point_clusters = []
        for cluster in clusters:
            for point in cluster.Points:
                point_clusters.append((point.Index, cluster.ClusterIndex))
        point_clusters = sorted(point_clusters, key = lambda x: x[0])
        self.labels[len(clusters)] = [x[1] for x in point_clusters]

        while len(clusters) > self.min_clusters_:
            current_clusters = len(clusters)

            # find the closest pair of clusters
            closest_pair = None
            closest_distance = np.inf
            for i in range(current_clusters):
                for j in range(i + 1, current_clusters):
                    distance = clusters[i].distance_from(clusters[j])
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_pair = (i, j)
            
            # if no closest pair is found, that means the clustering can't go any further
            # without violating the constraint that columns from the same table must be in
            # different clusters
            if not closest_pair:
                print(f"Breaking out of cluster fitting at n={current_clusters}, too few clusters specified")
                self.broke_out = True
                break

            # combine the closest pair
            i, j = closest_pair
            clusters[i].combine_with(clusters[j])
            clusters.remove(clusters[j])

            point_clusters = []
            for cluster in clusters:
                for point in cluster.Points:
                    point_clusters.append((point.Index, cluster.ClusterIndex))
            point_clusters = sorted(point_clusters, key = lambda x: x[0])
            self.labels[current_clusters - 1] = [x[1] for x in point_clusters]