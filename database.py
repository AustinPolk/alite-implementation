import os
from table import RelationalTable
from sentence_transformers import SentenceTransformer
from column_clustering import ColumnClustering
from sklearn.metrics import silhouette_score
import numpy as np

class RelationalDatabase:
    def __init__(self):
        self.Tables: list[RelationalTable] = []
        self.IntegrationIDsAssigned: bool = False
        # for benchmarking purposes
        self.SilhouetteScores: dict[int, float] = {}
        self.ColumnClusterSizes: list[int] = None

    # Load all CSV files within the folder into tables in this database
    def LoadFromFolder(self, data_folder: str):
        for root, dirs, files in os.walk(data_folder):
            if os.path.realpath(root) == os.path.realpath(data_folder):
                for file in files:
                    print(f"Loading data from file {file} into relational table")
                    new_table = RelationalTable()
                    filepath = os.path.join(root, file)
                    new_table.LoadFromCSV(filepath)
                    self.Tables.append(new_table)

    def TupleCount(self):
        return sum(table.TupleCount() for table in self.Tables)

    # Assign integration IDs to the columns of each table in the database
    def AssignIntegrationIDs(self):
        # load a pretrained transformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # minimum and maximum columns that could be in the full disjunction
        minimum_columns = 0
        maximum_columns = 0

        # initialize the tables with unique integration IDs and column embeddings
        offset = 0
        all_integrationIDs = []
        all_column_embeddings = {}
        from_table = []
        for idx, table in enumerate(self.Tables):
            print(f"Initializing table {idx}")
            offset = table.InitializeIntegrationIDs(offset)
            table.InitializeColumnEmbeddings(model)
            column_count = len(table.ColumnNames)

            # minimum columns is the size of the largest single table
            if not minimum_columns or column_count > minimum_columns:
                minimum_columns = column_count
            # maximum columns is the sum of the sizes of all tables
            maximum_columns += column_count
            
            #print(f"Table {idx} embeddings: {table.ColumnEmbeddings}")

            all_column_embeddings.update(table.ColumnEmbeddings)
            all_integrationIDs.extend(table.IntegrationIDToColumnIndex.keys())
            from_table.extend([idx]*column_count)
        all_embeddings = list(all_column_embeddings.values())

        print(f"Total embeddings: {len(all_embeddings)}")
        print(f"Minimum columns: {minimum_columns}\tMaximum columns: {maximum_columns}")

        # compute all possible clusterings here, choose from them below
        print("Clustering column embeddings")
        column_clustering = ColumnClustering(min_clusters=minimum_columns)
        column_clustering.fit(all_embeddings, from_table)
        best_clustering = None
        best_score = -1

        # try all possible cluster sizes, select the size that maximizes silhouette score
        for n_clusters in range(minimum_columns, maximum_columns):
            
            if n_clusters not in column_clustering.labels:
                print(f"Skipping {n_clusters} clusters")
                continue
            cluster_labels = column_clustering.labels[n_clusters]

            silhouette = silhouette_score(all_embeddings, cluster_labels)
            self.SilhouetteScores[n_clusters] = silhouette
            print(f"Silhouette score for {n_clusters} clusters: {silhouette}")

            if best_score < silhouette:
                best_score = silhouette
                best_clustering = cluster_labels

        print(f"Best clustering achieved using {len(set(best_clustering))} clusters")
        self.ColumnClusterSizes = [minimum_columns, maximum_columns, len(set(best_clustering))]

        # now cluster the table columns with this model
        column_clusters = {id: cluster for cluster, id in zip(best_clustering, all_integrationIDs)}

        # now reassign the table column names to be which cluster that column is in (the cluster is the new integration ID)
        for idx, table in enumerate(self.Tables):
            table.RenameColumns(column_clusters)
            print(f"Table {idx} ({table.TableName}) final integration IDs: {table.DataFrame.columns}")
            print(f"Integration ID Mapping: {table.ColumnNames}")
            
        self.IntegrationIDsAssigned = True
        print("Integration IDs assigned to all tables.")

    # Run the ALITE algorithm on the database
    def RunALITE(self, output_folder: str):

        # Step 1: Assign integration IDs
        if not self.IntegrationIDsAssigned:
            self.AssignIntegrationIDs()

        # Step 2: Create a new table for the full disjunction
        fullDisjunction = RelationalTable()

        print("Outer Union Start")
        
        # Step 3: Generate labeled nulls for each table and perform outer union
        for table in self.Tables:
            table.GenerateLabeledNulls()
            fullDisjunction.OuterUnionWith(table)
        
        fullDisjunction.saveToFile(os.path.join(output_folder, "1 - PostOuterJoinAndLabeledNulls"))
            
        print("Outer Union Done")
        print(f"Tuple count: {fullDisjunction.TupleCount()}")

        print("Complement Start")
        # Step 4: Complement phase
        fullDisjunction.Complement()

        fullDisjunction.saveToFile(os.path.join(output_folder, "2 - PostComplement"))
        print(f"Tuple count: {fullDisjunction.TupleCount()}")
        
        print("Complement Done")

        # Step 5: Replace labeled nulls with actual values (if any replacement logic applies)
        fullDisjunction.ReplaceLabeledNulls()

        fullDisjunction.saveToFile(os.path.join(output_folder, "3 - ReplacingLabeledNulls"))

        # Step 6: Subsumption - remove subsumable tuples
        fullDisjunction.SubsumeTuples()

        fullDisjunction.saveToFile(os.path.join(output_folder, "4 - PostSubsumption"))
        print(f"Tuple count: {fullDisjunction.TupleCount()}")

        return fullDisjunction