import os
from table import RelationalTable
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

class RelationalDatabase:
    def __init__(self):
        self.Tables: list[RelationalTable] = []

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
        for idx, table in enumerate(self.Tables):
            print(f"Initializing table {idx}")
            offset = table.InitializeIntegrationIDs(offset)
            table.InitializeColumnEmbeddings(model)
            column_count = len(table.ColumnNames)

            if not minimum_columns or column_count < minimum_columns:
                minimum_columns = column_count
            maximum_columns += column_count
            
            print(f"Table {idx} embeddings: {table.ColumnEmbeddings}")

            all_column_embeddings.update(table.ColumnEmbeddings)
            all_integrationIDs.extend(table.IntegrationIDToColumnIndex.keys())
        all_embeddings = np.array(list(all_column_embeddings.values()))

        print(f"Total embeddings: {len(all_embeddings)}")
        print(f"Minimum columns: {minimum_columns}\tMaximum columns: {maximum_columns}")

        best_clustering = None
        best_score = -1

        # try all possible cluster sizes, select the size that maximizes silhouette score
        for n_clusters in range(minimum_columns, maximum_columns):
            print(f"Clustering into {n_clusters} clusters")
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            clustering.fit(all_embeddings)
            silhouette = silhouette_score(all_embeddings, clustering.labels_)
            print(f"Silhouette score for {n_clusters} clusters: {silhouette}")

            if best_score < silhouette:
                best_score = silhouette
                best_clustering = clustering

        print(f"Best clustering achieved using {best_clustering.n_clusters_} clusters")

        # now cluster the table columns with this model
        column_clusters = {id: cluster for cluster, id in zip(best_clustering.labels_, all_integrationIDs)}

        # now reassign the table column names to be which cluster that column is in (the cluster is the new integration ID)
        for idx, table in enumerate(self.Tables):
            table.RenameColumns(column_clusters)
            print(f"Table {idx} final integration IDs: {table.DataFrame.columns}")
            
        print("Integration IDs assigned to all tables.")

    # Run the ALITE algorithm on the database
    def RunALITE(self):

        # Step 1: Assign integration IDs
        self.AssignIntegrationIDs()

        # Step 2: Create a new table for the full disjunction
        fullDisjunction = RelationalTable()

        print("Outer Union Start")
        
        # Step 3: Generate labeled nulls for each table and perform outer union
        for table in self.Tables:
            table.GenerateLabeledNulls()
            fullDisjunction.OuterUnionWith(table)
            
        print("Outer Union Done")


        print("Complement Start")
        # Step 4: Complement phase
        fullDisjunction.Complement()
        
        print("Complement Done")

        # Step 5: Replace labeled nulls with actual values (if any replacement logic applies)
        fullDisjunction.ReplaceLabeledNulls()

        # Step 6: Subsumption - remove subsumable tuples
        fullDisjunction.SubsumeTuples()

        return fullDisjunction