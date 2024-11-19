import os
from table import RelationalTable

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

        offset = 0
        for idx, table in enumerate(self.Tables):
            offset = table.InitializeIntegrationIDs(offset)
            print(f"Table {idx} Integration IDs: {table.IntegrationIDToColumnIndex}")
        

        # assign integration IDs by clustering columns based on column name using TURL embeddings
        # (cannot do on a per-table basis, must take all tables into account, but can apply TURL embeddings per table, then do clustering)
        print("Integration IDs assigned to all tables.")

    # Run the ALITE algorithm on the database
    def RunALITE(self):

        # remove this once actually implemented (necessary for checking that visualizations work)
        #import time
        #import random
        #time.sleep(self.TupleCount() * 0.0005 + random.random() * 1.5)

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

    def RunBIComNLoj(self):

        # remove this once actually implemented (necessary for checking that visualizations work)
        import time
        import random
        time.sleep(self.TupleCount() * 0.0005 + random.random() * 1.5)

        pass