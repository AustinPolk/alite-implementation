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
        for table in self.Tables:
            table.AssignIntegrationIDs()
        print("Integration IDs assigned to all tables.")

    # Run the ALITE algorithm on the database
    def RunALITE(self):
        # Step 1: Assign integration IDs
        self.AssignIntegrationIDs()

        # Step 2: Create a new table for the full disjunction
        fullDisjunction = RelationalTable()

        # Step 3: Generate labeled nulls for each table and perform outer union
        for table in self.Tables:
            table.GenerateLabeledNulls()
            fullDisjunction.OuterUnionWith(table)

        # Step 4: Complement phase
        fullDisjunction.Complement()

        # Step 5: Replace labeled nulls with actual values (if any replacement logic applies)
        fullDisjunction.ReplaceLabeledNulls()

        # Step 6: Subsumption - remove redundant tuples
        fullDisjunction.SubsumeTuples()

        return fullDisjunction

    def RunBIComNLoj(self):
        pass

# Initialize the database
database = RelationalDatabase()

# Load data from the benchmarks folder
# '311_calls_historic_data', 'abandoned_wells', 'city_jobs_requisition_requests'
data_folder = "benchmarks/abandoned_wells"
database.LoadFromFolder(data_folder)

# Run the ALITE algorithm on the loaded data
result_table = database.RunALITE()

# Print the resulting table
print("Full Disjunction Result:")
print(result_table.DataFrame)
