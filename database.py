import os
from table import RelationalTable

class RelationalDatabase:
    def __init__(self):
        self.Tables: list[RelationalTable] = []

    # load all csv files within the folder into tables in this database
    def LoadFromFolder(self, data_folder: str):
        for root, dirs, files in os.walk(data_folder):
            print(root)
            if os.path.realpath(root) == os.path.realpath(data_folder):
                for file in files:
                    print(f"Loading data from file {file} into relational table")
                    new_table = RelationalTable()
                    
                    filepath = os.path.join(root, file)
                    new_table.LoadFromCSV(filepath)
                    self.Tables.append(new_table)
    
    def TupleCount(self):
        count = 0
        for table in self.Tables:
            count += table.TupleCount()

    # assign integration IDs to the columns of each table in the database
    def AssignIntegrationIDs(self):
        pass

    # run the ALITE algorithm on the database
    def RunALITE(self):
        self.AssignIntegrationIDs()

        fullDisjunction = RelationalTable()

        for table in self.Tables:
            table.GenerateLabeledNulls()
            fullDisjunction.OuterUnionWith(table)
        
        fullDisjunction.Complement()
        fullDisjunction.ReplaceLabeledNulls()
        fullDisjunction.SubsumeTuples()

        return fullDisjunction

    def RunBIComNLoj(self):
        pass
        

