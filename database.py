from table import RelationalTable

class RelationalDatabase:
    def __init__(self):
        self.Tables: list[RelationalTable]

    # load all csv files within the folder into tables in this database
    def LoadFromFolder(self, data_folder: str):
        pass
    
    # assign integration IDs to the columns of each table in the database
    def AssignIntegrationIDs(self):
        pass

    # run the ALITE algorithm on the database
    def RunALITE(self):
        fullDisjunction = RelationalTable()

        for table in self.Tables:
            table.GenerateLabeledNulls()
            fullDisjunction.OuterUnionWith(table)
        
        fullDisjunction.Complement()
        fullDisjunction.ReplaceLabeledNulls()
        fullDisjunction.SubsumeTuples()

        return fullDisjunction

        

