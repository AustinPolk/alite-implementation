import pandas as pd

class RelationalTable:
    def __init__(self):
        self.IntegrationIDToColumnIndex: dict[int, int] = None
        self.DataFrame: pd.DataFrame = pd.DataFrame()

    # load csv data into the data frame
    def LoadFromCSV(self, csv_file: str):
        self.DataFrame = pd.read_csv(csv_file)

    def TupleCount(self):
        return len(self.DataFrame.index)

    # to be done on each table before outer union
    # need to be able to tell between missing nulls (which are replaced by labeled nulls) and nulls created from outer joining
    def GenerateLabeledNulls(self):
        pass

    # replace the labeled nulls with missing nulls
    def ReplaceLabeledNulls(self):
        pass

    # to be done between all tables in the database, uses the integration IDs to match columns
    def OuterUnionWith(self, other_table):
        # if this table is empty, just copy the other into this one
        if self.DataFrame.empty:
            self.IntegrationIDToColumnIndex = other_table.IntegrationIDToColumnIndex
            self.DataFrame = other_table.DataFrame
            return

        pass

    # to be done on the result of outer unions
    def Complement(self):
        pass

    # to be done on result of complementation
    def SubsumeTuples(self):
        pass