import pandas as pd
import numpy as np

class RelationalTable:
    def __init__(self):
        self.IntegrationIDToColumnIndex: dict[int, int] = {}
        self.DataFrame: pd.DataFrame = pd.DataFrame()
        self.labeled_null_counter = 0  # Counter to track unique labeled nulls

    # Load CSV data into the DataFrame
    def LoadFromCSV(self, csv_file: str):
        self.DataFrame = pd.read_csv(csv_file, encoding="ISO-8859-1", on_bad_lines='skip')

    def TupleCount(self):
        return len(self.DataFrame.index)

    # Assign unique integration IDs to each column (must be unique between tables as well, hence an offset)
    def InitializeIntegrationIDs(self, offset: int):
        for i in range(len(self.DataFrame.columns)):
            integrationID = i + offset
            column_index = i
            self.IntegrationIDToColumnIndex[integrationID] = column_index
        return offset + len(self.DataFrame.columns)
        
    class LabeledNull:
        def __init__(self, idx):
            self.idx = idx
        def __hash__(self):
            return self.idx
        def __eq__(self, other):
            return self.idx == other.idx

    # Generate labeled nulls to distinguish missing values in the data
    def GenerateLabeledNulls(self):
        def label_missing(value):
            if pd.isna(value):
                self.labeled_null_counter += 1
                return self.LabeledNull(self.labeled_null_counter)
            return value

        self.DataFrame = self.DataFrame.map(label_missing)

    # Replace labeled nulls back to NaN or missing values
    def ReplaceLabeledNulls(self):
        def remove_label(value):
            if isinstance(value, self.LabeledNull):
                return None  # Convert labeled nulls to None
            return value

        self.DataFrame = self.DataFrame.map(remove_label)

    # Perform an outer union with another table
    def OuterUnionWith(self, other_table):
        aligned_self = self.DataFrame.reindex(columns=self.IntegrationIDToColumnIndex.keys())
        aligned_other = other_table.DataFrame.reindex(columns=other_table.IntegrationIDToColumnIndex.keys())

        self.DataFrame = pd.concat([aligned_self, aligned_other], axis=0, ignore_index=True).fillna("")

        # TODO: will need to be a little more complex to capture the column indices in the new dataframe
        self.IntegrationIDToColumnIndex.update(other_table.IntegrationIDToColumnIndex)

    # Complement the DataFrame to ensure completeness of tuples
    def Complement(self):
        self.DataFrame = self.DataFrame.applymap(lambda x: x if x != "" else np.nan)
        # TODO: implement complementation from the paper
        print("Complement operation performed.")


    # Remove tuples that are subsumed by other tuples
    def SubsumeTuples(self):
        original_row_count = len(self.DataFrame)
        # TODO: also drop subsumable tuples, not just duplicates
        self.DataFrame.drop_duplicates(inplace=True)
        new_row_count = len(self.DataFrame)
        print(f"Subsumed tuples: {original_row_count - new_row_count}")
