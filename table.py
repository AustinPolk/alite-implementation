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
        
    # Generate labeled nulls to distinguish missing values in the data
    def GenerateLabeledNulls(self):
        def label_missing(value):
            if pd.isna(value):
                self.labeled_null_counter += 1
                return f"LN{self.labeled_null_counter}"
            return value

        self.DataFrame = self.DataFrame.map(label_missing)

    # Replace labeled nulls back to NaN or missing values
    def ReplaceLabeledNulls(self):
        def remove_label(value):
            if isinstance(value, str) and value.startswith("LN"):
                return np.nan  # Convert labeled nulls back to NaN
            return value

        self.DataFrame = self.DataFrame.map(remove_label)

    # Perform an outer union with another table
    def OuterUnionWith(self, other_table):
        # Get all unique integration IDs
        all_integration_ids = set(self.IntegrationIDToColumnIndex.keys()).union(other_table.IntegrationIDToColumnIndex.keys())
        
        # Align columns based on integration IDs
        aligned_self = self.DataFrame.copy()
        aligned_other = other_table.DataFrame.copy()

        # Map columns to integration IDs
        aligned_self.columns = [self.IntegrationIDToColumnIndex.get(col, col) for col in aligned_self.columns]
        aligned_other.columns = [other_table.IntegrationIDToColumnIndex.get(col, col) for col in aligned_other.columns]

        # Reindex DataFrames to have the same columns
        aligned_self = aligned_self.reindex(columns=all_integration_ids)
        aligned_other = aligned_other.reindex(columns=all_integration_ids)

        # Concatenate DataFrames
        self.DataFrame = pd.concat([aligned_self, aligned_other], axis=0, ignore_index=True).fillna("")

        # Update integration ID mappings
        self.IntegrationIDToColumnIndex.update(other_table.IntegrationIDToColumnIndex)

    # Complement the DataFrame to ensure completeness of tuples
    def Complement(self):
        U_ou = self.DataFrame.copy()  # Outer unioned tuples
        U_comp = U_ou.copy()
        U_temp = pd.DataFrame(columns=U_comp.columns)

        # Iterate until no changes are made
        while not U_temp.equals(U_comp):
            U_temp = U_comp.copy()
            U_comp = pd.DataFrame(columns=U_comp.columns)

            # Iterate over each tuple in U_temp
            for index1, t_1 in U_temp.iterrows():
                complement_count = 0

                # Iterate over each tuple in U_ou
                for index2, t_2 in U_ou.iterrows():
                    R, complement_status = self.k(t_1, t_2)
                    if complement_status:
                        # Add the new tuple R to U_comp
                        U_comp = pd.concat([U_comp, pd.DataFrame([R])], ignore_index=True)
                        complement_count += 1

                if complement_count == 0:
                    # If no complements were found, add t_1 back to U_comp
                    U_comp = pd.concat([U_comp, pd.DataFrame([t_1])], ignore_index=True)

        # Update the DataFrame with the complemented tuples
        self.DataFrame = U_comp
        print("Complement operation performed.")
        
    # Complement function k(t_1, t_2)
    def k(self, t_1, t_2):
        complement_status = True

        # Check if t_1 and t_2 are complementable
        for col in self.DataFrame.columns:
            val1 = t_1[col]
            val2 = t_2[col]

            if pd.notna(val1) and pd.notna(val2):
                if val1 != val2:
                    # Cannot complement if common attributes differ
                    complement_status = False
                    break

        if complement_status:
            # Create a new tuple R by combining t_1 and t_2
            R = {}
            for col in self.DataFrame.columns:
                val1 = t_1[col]
                val2 = t_2[col]
                if pd.notna(val1):
                    R[col] = val1
                elif pd.notna(val2):
                    R[col] = val2
                else:
                    R[col] = np.nan
            return R, True
        else:
            return None, False


    # Remove tuples that are subsumed by other tuples
    def SubsumeTuples(self):
        original_row_count = len(self.DataFrame)
        # TODO: also drop subsumable tuples, not just duplicates
        self.DataFrame.drop_duplicates(inplace=True)
        new_row_count = len(self.DataFrame)
        print(f"Subsumed tuples: {original_row_count - new_row_count}")
