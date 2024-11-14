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

        # Create a mapping from column names to integration IDs for both tables
        self_col_to_id = {self.DataFrame.columns[idx]: integration_id for integration_id, idx in self.IntegrationIDToColumnIndex.items()}
        other_col_to_id = {other_table.DataFrame.columns[idx]: integration_id for integration_id, idx in other_table.IntegrationIDToColumnIndex.items()}

        # Rename columns to integration IDs
        self.DataFrame.rename(columns=self_col_to_id, inplace=True)
        other_table.DataFrame.rename(columns=other_col_to_id, inplace=True)
        
        # After renaming, check for duplicate columns
        if self.DataFrame.columns.duplicated().any():
            print("Duplicates in self.DataFrame columns after renaming")
            print(self.DataFrame.columns[self.DataFrame.columns.duplicated()])
        if other_table.DataFrame.columns.duplicated().any():
            print("Duplicates in other_table.DataFrame columns after renaming")
            print(other_table.DataFrame.columns[other_table.DataFrame.columns.duplicated()])


        # Reindex DataFrames to have all integration IDs as columns
        aligned_self = self.DataFrame.reindex(columns=all_integration_ids)
        aligned_other = other_table.DataFrame.reindex(columns=all_integration_ids)

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
            U_comp_new = pd.DataFrame(columns=U_comp.columns)

            # Iterate over each tuple in U_temp
            for _, t_1 in U_temp.iterrows():
                complement_count = 0

                # Iterate over each tuple in U_ou
                for _, t_2 in U_ou.iterrows():
                    R, complement_status = self.k(t_1, t_2)
                    if complement_status:
                        # Add the new tuple R to U_comp_new
                        U_comp_new = pd.concat([U_comp_new, pd.DataFrame([R])], ignore_index=True)
                        complement_count += 1

                if complement_count == 0:
                    # If no complements were found, add t_1 back to U_comp_new
                    U_comp_new = pd.concat([U_comp_new, pd.DataFrame([t_1])], ignore_index=True)

            # Remove duplicates to prevent extra tuples
            U_comp_new.drop_duplicates(inplace=True, ignore_index=True)
            U_comp = U_comp_new

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

            # Treat labeled nulls as nulls
            val1_is_null = pd.isna(val1) or (isinstance(val1, str) and val1.startswith('LN'))
            val2_is_null = pd.isna(val2) or (isinstance(val2, str) and val2.startswith('LN'))

            if not val1_is_null and not val2_is_null:
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

                val1_is_null = pd.isna(val1) or (isinstance(val1, str) and val1.startswith('LN'))
                val2_is_null = pd.isna(val2) or (isinstance(val2, str) and val2.startswith('LN'))

                if not val1_is_null:
                    R[col] = val1
                elif not val2_is_null:
                    R[col] = val2
                else:
                    R[col] = np.nan
            return R, True
        else:
            return None, False
    
    # Remove tuples that are subsumed by other tuples
    def SubsumeTuples(self):
        original_row_count = len(self.DataFrame)
        df = self.DataFrame.reset_index(drop=True)
        is_subsumed = [False] * len(df)
        
        # Iterate over each tuple t1
        for i in range(len(df)):
            if is_subsumed[i]:
                continue  # Skip if already subsumed
            t1 = df.loc[i]
            # Compare with every other tuple t2
            for j in range(len(df)):
                if i == j or is_subsumed[j]:
                    continue  # Skip same tuple or already subsumed
                t2 = df.loc[j]
                if self.is_subsumed(t1, t2):
                    # t2 is subsumed by t1
                    is_subsumed[j] = True
                elif self.is_subsumed(t2, t1):
                    # t1 is subsumed by t2
                    is_subsumed[i] = True
                    break  # No need to compare t1 further
        # Remove subsumed tuples
        self.DataFrame = df[~pd.Series(is_subsumed)].reset_index(drop=True)
        new_row_count = len(self.DataFrame)
        print(f"Subsumed tuples: {original_row_count - new_row_count}")

    # Helper function to check if t1 subsumes t2
    def is_subsumed(self, t1, t2):
        for col in self.DataFrame.columns:
            val1 = t1[col]
            val2 = t2[col]
            if pd.notna(val2):
                if pd.isna(val1) or val1 != val2:
                    return False
        return True
