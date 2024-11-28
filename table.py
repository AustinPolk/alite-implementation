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
        if self.DataFrame.empty and other_table.DataFrame.empty:
            # Both tables are empty; nothing to do
            return
        elif self.DataFrame.empty:
            # If self is empty, take the other_table
            self.DataFrame = other_table.DataFrame.copy()
            self.DataFrame.columns = range(len(self.DataFrame.columns))
            self.IntegrationIDToColumnIndex = other_table.IntegrationIDToColumnIndex.copy()
            return
        elif other_table.DataFrame.empty:
            # If other_table is empty, keep self as it is
            self.DataFrame.columns = range(len(self.DataFrame.columns))
            return
        
        if self.DataFrame.equals(other_table.DataFrame):
            # If both tables are identical, no need to union, return self
            self.DataFrame.columns = range(len(self.DataFrame.columns))
            return
        
        matching_columns = list(set(self.DataFrame.columns).intersection(set(other_table.DataFrame.columns)))
        self_unique_columns = list(set(self.DataFrame.columns) - set(other_table.DataFrame.columns))
        other_unique_columns = list(set(other_table.DataFrame.columns) - set(self.DataFrame.columns))

        if matching_columns:
            # Align on matching columns and keep all other unique columns
            aligned_self = self.DataFrame.copy()
            aligned_other = other_table.DataFrame.copy()

            # For matching columns, ensure they have the same names in both DataFrames
            aligned_self = aligned_self[matching_columns + self_unique_columns]
            aligned_other = aligned_other[matching_columns + other_unique_columns]

            # Fill missing unique columns in each table with empty strings
            for col in other_unique_columns:
                aligned_self[col] = ""
            for col in self_unique_columns:
                aligned_other[col] = ""
        
            # Align DataFrames to have the same columns and order
            aligned_self = aligned_self.reindex(columns=matching_columns + self_unique_columns + other_unique_columns)
            aligned_other = aligned_other.reindex(columns=matching_columns + self_unique_columns + other_unique_columns)
        
        else:
            # Get all unique integration IDs
            all_integration_ids = sorted(set(self.IntegrationIDToColumnIndex.keys()).union(other_table.IntegrationIDToColumnIndex.keys()))

            # Map integration IDs to column names for both tables
            self_id_to_colname = {integration_id: self.DataFrame.columns[col_index] for integration_id, col_index in self.IntegrationIDToColumnIndex.items()}
            other_id_to_colname = {integration_id: other_table.DataFrame.columns[col_index] for integration_id, col_index in other_table.IntegrationIDToColumnIndex.items()}

            # Build aligned DataFrames based on integration IDs
            self_rows = []
            for idx, row in self.DataFrame.iterrows():
                new_row = {}
                for integration_id in all_integration_ids:
                    if integration_id in self.IntegrationIDToColumnIndex:
                        col_name = self_id_to_colname[integration_id]
                        new_row[integration_id] = row[col_name]
                    else:
                        new_row[integration_id] = ''
                self_rows.append(new_row)

            other_rows = []
            for idx, row in other_table.DataFrame.iterrows():
                new_row = {}
                for integration_id in all_integration_ids:
                    if integration_id in other_table.IntegrationIDToColumnIndex:
                        col_name = other_id_to_colname[integration_id]
                        new_row[integration_id] = row[col_name]
                    else:
                        new_row[integration_id] = ''
                other_rows.append(new_row)

            # Create DataFrames
            aligned_self = pd.DataFrame(self_rows)
            aligned_other = pd.DataFrame(other_rows)
        
        #print(aligned_self)
        
        #print(aligned_other)

        # Concatenate DataFrames
        self.DataFrame = pd.concat([aligned_self, aligned_other], axis=0, ignore_index=True).fillna("")
        
        if matching_columns:
            self.DataFrame.columns = range(len(self.DataFrame.columns))
        
        #print(self.DataFrame)

        # Update integration ID mappings
        self.IntegrationIDToColumnIndex.update(other_table.IntegrationIDToColumnIndex)


    def Complement(self):
        U_ou = self.DataFrame.copy()  # Outer unioned tuples
        U_comp = U_ou.copy()
        U_temp = pd.DataFrame(columns=U_comp.columns)

        i = 0
        while not U_temp.equals(U_comp):
            print(f"Iter: {i}")
            i += 1
            U_temp = U_comp.copy()
            U_comp_new = pd.DataFrame(columns=U_comp.columns)

            for _, t_1 in U_temp.iterrows():
                complement_count = 0
                for _, t_2 in U_ou.iterrows():
                    R, complement_status = self.k(t_1, t_2)
                    if complement_status:
                        U_comp_new = pd.concat([U_comp_new, pd.DataFrame([R])], ignore_index=True)
                        complement_count += 1

                if complement_count == 0:
                    U_comp_new = pd.concat([U_comp_new, pd.DataFrame([t_1])], ignore_index=True)

            U_comp_new.drop_duplicates(inplace=True, ignore_index=True)
            U_comp = U_comp_new

        self.DataFrame = U_comp.replace({pd.NA: None})
        print("Complement operation performed.")


    def k(self, t_1, t_2):
        complement_status = True
        R = {}

        for col in self.DataFrame.columns:
            val1 = t_1[col]
            val2 = t_2[col]

            is_null1 = pd.isna(val1) or (isinstance(val1, str) and val1.startswith('LN'))
            is_null2 = pd.isna(val2) or (isinstance(val2, str) and val2.startswith('LN'))

            if not is_null1 and not is_null2:
                if val1 != val2:
                    complement_status = False
                    break

        if complement_status:
            for col in self.DataFrame.columns:
                val1 = t_1[col]
                val2 = t_2[col]

                is_null1 = pd.isna(val1) or (isinstance(val1, str) and val1.startswith('LN'))
                is_null2 = pd.isna(val2) or (isinstance(val2, str) and val2.startswith('LN'))

                if not is_null1:
                    R[col] = val1
                elif not is_null2:
                    R[col] = val2
                else:
                    R[col] = pd.NA
            return R, True
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
