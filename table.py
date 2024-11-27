import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class RelationalTable:
    def __init__(self):
        self.IntegrationIDToColumnIndex: dict[int, int] = {}
        self.DataFrame: pd.DataFrame = pd.DataFrame()
        self.labeled_null_counter = 0  # Counter to track unique labeled nulls
        self.ColumnEmbeddings: dict[int, np.ndarray] = {}
        self.ColumnDatatypes: dict[int, str] = {}
        self.ColumnNames: dict[int|str, str] = {}

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
    
    # Record all the datatypes and names for columns in the table
    def GetColumnDatatypesAndNames(self):
        column_names = self.DataFrame.columns.to_list()
        for integrationID, columnIndex in self.IntegrationIDToColumnIndex.items():
            column_name = column_names[columnIndex]
            self.ColumnNames[integrationID] = column_name

            column = self.DataFrame.iloc[:, columnIndex]
            dtype = column.dtype

            if pd.api.types.is_any_real_numeric_dtype(dtype):
                self.ColumnDatatypes[integrationID] = 'real'
            elif pd.api.types.is_integer_dtype(dtype):
                self.ColumnDatatypes[integrationID] = 'integer'
            elif pd.api.types.is_string_dtype(dtype):
                self.ColumnDatatypes[integrationID] = 'string'
            else:
                self.ColumnDatatypes[integrationID] = str(dtype)
    
    # For each column in the table, assign a unique embedding for clustering later
    def InitializeColumnEmbeddings(self, transformer: SentenceTransformer, random_sample: bool = True):
        self.GetColumnDatatypesAndNames()
        
        for integrationID, columnIndex in self.IntegrationIDToColumnIndex.items():
            # begin by embedding the string representation of the column data type
            data_type = self.ColumnDatatypes[integrationID]
            type_embedding = transformer.encode(data_type)
            
            # if this isn't a data type we can either embed or take the average of, assign a random embedding
            if data_type not in ['string', 'integer', 'real']:
                random_embedding = np.random.rand(type_embedding.shape) * 2 - 1  # get a uniform distribution from -1 to 1
                self.ColumnEmbeddings[integrationID] = (type_embedding + random_embedding) / 2
                continue

            # read all available entries, generate an embedding by taking the mean of their embeddings
            sum = type_embedding
            column = self.DataFrame.iloc[:, columnIndex]
            column_values = column.values
            value_count = 0
            
            # if using a random sample, take the first 50 available values as the sample
            if random_sample:
                column_values = sorted(column_values, key = lambda x: 1 if pd.isna(x) else np.random.rand())[:50]

            for value in column_values:
                if pd.isna(value):
                    continue
                else:
                    value_count += 1
                
                # if this is a string, use the transformer embedding
                if data_type == 'string':
                    str_value = str(value)
                    embedding = transformer.encode(str_value)
                # if this is an integer or real number, take the embedding to be a vector filled with that value
                elif data_type == 'integer':
                    int_value = int(value)
                    embedding = np.full(type_embedding.shape, int_value)
                elif data_type == 'real':
                    real_value = float(value)
                    embedding = np.full(type_embedding.shape, real_value)
                
                sum += embedding
            
            # take the mean if there were valid values in the column
            if value_count:
                self.ColumnEmbeddings[integrationID] = sum / (value_count + 1)
                continue
            # otherwise just use a random embedding
            else:
                random_embedding = np.random.rand(type_embedding.shape) * 2 - 1
                self.ColumnEmbeddings[integrationID] = (type_embedding + random_embedding) / 2
                pass
        
    def RenameColumns(self, column_clusters):
        # change the column names to the new Integration ID (i.e. which cluster the column falls into)
        column_name_map = {}
        reverse_map = {}
        for integrationID in self.IntegrationIDToColumnIndex:
            old_name = self.ColumnNames[integrationID]
            new_name = str(column_clusters[integrationID])
            column_name_map[old_name] = new_name
            reverse_map[new_name] = old_name
        self.DataFrame.rename(columns=column_name_map, inplace=True)

        # clear data used to assign the integration IDs that is no longer needed
        self.ColumnEmbeddings.clear()
        self.IntegrationIDToColumnIndex.clear()
        self.ColumnDatatypes.clear()

        # record the original column names that are represented by the current ones
        self.ColumnNames = reverse_map

    class LabeledNull:
        def __init__(self, idx):
            self.idx = idx
        def __hash__(self):
            return self.idx
        def __eq__(self, other):
            return other is self    # labeled nulls cannot be equal unless they are the same

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
        
        #print(self.DataFrame)

        # Update integration ID mappings
        self.IntegrationIDToColumnIndex.update(other_table.IntegrationIDToColumnIndex)


    # Complement the DataFrame to ensure completeness of tuples
    def Complement(self):
        U_ou = self.DataFrame.copy()  # Outer unioned tuples
        U_comp = U_ou.copy()
        U_temp = pd.DataFrame(columns=U_comp.columns)

        i = 0
        # Iterate until no changes are made
        while not U_temp.equals(U_comp):
            print("Iter: ", i)
            i += 1
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
