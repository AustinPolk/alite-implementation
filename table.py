import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import datetime


class RelationalTable:
    def __init__(self):
        self.IntegrationIDToColumnIndex: dict[int, int] = {}
        self.DataFrame: pd.DataFrame = pd.DataFrame()
        self.labeled_null_counter = 0  # Counter to track unique labeled nulls
        self.ColumnEmbeddings: dict[int, np.ndarray] = {}
        self.ColumnNames: dict[int|str, str] = {}
        self.TableName: str = None


    # Save attributes to file (including table)
    def saveToFile(self, prefix=""):
        # Use datetime.datetime.now() to generate the filename with the current date and time
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%d-%m-%Y-%H%M%S")
        
        # Create filenames for the result file and the CSV file, with optional prefix
        result_filename = f"{prefix}Result-{timestamp}.txt"
        csv_filename = f"{prefix}TableData-{timestamp}.csv"

        # Write the metadata and attributes to the result file
        with open(result_filename, 'w', encoding='utf-8') as result_file:
            # Write the table name
            result_file.write(f"Table Name: {self.TableName}\n")
            
            # Write the column names
            result_file.write("Column Names:\n")
            for col_id, col_name in self.ColumnNames.items():
                result_file.write(f"  {col_id}: {col_name}\n")
            
            # Write the IntegrationIDToColumnIndex mapping
            result_file.write("\nIntegrationIDToColumnIndex:\n")
            for integration_id, col_index in self.IntegrationIDToColumnIndex.items():
                result_file.write(f"  {integration_id}: {col_index}\n")
            
            # Write the labeled null counter
            result_file.write(f"\nLabeled Null Counter: {self.labeled_null_counter}\n")
            
            # Write the ColumnEmbeddings (if they exist)
            if self.ColumnEmbeddings:
                result_file.write("\nColumn Embeddings:\n")
                for col_id, embedding in self.ColumnEmbeddings.items():
                    result_file.write(f"  Column {col_id}: {embedding.tolist()}\n")
            else:
                result_file.write("\nColumn Embeddings: None\n")
            
            # Mention the CSV file where the table data is saved
            result_file.write(f"\nTable Data saved to {csv_filename}\n")

        # Write the table data to a CSV file
        if not self.DataFrame.empty:
            self.DataFrame.to_csv(csv_filename, index=False)
            print(f"Table data saved to {csv_filename}")
        else:
            print("The table is empty. No CSV file created.")

        print(f"Metadata and attributes saved to {result_filename}")

    # Load CSV data into the DataFrame
    def LoadFromCSV(self, csv_file: str):
        self.TableName = os.path.basename(csv_file)
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
    def GetColumnNames(self):
        column_names = self.DataFrame.columns.to_list()
        for integrationID, columnIndex in self.IntegrationIDToColumnIndex.items():
            column_name = column_names[columnIndex]
            self.ColumnNames[integrationID] = column_name
    
    # For each column in the table, assign a unique embedding for clustering later
    def InitializeColumnEmbeddings(self, transformer: SentenceTransformer, random_sample: bool = True):
        self.GetColumnNames()

        for integrationID, columnIndex in self.IntegrationIDToColumnIndex.items():
            # read all available entries, generate an embedding by taking the mean of their embeddings
            sum = np.zeros(transformer.get_sentence_embedding_dimension())
            column = self.DataFrame.iloc[:, columnIndex]
            column_values = column.values
            value_count = 0
            
            # if using a random sample, take the first 100 available values as the sample
            if random_sample:
                column_values = sorted(column_values, key = lambda x: 1 if pd.isna(x) else np.random.rand())[:100]

            for value in column_values:
                if pd.isna(value):
                    continue
                else:
                    value_count += 1
                
                # embed the string representation of the value (works for all types)
                str_value = str(value)
                column_annotated = f"{str_value}"
                embedding = transformer.encode(column_annotated, normalize_embeddings=True)
            
                sum += embedding
            
            # take the mean if there were valid values in the column
            if value_count:
                self.ColumnEmbeddings[integrationID] = sum / (value_count)
                continue
            # otherwise just use a random embedding
            else:
                random_embedding = np.random.rand(sum.shape[0]) * 2 - 1
                self.ColumnEmbeddings[integrationID] = (sum + random_embedding) / 2
        
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
        if other_table.DataFrame.empty:
            # The other table is empty, do not modify this table
            return
        elif self.DataFrame.empty:
            # If self is empty, take the other_table
            self.DataFrame = other_table.DataFrame.copy().fillna("")
            self.ColumnNames.update(other_table.ColumnNames)
            return
        elif self.DataFrame.equals(other_table.DataFrame):
            # # If both tables are identical, do a regular union
            self.DataFrame = pd.concat([self.DataFrame, other_table.DataFrame], ignore_index=True).fillna("")
            return
        
        all_columns = list(set(self.DataFrame.columns) | set(other_table.DataFrame.columns))
        matching_columns = list(set(self.DataFrame.columns) & set(other_table.DataFrame.columns))
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
            # Build aligned DataFrames based on column names
            self_rows = []
            for _, row in self.DataFrame.iterrows():
                new_row = {}
                for column in all_columns:
                    if column in self.DataFrame.columns:
                        new_row[column] = row[column]
                    else:
                        new_row[column] = ''
                self_rows.append(new_row)

            other_rows = []
            for _, row in other_table.DataFrame.iterrows():
                new_row = {}
                for column in all_columns:
                    if column in other_table.DataFrame.columns:
                        new_row[column] = row[column]
                    else:
                        new_row[column] = ''
                other_rows.append(new_row)

            # Create DataFrames
            aligned_self = pd.DataFrame(self_rows)
            aligned_other = pd.DataFrame(other_rows)

        # Concatenate the aligned tables
        self.DataFrame = pd.concat([aligned_self, aligned_other], axis=0, ignore_index=True).fillna("")
        
        # alphabetically order the columns by name to create a consistent ordering
        self.DataFrame = self.DataFrame.reindex(sorted(self.DataFrame.columns), axis=1)

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
            print("\n")

            for _, t_1 in U_temp.iterrows():
                complement_count = 0
                for _, t_2 in U_ou.iterrows():
                    if t_1.equals(t_2):
                        continue
                    R, complement_status = self.k(t_1, t_2)
                    if complement_status:
                        U_comp_new = pd.concat([U_comp_new, pd.DataFrame([R])], ignore_index=True)
                        #print("Tuple 1: \n", t_1, "\n")
                        #print("Tuple 2: \n", t_2, "\n")
                        #print("Result: \n", pd.DataFrame([R]), "\n")
                        #print("New tuple: \n", U_comp_new, "\n")
                        complement_count += 1

                if complement_count == 0:
                    U_comp_new = pd.concat([U_comp_new, pd.DataFrame([t_1])], ignore_index=True)

            U_comp_new.drop_duplicates(inplace=True, ignore_index=True)
            U_comp = U_comp_new

        self.DataFrame = U_comp.replace({pd.NA: None})
        print("original tuples: \n", U_ou, "\n")
        print("final tuples: \n", U_comp, "\n")
        print("Complement operation performed.")


    def k(self, t_1, t_2):
        complement_status = True
        R = {}

        for col in self.DataFrame.columns:
            val1 = t_1[col]
            val2 = t_2[col]
            
            print(val1)
            print(val2)

            #is_null1 = pd.isna(val1) or (isinstance(val1, str) and val1.startswith('LN'))
            #is_null2 = pd.isna(val2) or (isinstance(val2, str) and val2.startswith('LN'))
            
            is_null1 = pd.isna(val1) or (isinstance(val1, self.LabeledNull)) or str(val1) == ''
            is_null2 = pd.isna(val2) or (isinstance(val2, self.LabeledNull)) or str(val2) == ''

            if not is_null1 and not is_null2:
                if val1 != val2:
                    complement_status = False
                    break

        if complement_status:
            for col in self.DataFrame.columns:
                val1 = t_1[col]
                val2 = t_2[col]

                #is_null1 = pd.isna(val1) or (isinstance(val1, str) and val1.startswith('LN'))
                #is_null2 = pd.isna(val2) or (isinstance(val2, str) and val2.startswith('LN'))
                
                is_null1 = pd.isna(val1) or (isinstance(val1, self.LabeledNull)) or str(val1) == ''
                is_null2 = pd.isna(val2) or (isinstance(val2, self.LabeledNull)) or str(val2) == ''

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
