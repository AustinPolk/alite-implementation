import unittest
import pandas as pd
from table import RelationalTable
import numpy as np


class TestRelationalTableFunctions(unittest.TestCase):
    def test_outer_union_with_basic(self):
        # Create Table A
        table_a = RelationalTable()
        table_a.DataFrame = pd.DataFrame({
            'A1': [1, 2],
            'A2': ['x', 'y']
        })
        table_a.InitializeIntegrationIDs(0)

        # Create Table B
        table_b = RelationalTable()
        table_b.DataFrame = pd.DataFrame({
            'B1': [3, 4],
            'B2': ['u', 'v']
        })
        table_b.InitializeIntegrationIDs(2)

        # Perform Outer Union
        table_a.OuterUnionWith(table_b)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [1, 2, '', ''],
            1: ['x', 'y', '', ''],
            2: ['', '', 3, 4],
            3: ['', '', 'u', 'v']
        })

        pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

    def test_outer_union_with_different_row_counts(self):
        # Create Table A with 2 rows
        table_a = RelationalTable()
        table_a.DataFrame = pd.DataFrame({
            'Col1': [1, 2],
            'Col2': ['A', 'B']
        })
        table_a.InitializeIntegrationIDs(0)

        # Create Table B with 3 rows
        table_b = RelationalTable()
        table_b.DataFrame = pd.DataFrame({
            'Col3': [3, 4, 5],
            'Col4': ['C', 'D', 'E']
        })
        table_b.InitializeIntegrationIDs(len(table_a.DataFrame.columns))

        # Perform Outer Union
        table_a.OuterUnionWith(table_b)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [1, 2, '', '', ''],
            1: ['A', 'B', '', '', ''],
            2: ['', '', 3, 4, 5],
            3: ['', '', 'C', 'D', 'E']
        })

        expected_df = expected_df.reset_index(drop=True)

        pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

    def test_outer_union_with_overlapping_columns(self):
        # Create Table A
        table_a = RelationalTable()
        table_a.DataFrame = pd.DataFrame({
            'ID': [1, 2],
            'Value': ['A', 'B']
        })
        table_a.InitializeIntegrationIDs(0)

        # Create Table B
        table_b = RelationalTable()
        table_b.DataFrame = pd.DataFrame({
            'ID': [3, 4],
            'Description': ['C', 'D']
        })
        table_b.InitializeIntegrationIDs(len(table_a.DataFrame.columns))
        
        # Perform Outer Union
        table_a.OuterUnionWith(table_b)
        
        #print("\nActual")

        #print(table_a.DataFrame)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [1, 2, 3, 4],
            1: ['A', 'B', '', ''],
            2: ['', '', 'C', 'D']
        })
        
        #print("Expected")
        
        #print(expected_df)

        pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

    def test_outer_union_with_different_schemas(self):
        # Create Table A
        table_a = RelationalTable()
        table_a.DataFrame = pd.DataFrame({
            'A1': [10, 20],
            'A2': ['foo', 'bar']
        })
        table_a.InitializeIntegrationIDs(0)

        # Create Table B
        table_b = RelationalTable()
        table_b.DataFrame = pd.DataFrame({
            'B1': [30, 40],
            'B2': ['baz', 'qux']
        })
        table_b.InitializeIntegrationIDs(len(table_a.DataFrame.columns))

        # Perform Outer Union
        table_a.OuterUnionWith(table_b)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [10, 20, '', ''],
            1: ['foo', 'bar', '', ''],
            2: ['', '', 30, 40],
            3: ['', '', 'baz', 'qux']
        })

        pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

    def test_outer_union_with_identical_tables(self):
        # Create Table A and B with the same data
        data = {
            'Key': [1, 2],
            'Value': ['X', 'Y']
        }

        table_a = RelationalTable()
        table_a.DataFrame = pd.DataFrame(data)
        table_a.InitializeIntegrationIDs(0)

        table_b = RelationalTable()
        table_b.DataFrame = pd.DataFrame(data)
        table_b.InitializeIntegrationIDs(len(table_a.DataFrame.columns))

        # Perform Outer Union
        table_a.OuterUnionWith(table_b)
        
        print("\nIdentical Tables")
        
        print("Actual")

        print(table_a.DataFrame)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [1, 2],
            1: ['X', 'Y']
        })
        
        print("Expected")
        print(expected_df)

        pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

    def test_outer_union_with_empty_table(self):
        # Create Table A with data
        table_a = RelationalTable()
        table_a.DataFrame = pd.DataFrame({
            'A1': [5, 6],
            'A2': ['M', 'N']
        })
        table_a.InitializeIntegrationIDs(0)

        # Create Empty Table B
        table_b = RelationalTable()
        table_b.DataFrame = pd.DataFrame(columns=['B1', 'B2'])
        table_b.InitializeIntegrationIDs(len(table_a.DataFrame.columns))

        # Perform Outer Union
        table_a.OuterUnionWith(table_b)
        
        print("\nEmpty table")
        
        print("Actual")

        print(table_a.DataFrame)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [5, 6],
            1: ['M', 'N']
        })
        
        print("Expected")
        print(expected_df)

        pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

    def test_complement_identical_tables(self):
        # Identical tables should result in no change after complement
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', 'B', 'C'],
            'Col2': [1, 2, 3]
        })
        #table.GenerateLabeledNulls()
        table.Complement()
        #table.ReplaceLabeledNulls()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'B', 'C'],
            'Col2': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df, check_dtype=False)


    def test_complement_disjoint_schemas(self):
        # Complement does not apply when schemas are disjoint
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', 'B'],
            'Col2': [1, None]
        })
        table.Complement()

        expected_df = pd.DataFrame({
            'Col1': ['A', 'B'],
            'Col2': [1, None]
        })
        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df)



    # def test_complement_partial_overlap(self):
    #     # Prepare test data
    #     data = {'Col1': ['A', 'A', None, 'B'], 'Col2': [1, 3, 3, 2]}
    #     table = RelationalTable()
    #     table.DataFrame = pd.DataFrame(data)

    #     # Run Complement
    #     table.Complement()

    #     # Expected DataFrame
    #     expected_data = {'Col1': ['A', 'B', None, 'A'], 'Col2': [1, 2, 3, 3]}
    #     expected_df = pd.DataFrame(expected_data).reset_index(drop=True)

    #     # Align indices
    #     actual_df = table.DataFrame.reset_index(drop=True)

    #     # Compare
    #     print("Actual DataFrame (Partial Overlap):\n", actual_df)
    #     print("Expected DataFrame (Partial Overlap):\n", expected_df)
    #     pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)


    # def test_complement_with_partial_data(self):
    #     # Prepare test data
    #     data = {'Col1': ['A', 'B', 'D', None], 'Col2': [None, 2, 3, 3], 'Col3': [1, None, 4, None]}
    #     table = RelationalTable()
    #     table.DataFrame = pd.DataFrame(data)

    #     # Run Complement
    #     table.Complement()

    #     # Expected DataFrame
    #     expected_data = {
    #         'Col1': ['A', 'B', 'D', None, 'D'],
    #         'Col2': [None, 2, 3, 3, None],
    #         'Col3': [1, None, 4, None, 4]
    #     }
    #     expected_df = pd.DataFrame(expected_data).reset_index(drop=True)

    #     # Align indices
    #     actual_df = table.DataFrame.reset_index(drop=True)

    #     # Compare
    #     print("Actual DataFrame (With Partial Data):\n", actual_df)
    #     print("Expected DataFrame (With Partial Data):\n", expected_df)
    #     pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)
    

    def test_complement_with_redundant_data(self):
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', 'B', 'A'],
            'Col2': [None, 2, None],
            'Col3': [1, None, 1]
        })

        table.Complement()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'B'],
            'Col2': [None, 2],
            'Col3': [1, None]
        })

        # Print actual and expected for debugging
        print("Actual DataFrame (Redundant Data):")
        print(table.DataFrame.reset_index(drop=True))
        print("Expected DataFrame (Redundant Data):")
        print(expected_df.reset_index(drop=True))

        pd.testing.assert_frame_equal(
            table.DataFrame.reset_index(drop=True),
            expected_df,
            check_dtype=False
        )





    def test_complement_with_all_nulls(self):
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': [None, None],
            'Col2': [None, None]
        })

        table.Complement()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': [None],
            'Col2': [None]
        })

        # Debug print for actual and expected DataFrames
        print("\nTest: Complement With All Nulls")
        print("Actual DataFrame:")
        print(table.DataFrame.reset_index(drop=True))
        print("Expected DataFrame:")
        print(expected_df)

        # Ensure dtypes match
        actual_df = table.DataFrame.reset_index(drop=True)
        expected_df = expected_df.reset_index(drop=True)
        for col in expected_df.columns:
            expected_df[col] = expected_df[col].astype(actual_df[col].dtype)

        # Assert equality
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)





    def test_complement_no_missing_values(self):
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', 'B', 'C'],
            'Col2': [1, 2, 3]
        })

        table.Complement()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'B', 'C'],
            'Col2': [1, 2, 3]
        })

        # Print actual and expected for debugging
        print("Actual DataFrame (No Missing Values):")
        print(table.DataFrame.reset_index(drop=True))
        print("Expected DataFrame (No Missing Values):")
        print(expected_df.reset_index(drop=True))

        pd.testing.assert_frame_equal(
            table.DataFrame.reset_index(drop=True),
            expected_df,
            check_dtype=False
        )


    def test_complement_non_complementable_tuples(self):
        # Create Table with conflicting values
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', None],
            'Col2': [1, 2]
        })
        #table.GenerateLabeledNulls()
        table.Complement()
        #table.ReplaceLabeledNulls()

        # Expected DataFrame (no tuples can be complemented)
        expected_df = pd.DataFrame({
            'Col1': ['A', None],
            'Col2': [1, 2]
        })

        actual_df = table.DataFrame.reset_index(drop=True)
        expected_df = expected_df.reset_index(drop=True)

        # Ensure dtypes match
        for col in expected_df.columns:
            expected_df[col] = expected_df[col].astype(actual_df[col].dtype)

        # Assert equality
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_complement_multiple_missing_values(self):
        # Create Table
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', None, 'A'],
            'Col2': [None, 2, 2],
            'Col3': [3, 3, None]
        })
        #table.GenerateLabeledNulls()
        table.Complement()
        #table.ReplaceLabeledNulls()
        table.SubsumeTuples()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A'],
            'Col2': [2],
            'Col3': [3]
        })

        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df, check_dtype=False)

    def test_complement_all_missing_values(self):
        # Create Table with all missing values
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': [None, None],
            'Col2': [None, None]
        })

        # print(table)
        # table.GenerateLabeledNulls()
        # print(table)
        table.Complement()
        # print(table)
        # table.ReplaceLabeledNulls()

        print(table)

        # Expected DataFrame (should contain labeled nulls)
        expected_df = pd.DataFrame({
            'Col1': [None, None],
            'Col2': [None, None]
        })

        #pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df)

    def test_complement_mixed_tuples(self):
        # Create Table
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', None, 'B', 'A'],
            'Col2': [1, 1, 2, None]
        })
        #table.GenerateLabeledNulls()
        table.Complement()
        #table.ReplaceLabeledNulls()
        table.SubsumeTuples()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'B'],
            'Col2': [1, 2]
        })

        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df, check_dtype=False)

    def test_complement_basic(self):
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', None, 'A'],
            'Col2': [None, 1, 1]
        })

        table.Complement()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'A', None],
            'Col2': [None, 1, 1]
        })

        # Debug print for actual and expected DataFrames
        print("\nTest: Complement Basic")
        print("Actual DataFrame:")
        print(table.DataFrame.reset_index(drop=True))
        print("Expected DataFrame:")
        print(expected_df)

        # Ensure dtypes match
        actual_df = table.DataFrame.reset_index(drop=True)
        expected_df = expected_df.reset_index(drop=True)
        for col in expected_df.columns:
            expected_df[col] = expected_df[col].astype(actual_df[col].dtype)

        # Assert equality
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)

    def test_subsume_tuples_basic(self):
        # Create Table
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', 'A', 'B'],
            'Col2': [1, 1, 2]
        })
        table.SubsumeTuples()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'B'],
            'Col2': [1, 2]
        })

        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df)


if __name__ == '__main__':
    unittest.main()
