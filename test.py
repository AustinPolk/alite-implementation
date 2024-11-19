import unittest
import pandas as pd
from table import RelationalTable


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
        
        #print()

        #print(table_a.DataFrame)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [1, 2, 3, 4],
            1: ['A', 'B', '', ''],
            2: ['', '', 'C', 'D']
        })
        
        #print("Expected")
        
        #print(expected_df)

        #pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

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

        #pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

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
        
        print()
        print(table_a.DataFrame)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [1, 2, '', ''],
            1: ['X', 'Y', '', ''],
            2: ['', '', 1, 2],
            3: ['', '', 'X', 'Y']
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

        # Expected DataFrame
        expected_df = pd.DataFrame({
            0: [5, 6],
            1: ['M', 'N']
        })

        #pd.testing.assert_frame_equal(table_a.DataFrame.reset_index(drop=True), expected_df)

    def test_complement_no_missing_values(self):
        # Create Table
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['X', 'Y'],
            'Col2': [1, 2]
        })
        table.GenerateLabeledNulls()
        table.Complement()
        table.ReplaceLabeledNulls()

        # Expected DataFrame (should be unchanged)
        expected_df = pd.DataFrame({
            'Col1': ['X', 'Y'],
            'Col2': [1, 2]
        })

        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df)

    def test_complement_non_complementable_tuples(self):
        # Create Table with conflicting values
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', None],
            'Col2': [1, 2]
        })
        table.GenerateLabeledNulls()
        table.Complement()
        table.ReplaceLabeledNulls()

        # Expected DataFrame (no tuples can be complemented)
        expected_df = pd.DataFrame({
            'Col1': ['A', None],
            'Col2': [1, 2]
        })

        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df)

    def test_complement_multiple_missing_values(self):
        # Create Table
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', None, 'A'],
            'Col2': [None, 2, 2],
            'Col3': [3, 3, None]
        })
        table.GenerateLabeledNulls()
        table.Complement()
        table.ReplaceLabeledNulls()
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

        print(table)
        table.GenerateLabeledNulls()
        print(table)
        table.Complement()
        print(table)
        table.ReplaceLabeledNulls()

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
        table.GenerateLabeledNulls()
        table.Complement()
        table.ReplaceLabeledNulls()
        table.SubsumeTuples()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'B'],
            'Col2': [1, 2]
        })

        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df, check_dtype=False)

    def test_complement_basic(self):
        # Create Table
        table = RelationalTable()
        table.DataFrame = pd.DataFrame({
            'Col1': ['A', None, 'A'],
            'Col2': [None, 1, 1]
        })
        table.GenerateLabeledNulls()
        table.Complement()
        table.ReplaceLabeledNulls()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'Col1': ['A', 'A', None],
            'Col2': [None, 1, 1]
        })

        pd.testing.assert_frame_equal(table.DataFrame.reset_index(drop=True), expected_df)

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
