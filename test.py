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
