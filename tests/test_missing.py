import unittest
import pandas as pd
from lib.data import make_missing_zero

class TestConfig(unittest.TestCase):

    def get_test_df(self):
        return pd.DataFrame({
            "id": [-1, 0, 1],
            "id2": [-1, 2, 3],
            "other": [7, 7, 7]
        })

    def test_missing(self):
        df = self.get_test_df()
        df = make_missing_zero(df, [0, 1])
        for val in df.iloc[:,0]:
            self.assertGreaterEqual(val, 0)
        for val in df.iloc[:, 1]:
            self.assertGreaterEqual(val, 0)

if __name__ == '__main__':
    unittest.main()