import unittest
import pandas as pd
from lib.porto.feature_type import get_cat_features_idx

class TestConfig(unittest.TestCase):

    def get_test_df(self):
        return pd.DataFrame({
            "id": [-1, 0, 1],
            "id2_cat": [-1, 2, 3],
            "other": [7, 7, 7]
        })

    def test_config(self):
        df = self.get_test_df()
        idx = get_cat_features_idx(df)
        self.assertListEqual(idx, [1])


if __name__ == '__main__':
    unittest.main()