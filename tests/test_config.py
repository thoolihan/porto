import unittest
from lib.config import get_config

class TestConfig(unittest.TestCase):

    def test_config(self):
        cfg = get_config()
        self.assertEqual(cfg["name"], 'Porto Seguro Kaggle')

if __name__ == '__main__':
    unittest.main()