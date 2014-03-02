__author__ = 'elubin'

import unittest
from wright_fisher import WrightFisher

class TestCase(unittest.TestCase):
    def test_likelihood(self):
        w = WrightFisher()

if __name__ == '__main__':
    unittest.main()