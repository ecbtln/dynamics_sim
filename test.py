__author__ = 'elubin'

import unittest
from games import HawkDove
import numpy as np


class TestCase(unittest.TestCase):
    def setUp(self):
        v = 30
        c = 60
        self.hd_params = dict(a=(v-c) / 2.0, b=v, c=0, d=v / 2.0)
        self.game = HawkDove(v=30, c=60)

    def test_get_payoff(self):
        self.assertEqual(self.game.pm.get_payoff(0, *[0, 0]), self.hd_params['a'])
        self.assertEqual(self.game.pm.get_payoff(1, *[0, 0]), self.hd_params['a'])
        self.assertEqual(self.game.pm.get_payoff(0, *[1, 1]), self.hd_params['d'])
        self.assertEqual(self.game.pm.get_payoff(1, *[1, 1]), self.hd_params['d'])
        self.assertEqual(self.game.pm.get_payoff(0, *[0, 1]), self.hd_params['b'])
        self.assertEqual(self.game.pm.get_payoff(1, *[0, 1]), self.hd_params['c'])
        self.assertEqual(self.game.pm.get_payoff(0, *[1, 0]), self.hd_params['c'])
        self.assertEqual(self.game.pm.get_payoff(1, *[1, 0]), self.hd_params['b'])

    def test_get_expected_payoff_1(self):
        state = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        self.assertEqual(self.game.pm.get_expected_payoff(0, 0, state), self.hd_params['a'])
        self.assertEqual(self.game.pm.get_expected_payoff(0, 1, state), self.hd_params['c'])
        self.assertEqual(self.game.pm.get_expected_payoff(1, 0, state), self.hd_params['a'])
        self.assertEqual(self.game.pm.get_expected_payoff(1, 1, state), self.hd_params['c'])
        state = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
        self.assertEqual(self.game.pm.get_expected_payoff(0, 0, state), self.hd_params['a'])
        self.assertEqual(self.game.pm.get_expected_payoff(0, 1, state), self.hd_params['c'])
        self.assertEqual(self.game.pm.get_expected_payoff(1, 0, state), self.hd_params['b'])
        self.assertEqual(self.game.pm.get_expected_payoff(1, 1, state), self.hd_params['d'])

    def test_get_expected_payoff_2(self):
        state = [np.array([0.75, 0.25]), np.array([0.1, 0.9])]
        a = self.hd_params['a']
        b = self.hd_params['b']
        c = self.hd_params['c']
        d = self.hd_params['d']
        self.assertEqual(self.game.pm.get_expected_payoff(0, 0, state), 0.1 * a + 0.9 * b)
        self.assertEqual(self.game.pm.get_expected_payoff(0, 1, state), 0.1 * c + 0.9 * d)
        self.assertEqual(self.game.pm.get_expected_payoff(1, 0, state), 0.75 * a + 0.25 * b)
        self.assertEqual(self.game.pm.get_expected_payoff(1, 1, state), 0.75 * c + 0.25 * d)

# TODO: test 3 or more players, 3 or more strategies (HDB)

if __name__ == '__main__':
    unittest.main()