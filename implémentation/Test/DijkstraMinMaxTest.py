import unittest

from DijkstraMinMax import convertPred2NbrSucc


class TestDijkstraMinMax(unittest.TestCase):

    def test_convertPred2NbrSuccTest(self):

        v0_pred = [(1, 42), (2, 42), (3, 42)]
        v1_pred = [(2, 42)]
        v2_pred = [(1, 42), (2, 42), (0, 42)]

        pred = [v0_pred, v1_pred, v2_pred, None]

        nbrSucc = convertPred2NbrSucc(pred)

        self.assertEqual(nbrSucc[3], 1)
        self.assertEqual(nbrSucc[0], 1)
        self.assertEqual(nbrSucc[1], 2)
        self.assertEqual(nbrSucc[2], 3)