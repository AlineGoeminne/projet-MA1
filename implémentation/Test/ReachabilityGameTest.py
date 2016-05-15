import unittest

from ReachabilityGame import *
from DijkstraMinMax import *

class TestAtteignability(unittest.TestCase):

    def test_nash_equilibrium(self):

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 2)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 1)
        vertex = [v0, v1, v2, v3, v4]

        pred0 = [(1, 1), (3, 1)]
        pred1 = [(0, 1)]
        pred2 = [(1, 1), (4, 2)]
        pred3 = [(2, 1), (4, 1)]
        pred4 = [(2, 4), (3, 1)]

        list_pred = [pred0, pred1, pred2, pred3, pred4]
        list_succ = Graph.list_pred_to_list_succ(list_pred)

        graph = Graph(vertex, None, list_pred, list_succ)
        goals = [set([3]), set([0])]
        init = 1

        game = ReachabilityGame(2, graph, init, goals, None)

        path1 = [v1, v2, v3, v4, v3, v4, v3, v4]  # EN ou J1 voit son objectif
        path2 = [v1, v2, v3, v0, v1, v0, v1, v0, v1]  # En ou les deux joueurs voient leur objectif
        path3 = [v1, v2, v4, v2, v4, v2, v4, v2, v4, v2, v4]  # pas un EN



        self.assertTrue(game.is_a_Nash_equilibrium(path1))
        self.assertTrue(game.is_a_Nash_equilibrium(path2))
        self.assertFalse(game.is_a_Nash_equilibrium(path3))

        cost_path1 = game.cost_for_all_players(path1)

        self.assertEqual(2, cost_path1[0])
        self.assertEqual(float("infinity"), cost_path1[1])

        cost_path2 = game.cost_for_all_players(path2)

        self.assertEqual(2, cost_path2[0])
        self.assertEqual(3, cost_path2[1])

        cost_path3 = game.cost_for_all_players(path3)

        self.assertEqual(float("infinity"), cost_path3[0])
        self.assertEqual(float("infinity"), cost_path3[1])


