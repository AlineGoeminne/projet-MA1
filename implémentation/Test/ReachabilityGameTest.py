import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest

from GraphGame import *

class TestReachabilityGame(unittest.TestCase):

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

        graph = Graph(vertex, None, list_pred, list_succ, 4)
        goals = [{3}, {0}]
        init = v1

        game = ReachabilityGame(2, graph, init, goals, None)

        path1 = [v1, v2, v3, v4, v3, v4, v3, v4]  # EN ou J1 voit son objectif
        path2 = [v1, v2, v3, v0, v1, v0, v1, v0, v1]  # En ou les deux joueurs voient leur objectif
        path3 = [v1, v2, v4, v2, v4, v2, v4, v2, v4, v2, v4]  # pas un EN


        (nash1, coal1) = game.is_a_Nash_equilibrium(path1)
        (nash2, coal2) = game.is_a_Nash_equilibrium(path2)
        (nash3, coal3) = game.is_a_Nash_equilibrium(path3)

        self.assertTrue(nash1)
        self.assertTrue(nash2)
        self.assertFalse(nash3)

        cost_path1 = game.cost_for_all_players(path1)

        self.assertEqual(2, cost_path1[1])
        self.assertEqual(float("infinity"), cost_path1[2])

        cost_path2 = game.cost_for_all_players(path2)

        self.assertEqual(2, cost_path2[1])
        self.assertEqual(3, cost_path2[2])

        cost_path3 = game.cost_for_all_players(path3)

        self.assertEqual(float("infinity"), cost_path3[1])
        self.assertEqual(float("infinity"), cost_path3[2])

        self.assertTrue(game.is_a_Nash_equilibrium(path2, {2}))



    def test_same_path(self):

        v0 = Vertex(0,1)
        v1 = Vertex(1,2)
        v2 = Vertex(2,1)
        path1 = [v0, v2, v1, v0]
        path2 = [v0, v2, v1]
        path3 = [v0, v2, v1, v1]
        path4 = [v0, v2, v1, v0]

        self.assertTrue(ReachabilityGame.same_paths(path1,path1))
        self.assertTrue(ReachabilityGame.same_paths(path1,path4))
        self.assertFalse(ReachabilityGame.same_paths(path1,path2))
        self.assertFalse(ReachabilityGame.same_paths(path1,path3))

    def test_path_vertex_to_path_index(self):
        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 1)

        path = [v0, v2, v1, v0]

        res = ReachabilityGame.path_vertex_to_path_index(path)

        self.assertEqual([0, 2, 1, 0], res)


    def test_generate_random_path(self):

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

        graph = Graph(vertex, None, list_pred, list_succ, 4)
        goals = [{3}, {0}]
        init = v1

        game = ReachabilityGame(2, graph, init, goals, None)

        en_set = game.test_random_path(50, 10)

        self.assertTrue(len(en_set) <= 50)
        self.assertEqual(10, len(en_set[0]))


    def test_path_cost_one_player(self):
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

        graph = Graph(vertex, None, list_pred, list_succ, 4)
        goals = [set([3]), set([0])]
        init = v1

        game = ReachabilityGame(2, graph, init, goals, None)

        path1 = [v1, v2, v3, v4, v3, v4, v3, v4] #En ou le joueur 1 atteint son objectif
        path2 = [v1, v2, v3, v0, v1, v0, v1, v0, v1]  # En ou les deux joueurs voient leur objectif
        path3 = [v1, v2, v4, v2, v4, v2, v4, v2, v4, v2, v4]  # pas un EN

        self.assertEqual(2, game.cost_for_one_player(path1, 1))
        self.assertEqual(float("infinity"), game.cost_for_one_player(path1, 2))

        self.assertEqual(2, game.cost_for_one_player(path2, 1))
        self.assertEqual(3, game.cost_for_one_player(path2, 2))

        self.assertEqual(float("infinity"), game.cost_for_one_player(path3, 1))
        self.assertEqual(float("infinity"), game.cost_for_one_player(path3, 2))


    def test_is_Nash_equilibrium_one_player(self):
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

        graph = Graph(vertex, None, list_pred, list_succ, 4)
        goals = [set([3]), set([0])]
        init = v1

        game = ReachabilityGame(2, graph, init, goals, None)

        path1 = [v1, v2, v3, v4, v3, v4, v3, v4]  # En ou le joueur 1 atteint son objectif
        path2 = [v1, v2, v3, v0, v1, v0, v1, v0, v1]  # En ou les deux joueurs voient leur objectif
        path3 = [v1, v2, v4, v2, v4, v2, v4, v2, v4, v2, v4]  # pas un EN

        (nash11, coalitions1) = game.is_a_Nash_equilibrium_one_player(path1, 1)
        self.assertTrue(nash11)
        (nash12, coalitions1) = game.is_a_Nash_equilibrium_one_player(path1, 2, coalitions1)
        self.assertTrue(nash12)

        (nash21, coalitions2) = game.is_a_Nash_equilibrium_one_player(path2, 1)
        (nash22, coalitions2) = game.is_a_Nash_equilibrium_one_player(path2, 2, coalitions2)
        self.assertTrue(nash21)
        self.assertTrue(nash22)

        (nash31, coalitions3) = game.is_a_Nash_equilibrium_one_player(path3, 1)
        (nash32, coalitions3) = game.is_a_Nash_equilibrium_one_player(path3, 2, coalitions3)

        self.assertFalse(nash31)
        self.assertTrue(nash32)

    def test_best_first_search_2(self):
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

        graph = Graph(vertex, None, list_pred, list_succ, 4)
        goals = [{3}, {0}]
        init = v1
        game = ReachabilityGame(2, graph, init, goals, None)

        a_star = game.best_first_search(ReachabilityGame.a_star_positive, None, 5)

        self.assertEqual(a_star, [v1, v2, v3, v0])


    def test_negative_slides(self):
        v0 = Vertex(0, 2)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 2)
        v4 = Vertex(4, 1)
        v5 = Vertex(5, 2)
        v6 = Vertex(6, 1)

        all_vertices = [v0, v1, v2, v3, v4, v5, v6]

        succ0 = [(1, (1, 1)), (3, (1, 1))]
        succ1 = [(2, (-1, -1))]
        succ2 = [(1, (4, 4)), (3, (2, 2)), (4, (1, 1))]
        succ3 = [(4, (1, 1)), (6, (1, 1))]
        succ4 = [(5, (1, 1)), (6, (-1, -1))]
        succ5 = [(3, (3, 3)), (5, (1, 1))]
        succ6 = [(0, (2, 2)), (5, (1, 1))]

        succ = [succ0, succ1, succ2, succ3, succ4, succ5, succ6]

        mat = Graph.list_succ_to_mat(succ, True, 2)
        pred = Graph.matrix_to_list_pred(mat)

        W = (4, 4)
        graph = Graph(all_vertices, mat, pred, succ, W)

        target1 = {6}
        target2 = {5}

        goals = [target1, target2]

        init = v0

        game = ReachabilityGame(2,graph,init,goals,None)

        a_star = game.best_first_search(ReachabilityGame.a_star_negative,None,30,True, True, None, None)

        en = [v0,v3,v6,v5]

        print en, "est un EN ", game.is_a_Nash_equilibrium(en, None,None,True,True)[0], " info ", game.get_info_path(en)

        print "A_star", a_star, "info ", game.get_info_path(a_star)



if __name__ == '__main__':

    unittest.main()