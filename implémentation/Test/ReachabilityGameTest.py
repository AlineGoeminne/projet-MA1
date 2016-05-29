import unittest

from GraphGame import *
from DijkstraMinMax import *

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

        graph = Graph(vertex, None, list_pred, list_succ)
        goals = [{3}, {0}]
        init = v1

        game = ReachabilityGame(2, graph, init, goals, None, None)

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


    def test_is_nash_equilibrium2(self):

        mat = [[0,8,8,8,4], [9,0,2,8,1], [3,3,0,10,2], [9,9,5,0,10],[5,2,6,5,0]]

        pred = Graph.matrix_to_list_pred(mat)
        succ = Graph.list_pred_to_list_succ(pred)

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 2)
        v3 = Vertex(3, 2)
        v4 = Vertex(4, 2)

        vertex = [v0, v1, v2, v3, v4]

        graph = Graph(vertex, mat, pred, succ, 10)

        obj = [set([0]), set([4])]

        game = ReachabilityGame(2, graph, v3, obj, None, {0:1, 1:1, 2:2, 3:2, 4:2})

        path = [v3, v0]

        #print game.is_a_Nash_equilibrium_one_player(path, 1)

        #res = game.best_first_search(game.heuristic_short_path, None, 2)

        #random_path = game.test_random_path(100, 150)
        #doc = open("res_random.txt", "w")
        #for res in random_path:
            #doc.write(str(ReachabilityGame.path_vertex_to_path_index(res)))
            #doc.write("\n")
        #doc.close()

        res_init = game.best_first_search_with_init_path(game.heuristic, 2)
        print res_init

    def test_is_nash_equilibrium_3(self):

        mat = [[0, 10, 6, 2, 4],[2, 0, 10, 4, 6],[3, 6, 0, 5, 6],[6, 5, 9, 0, 6],[7, 10, 3, 6, 0]]

        v0 = Vertex(0, 2)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 2)
        v4 = Vertex(4, 1)

        vertex = [v0, v1, v2, v3, v4]

        pred = Graph.matrix_to_list_pred(mat)
        succ = Graph.list_pred_to_list_succ(pred)


        graph = Graph(vertex, mat, pred, succ, 10)
        obj = [set([0]), set([4])]

        game = ReachabilityGame(2, graph, v3, obj, None, {0:2, 1:2, 2:1, 3:2, 4:1})

        path = [v3, v4]


        print game.is_a_Nash_equilibrium(path)


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

        graph = Graph(vertex, None, list_pred, list_succ)
        goals = [set([3]), set([0])]
        init = v1

        id_to_player = {0:1, 1:1, 2:2, 3:1, 4:1}

        game = ReachabilityGame(2, graph, init, goals, None, id_to_player)

        en_set = game.test_random_path(50, 10)

        self.assertTrue(len(en_set) <= 50)
        self.assertEqual(10, len(en_set[0]))

        for en in en_set:
            print ReachabilityGame.path_vertex_to_path_index(en)


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

        graph = Graph(vertex, None, list_pred, list_succ)
        goals = [set([3]), set([0])]
        init = v1

        game = ReachabilityGame(2, graph, init, goals, None, None)

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

        graph = Graph(vertex, None, list_pred, list_succ)
        goals = [set([3]), set([0])]
        init = v1

        game = ReachabilityGame(2, graph, init, goals, None, None)

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

    def test_parcours_d_arbre(self):
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
        init = v1
        game = ReachabilityGame(2, graph, init, goals, None, None)
        prof = (game.player +1)*4*len(game.graph.vertex)

        result = game.parcours_d_arbre(prof)

        print "nombre de resultats", len(result)
        for i in range(0 , len(result)):
            print ReachabilityGame.path_vertex_to_path_index(result[i])



    def test_strange_example(self):

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 1)

        vertex = [v0, v1, v2, v3]

        pred0 = [(1, 1), (2, 1)]
        pred1 = [(0, 1), (3, 1)]
        pred2 = [(0, 1)]
        pred3 = [(1, 1)]

        pred = [pred0, pred1, pred2, pred3]
        succ = Graph.list_pred_to_list_succ(pred)

        graph = Graph(vertex, None, pred, succ )

        goal = [set([3]), set([2])]

        game = ReachabilityGame(2, graph, v0, goal, None, {0: 1, 1:2, 2:1, 3:1})

        result = game.parcours_d_arbre(12)
        print "nombre de resultats", len(result)
        for i in range(0, len(result)):
            (cost, reached) = game.get_info_path(result[i])
            print ReachabilityGame.path_vertex_to_path_index(result[i]), " infos:: cost", cost, " reached", reached

        best_result = game.filter_best_result(result)
        (cost,reached) = game.get_info_path(best_result)
        print ReachabilityGame.path_vertex_to_path_index(best_result) , "info :: cost", cost, "reached", reached




    def test_best_first_search(self):

        #sur l'exemple "etrange"

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 1)

        vertex = [v0, v1, v2, v3]

        pred0 = [(1, 1), (2, 1)]
        pred1 = [(0, 1), (3, 1)]
        pred2 = [(0, 1)]
        pred3 = [(1, 1)]

        pred = [pred0, pred1, pred2, pred3]
        succ = Graph.list_pred_to_list_succ(pred)

        graph = Graph(vertex, None, pred, succ, 1)

        goal = [set([3]), set([2])]

        game = ReachabilityGame(2, graph, v0, goal, None, {0: 1, 1: 2, 2: 1, 3: 1})

        candidate = game.best_first_search(ReachabilityGame.heuristic)
        candidate_a_star = game.best_first_search(ReachabilityGame.a_star, None, 5)

        print ReachabilityGame.path_vertex_to_path_index(candidate)
        print "a_star ", str(candidate_a_star)

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
        goals = [set([3]), set([0])]
        init = v1
        game = ReachabilityGame(2, graph, init, goals, None, {0:1, 1:1, 2:2, 3:1, 4:1})

        candidate = game.best_first_search(game.heuristic)
        candidate2 = game.best_first_search(game.heuristic_short_path)

        #result = game.parcours_d_arbre((game.player+1) * 4 * 5)

        print " exhaustif"
        #for res in result:
        #    print ReachabilityGame.path_vertex_to_path_index(res)
        #    print game.get_info_path(res)

        #best =game.filter_best_result(result)
        #print "best", ReachabilityGame.path_vertex_to_path_index(best)

        print " heuristique"
        print ReachabilityGame.path_vertex_to_path_index(candidate)

        print "heuristique short path"
        print ReachabilityGame.path_vertex_to_path_index(candidate2)




    def test_best_first_search3(self):

        # exemple du rapport

        v0 = Vertex(0, 2)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 2)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 1)
        v5 = Vertex(5, 2)
        v6 = Vertex(6, 1)
        v7 = Vertex(7, 2)

        vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

        # pred tableau des pred tq (u,k) u = index du pred, et k = valeur de l'arc

        pred0 = [(1, 1), (2, 4), (3, 5)]
        pred1 = [ (2, 1)]
        pred2 = [(0, 1), (3, 4), (4, 5)]
        pred3 = [(4, 1), (2, 1)]
        pred4 = [(5, 1), (3, 1)]
        pred5 = [(4, 1), (6,2)]
        pred6 = [(5, 1), (7, 1)]
        pred7 = [(6, 1)]

        list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7]
        list_succ = Graph.list_pred_to_list_succ(list_pred)

        print "list_succ", list_succ

        graph = Graph(vertices, None, list_pred, list_succ, 5)
        goal = [{6}, {0}]
        game = ReachabilityGame(2, graph, v3, goal, None, {0:2 , 1:1, 2:2, 3:1, 4:1, 5:2, 6:1, 7:2})

        #heuristic = game.best_first_search(ReachabilityGame.heuristic ,None, 5)
        print "done"
        #heuristic2 = game.best_first_search(ReachabilityGame.heuristic_short_path, None,5)
        print"done"
        #heuristic3 = game.best_first_search_with_init_path(ReachabilityGame.heuristic, 5)
        print "done"
        #heuristic4 = game.restart_best_first_search(ReachabilityGame.heuristic, 5)
        #prof = game.restart_best_first_search(ReachabilityGame.profondeur, 5)
        a_star = game.best_first_search(ReachabilityGame.a_star, None, 5)

        both_two = game.best_first_search_with_init_path_both_two(ReachabilityGame.heuristic_short_path, 5)
        print "both_two"
        print str(both_two)

        print "heuristic 1"
        #print repr(heuristic)
        print "heuristic 2"
        #print repr(heuristic2)
        print "heuristic 3"
        #print repr(heuristic3)
        print "heuristic 4"
        #print repr(heuristic4)
        print "prof"
        #print repr(prof)
        #print game.is_a_Nash_equilibrium(prof)
        print "a_star"
        print str(a_star)
        print game.get_info_path(a_star)

        #path = [v3, v2, v1, v0, v2, v3, v4, v5, v6]
        #print str(path)
        #print game.is_a_Nash_equilibrium(path)
        #print game.get_info_path(path)




        #result = game.test_random_path(100, 3*5*8)

        #res = game.filter_best_result(result)
        #print len(result)
        #for res in result:
         #  print ReachabilityGame.path_vertex_to_path_index(res)
        #print res

    def test_foireux(self):

        # exemple du rapport

        v0 = Vertex(0, 2)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 2)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 1)
        v5 = Vertex(5, 2)
        v6 = Vertex(6, 1)
        v7 = Vertex(7, 2)

        vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

        # pred tableau des pred tq (u,k) u = index du pred, et k = valeur de l'arc

        pred0 = [(1, 1), (2, 1), (3, 5)]
        pred1 = [(2, 1)]
        pred2 = [(0, 1), (3, 1), (4, 5)]
        pred3 = [(4, 1)]
        pred4 = [(5, 1)]
        pred5 = []
        pred6 = [(5, 1), (7, 1)]
        pred7 = [(6, 1)]

        list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7]
        list_succ = Graph.list_pred_to_list_succ(list_pred)

        graph = Graph(vertices, None, list_pred, list_succ, 5)
        goal = [set([0]), set([6])]
        dijk_graph2 = ReachabilityGame.graph_transformer(graph, 2)
        dijk_graph1 = ReachabilityGame.graph_transformer(graph, 1)

        game = ReachabilityGame(2, graph, v1, goal, None, {0:2 , 1:1, 2:2, 3:1, 4:1, 5:2, 6:1, 7:2})
        T2 = dijkstraMinMax(dijk_graph2, set([6]))
        T1 = dijkstraMinMax(dijk_graph1, set([0]))

        path = [v3, v0]

        print game.is_a_Nash_equilibrium_one_player(path, 1)
        print game.get_info_path(path)

        print_result(T1, set([0]), list_succ)



    def super_test(self):

        nb_vertex = 10
        poids_max = 100
        allowed_time = 30

        possible_goal = range(0, nb_vertex)
        random.shuffle(possible_goal)
        goal_1 = possible_goal.pop()
        goal_2 = possible_goal.pop()

        possible_init = range(0, nb_vertex)
        random.shuffle(possible_init)
        init = possible_init.pop()
        print "init v"+str(init)

        print "goal : joueur 1: v"+str(goal_1)+" joueur 2: v"+str(goal_2)
        game = ReachabilityGame.generate_game(2, nb_vertex, init,[{goal_1}, {goal_2}], 1, poids_max)
        game.graph.max_weight = poids_max



        #prof = game.best_first_search(ReachabilityGame.profondeur,None, 5)
        #if prof is not None:
        #    print "prof", repr(prof)
        #    print "EN?", game.is_a_Nash_equilibrium(prof)
        #else:
        #    print "prof a echoue"

        #res3 = game.best_first_search_with_init_path(ReachabilityGame.heuristic, 5)
        #res4 = game.restart_best_first_search(ReachabilityGame.heuristic, 5)
        #if res4 is not None:
          #  print "heuristique4", ReachabilityGame.path_vertex_to_path_index(res4)
          #  print "En?:", game.is_a_Nash_equilibrium(res4)
          #  print "info", game.get_info_path(res4)

        #res5 = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star, allowed_time)
        #if res5 is not None:
            #print "init_both_two", ReachabilityGame.path_vertex_to_path_index(res5)
            #print "En?:", game.is_a_Nash_equilibrium(res5)
        #else:
            #print "both_two a echoue"

        res6 = game.best_first_search(ReachabilityGame.a_star,None, allowed_time)
        if res6 is not None:
            print "a_star", ReachabilityGame.path_vertex_to_path_index(res6)
            print "En?:", game.is_a_Nash_equilibrium(res6)
        else:
            print "a_star a echoue"




        #if res3 is not None :
         #   print "heuristique3", ReachabilityGame.path_vertex_to_path_index(res3)
         #   print "En?: ", game.is_a_Nash_equilibrium(res3)
         #   print "info", game.get_info_path(res3)

        #res1 = game.best_first_search(game.heuristic, None, 5)
        #if res1 is not None:
           # print "heuristique1", ReachabilityGame.path_vertex_to_path_index(res1)
           # print "En?: ", game.is_a_Nash_equilibrium(res1)
           # print "info", game.get_info_path(res1)

        #res2 = game.best_first_search(game.heuristic_short_path,None, 5)
        #if res2 is not None:
            #print "heuristique2", ReachabilityGame.path_vertex_to_path_index(res2)
            #print "En?: ", game.is_a_Nash_equilibrium(res2)


        #result = game.test_random_path(100, (game.player + 1) * game.graph.max_weight * len(game.graph.vertex))

        #for res in result:
            #print str(res)






