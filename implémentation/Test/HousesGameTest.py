import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import time


from HousesGame import *
from Value import *
from GraphToDotConverter import *

class TestHousesGame(unittest.TestCase):

    def test_first_example(self):

        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("Maisons/file_houses.txt")
        game = HousesGame(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)


        print "nombres de sommets", len(game.graph.vertex)

        time_1 = time.time()
        strategies = game.backward()
        #backward_house_to_dot(game,strategies, "DOT/maison_1.dot")
        length = (game.nb_interval * game.player) + 1

        SPE = game.get_SPE_until_last_reach(copy.deepcopy(strategies),length,nb_houses)
        print SPE
        SPE = game.get_SPE_until_last_reach(strategies,length,nb_houses)

        time_1_end = time.time() - time_1

        time_2 = time.time()

        mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
        game.graph.mat = mat

        list_pred = Graph.matrix_to_list_pred(mat)
        game.graph.pred = list_pred

        coalitions = {}
        for p in range(game.player):
            graph_min_max = ReachabilityGame.graph_transformer(game.graph, p+1)

            values_player = compute_value_with_negative_weight(graph_min_max, game.goal[p], True)[0]

            coalitions[p+1] = values_player

        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True, coalitions)
        print "A_star", a_star, " info : ", game.get_info_path(a_star)
        time_2_end = time.time() - time_2
        print "time_a_star", time_2_end
        print "SPE ", SPE, " info : ", game.get_info_path(SPE)
        print "time_backward", time_1_end

        self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(a_star)),
                     ReachabilityGame.sum_cost_path(game.get_info_path(SPE)))

    def test_houses_3(self):

        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("Maisons/houses_3.txt")
        game = HousesGame(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

        print "nombres de sommets", len(game.graph.vertex)


        time_1 = time.time()
        strategies = game.backward()
        length = (game.nb_interval * game.player) + 1

        SPE = game.get_SPE_until_last_reach(strategies, length, nb_houses)
        time_1_end = time.time() - time_1
        print "time_backward", time_1_end

        mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
        game.graph.mat = mat

        list_pred = Graph.matrix_to_list_pred(mat)
        game.graph.pred = list_pred

        coalitions = {}
        for p in range(game.player):
            graph_min_max = ReachabilityGame.graph_transformer(game.graph, p + 1)

            values_player = compute_value_with_negative_weight(graph_min_max, game.goal[p], True)[0]

            coalitions[p+1] = values_player



        time_2 = time.time()
        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True)
        time_2_end =  time.time() - time_2
        print "time_a_star", time_2_end

        print "A ", a_star,game.get_info_path(a_star)
        info_A = game.get_info_path(a_star)
        print "SPE", SPE, game.get_info_path(SPE)
        info_S = game.get_info_path(SPE)

        self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(a_star)), ReachabilityGame.sum_cost_path(game.get_info_path(SPE)))

    def test_houses_2(self):

            nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("Maisons/houses_2.txt")
            game = HousesGame(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

            print "nombres de sommets", len(game.graph.vertex)

            time_1 = time.time()
            strategies = game.backward()
            length = (game.nb_interval * game.player) + 1

            SPE = game.get_SPE_until_last_reach(copy.deepcopy(strategies), length, nb_houses)
            time_1_end = time.time() - time_1
            print "time_backward", time_1_end

            #backward_house_to_dot(game,strategies, "DOT/houses_2_backward.dot")

            mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
            game.graph.mat = mat

            list_pred = Graph.matrix_to_list_pred(mat)
            game.graph.pred = list_pred


            coalitions = {}
            for p in range(game.player):
                graph_min_max = ReachabilityGame.graph_transformer(game.graph, p + 1)

                values_player = compute_value_with_negative_weight(graph_min_max, game.goal[p], True)[0]

                coalitions[p+1] = values_player

            time_2 = time.time()
            a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True)
            time_2_end = time.time() - time_2
            print "time_a_star", time_2_end
            a_star_index = ReachabilityGame.path_vertex_to_path_index(a_star)
            print a_star_index

            (cout_A, reach_A) = game.get_info_path(a_star)
            print "cout " + str(cout_A) + " | atteints :" + str(reach_A)
            print "SPE: ", SPE, "Info ", game.get_info_path(SPE)

            self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(a_star)),
                             ReachabilityGame.sum_cost_path(game.get_info_path(SPE)))

    def test_A_star_vs_backward(self):

        #cet exemple montre que A* peut renvoyer un meilleur equilibre de le backward

        v0 = Vertex(0,1)
        v1 = Vertex(1,2)
        v2 = Vertex(2,2)
        v3 = Vertex(3,1)
        v4 = Vertex(4,1)
        v5 = Vertex(5,1)
        v6 = Vertex(6,1)

        all_vertices = [v0, v1, v2, v3, v4, v5, v6]

        succ0 = [(1,(0,0)), (2,(0,0))]
        succ1 = [(3,(1,2)), (4, (4,0))]
        succ2 = [(5,(5,4)), (6, (3,2))]
        succ3 = [(3,(0,0))]
        succ4 = [(4,(0,0))]
        succ5 = [(5,(0,0))]
        succ6 = [(6,(0,0))]

        succ = [succ0,succ1,succ2,succ3,succ4,succ5, succ6]

        mat = Graph.list_succ_to_mat(succ, True, 2)

        pred = Graph.matrix_to_list_pred(mat)

        goal = [{3,4,5,6}, {3,4,5,6}]

        game = HousesGame(2,1,1,[[],[]],4,2)

        game.graph.mat = mat
        game.graph.pred = pred
        game.graph.succ = succ
        game.goal = goal

        game.graph.max_weight = (5,2)

        game.graph.vertex = all_vertices


        a_star = game.best_first_search(ReachabilityGame.a_star_negative,None,float("infinity"),True, True, None)
        print "A_star", a_star, " info : ", game.get_info_path(a_star)

        strategies = game.backward()
        length = 3

        #backward_house_to_dot(game,strategies, "DOT/backward_vs_A_star.dot")
        SPE = game.get_SPE_until_last_reach(strategies,length,2)

        print "SPE", SPE, " info : ", game.get_info_path(SPE)

        self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(a_star)),4)
        self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(SPE)),5)


    def test_backward_2(self):

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 2)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 1)
        v5 = Vertex(5, 1)
        v6 = Vertex(6, 1)

        all_vertices = [v0, v1, v2, v3, v4, v5, v6]

        succ0 = [(1, (1, 1)), (2, (3, 3))]
        succ1 = [(3, (5, 1)), (4, (2, 2))]
        succ2 = [(5, (1, 5)), (6, (2, 3))]
        succ3 = [(3, (0, 0))]
        succ4 = [(4, (0, 0))]
        succ5 = [(5, (0, 0))]
        succ6 = [(6, (0, 0))]

        succ = [succ0, succ1, succ2, succ3, succ4, succ5, succ6]

        mat = Graph.list_succ_to_mat(succ, True, 2)

        pred = Graph.matrix_to_list_pred(mat)

        goal = [{3, 4, 5}, {1, 4}]

        graph = Graph(all_vertices, mat, pred, succ, (5,5))

        game = ReachabilityGame(2, graph, v0, goal, None)






        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True, None)
        print"A", a_star, "info ", game.get_info_path(a_star)

        strategies = game.backward()
        #print strategies
        length = 3

        #backward_house_to_dot(game,strategies, "DOT/backward_ameliore.dot")


        SPE = game.get_SPE_until_last_reach(strategies, length, 2)
        print "SPE", SPE, "info : ", game.get_info_path(SPE)

        self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(a_star)), ReachabilityGame.sum_cost_path(game.get_info_path(SPE)))


    def test_backward_3(self):

        #exemple p80 memoire

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 2)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 1)
        v5 = Vertex(5, 1)
        v6 = Vertex(6, 1)

        all_vertices = [v0, v1, v2, v3, v4, v5, v6]

        succ0 = [(1, (10, 1)), (2, (1, 10))]
        succ1 = [(3, (1, 2)), (4, (3, 4))]
        succ2 = [(5, (5, 6)), (6, (1, 1))]
        succ3 = [(3, (0, 0))]
        succ4 = [(4, (0, 0))]
        succ5 = [(5, (0, 0))]
        succ6 = [(6, (0, 0))]

        succ = [succ0, succ1, succ2, succ3, succ4, succ5, succ6]

        mat = Graph.list_succ_to_mat(succ, True, 2)

        pred = Graph.matrix_to_list_pred(mat)

        goal = [{1, 6}, {6, 4}]

        graph = Graph(all_vertices, mat, pred, succ, (5, 5))

        game = ReachabilityGame(2, graph, v0, goal, None)

        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True, None)
        print"A", a_star, "info ", game.get_info_path(a_star)

        strategies = game.backward()
        # print strategies
        length = 3

        #backward_house_to_dot(game, strategies, "DOT/backward_memoire.dot")

        SPE = game.get_SPE_until_last_reach(strategies, length, 2)
        print "SPE", SPE, "info : ", game.get_info_path(SPE)

        self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(a_star)), ReachabilityGame.sum_cost_path(game.get_info_path(SPE)))


    def test_houses_1_fixed_order(self):

        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("Maisons/file_houses.txt")
        game = HousesGameTest(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

        print "Nombres de sommets", len(game.graph.vertex)

        time_1 = time.time()
        strategies = game.backward()
        length = (game.nb_interval * game.player) + 1
        SPE = game.get_SPE_until_last_reach(copy.deepcopy(strategies), length, nb_houses)

        time_1_end = time.time() - time_1
        #backward_house_to_dot(game, strategies, "DOT/houses_1_fixed.dot")



        mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
        game.graph.mat = mat

        list_pred = Graph.matrix_to_list_pred(mat)
        game.graph.pred = list_pred

        time_2 = time.time()

        coalitions = {}
        for p in range(game.player):
            graph_min_max = ReachabilityGame.graph_transformer(game.graph, p + 1)

            values_player = compute_value_with_negative_weight(graph_min_max, game.goal[p], True)[0]

            coalitions[p+1] = values_player

        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True,
                                        coalitions)
        time_2_end = time.time() - time_2
        print "Temps pour A_star", time_2_end
        print "A_star", a_star, "Info :", game.get_info_path(a_star)
        print ""
        print "Temps pour le backward", time_1_end

        print "SPE ",
        print SPE, "Info :", game.get_info_path(SPE)


        self.assertEquals((6,4), game.graph.max_weight)

        self.assertEquals([0, 1, 2, 3, 4], game.path_vertex_to_path_index(a_star))

    def test_houses_2_fixed_order(self):

        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("Maisons/houses_2.txt")
        game = HousesGameTest(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

        print "Nombres de sommets", len(game.graph.vertex)

        time_1 = time.time()
        strategies = game.backward()
        print "STRAT", strategies
        length = (game.nb_interval * game.player) + 1
        SPE = game.get_SPE_until_last_reach(copy.deepcopy(strategies), length, nb_houses)

        time_1_end = time.time() - time_1
        #backward_house_to_dot(game, strategies, "DOT/houses_2_fixed.dot")

        mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
        game.graph.mat = mat

        list_pred = Graph.matrix_to_list_pred(mat)
        game.graph.pred = list_pred

        time_2 = time.time()

        coalitions = {}
        for p in range(game.player):
            graph_min_max = ReachabilityGame.graph_transformer(game.graph, p + 1)

            values_player = compute_value_with_negative_weight(graph_min_max, game.goal[p], True)[0]

            coalitions[p+1] = values_player

        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True,
                                        coalitions)

        time_2_end = time.time() - time_2
        print "Temps pour A_star", time_2_end
        print "A_star", a_star, "Info :", game.get_info_path(a_star)
        print ""
        print "Temps pour le backward", time_1_end

        print "SPE ", SPE , " Info :", game.get_info_path(SPE)



        self.assertEquals((4, 6, 10), game.graph.max_weight)

        self.assertEquals([0, 1, 2, 35, 36, 37, 38, 39, 40, 41], game.path_vertex_to_path_index(a_star))

    def test_houses_4_fixed_order(self):

        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("Maisons/houses_4.txt")
        game = HousesGameTest(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

        print "Nombres de sommets", len(game.graph.vertex)

        time_1 = time.time()
        strategies = game.backward()
        length = (game.nb_interval * game.player) + 1
        SPE = game.get_SPE_until_last_reach(copy.deepcopy(strategies), length, nb_houses)

        time_1_end = time.time() - time_1

        #backward_house_to_dot(game,strategies, "DOT/houses_4_fixed.dot")



        print "SPE OK"

        mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
        game.graph.mat = mat
        print "mat OK"


        time_2 = time.time()

        print "begin coal"
        coalitions = {}
        for p in range(game.player):
            print "Coal"
            graph_min_max = ReachabilityGame.graph_transformer(game.graph, p + 1)

            values_player = compute_value_with_negative_weight(graph_min_max, game.goal[p], True)[0]

            coalitions[p+1] = values_player

        print "COALITIONS OK"

        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True,
                                        coalitions)

        time_2_end = time.time() - time_2
        print "Temps pour A_star", time_2_end
        print "A_star", a_star, "Info :", game.get_info_path(a_star)
        print ""
        print "Temps pour le backward", time_1_end

        print "SPE:" , SPE , "Info :", game.get_info_path(SPE)

        self.assertEqual(ReachabilityGame.sum_cost_path(game.get_info_path(a_star)), ReachabilityGame.sum_cost_path(game.get_info_path(SPE)))





