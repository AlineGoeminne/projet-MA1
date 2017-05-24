import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import time


from HousesGame import *
from Value import *

class TestHousesGame(unittest.TestCase):

    def test_first_example(self):
        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("../file_houses.txt")
        game = HousesGame(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

        print "nombres de sommets", len(game.graph.vertex)

        time_1 = time.time()
        strategies = game.backward()
        SPE = HousesGame.get_SPE(strategies, nb_interval, nb_houses)
        time_1_end = time.time() - time_1
        print "time_backward", time_1_end

        mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
        game.graph.mat = mat

        list_pred = Graph.matrix_to_list_pred(mat)
        game.graph.pred = list_pred

        coalitions = {}
        for p in range(game.player):
            graph_min_max = ReachabilityGame.graph_transformer(game.graph, p+1)

            values_player = compute_value_with_negative_weight(graph_min_max, game.goal[p], True)[0]

            coalitions[p] = values_player

        time_2 = time.time()
        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True, coalitions)
        time_2_end = time.time() - time_2
        print "time_a_star", time_2_end
        a_star_index = ReachabilityGame.path_vertex_to_path_index(a_star)

        self.assertEqual(SPE, a_star_index)

    def test_houses_3(self):

        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("houses_3.txt")
        game = HousesGame(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

        print "nombres de sommets", len(game.graph.vertex)


        time_1 = time.time()
        strategies = game.backward()
        SPE = HousesGame.get_SPE(strategies, nb_interval, nb_houses)
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

            coalitions[p] = values_player



        time_2 = time.time()
        a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True)
        time_2_end =  time.time() - time_2
        print "time_a_star", time_2_end
        a_star_index = ReachabilityGame.path_vertex_to_path_index(a_star)

        self.assertEqual(SPE, a_star_index)

    def test_houses_2(self):

            nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("houses_2.txt")
            game = HousesGame(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

            print "nombres de sommets", len(game.graph.vertex)

            time_1 = time.time()
            strategies = game.backward()
            SPE = HousesGame.get_SPE(strategies, nb_interval, nb_houses)
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

                coalitions[p] = values_player

            time_2 = time.time()
            a_star = game.best_first_search(ReachabilityGame.a_star_negative, None, float("infinity"), True, True)
            time_2_end = time.time() - time_2
            print "time_a_star", time_2_end
            a_star_index = ReachabilityGame.path_vertex_to_path_index(a_star)

            self.assertEqual(SPE, a_star_index)
