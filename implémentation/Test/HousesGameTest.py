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

        file = open("debug.txt","a")
        nb_houses, energy_production, nb_interval, pref_tasks_list, p_out, p_in = parser_houses("../file_houses.txt")
        game = HousesGame(nb_houses, nb_interval, energy_production, pref_tasks_list, p_in, p_out)

        graph_house_to_dot(game, "12_intervalles.dot")

        print "nombres de sommets", len(game.graph.vertex)
        file.write("nombres de sommets "+ str(len(game.graph.vertex))+"\n")

        time_1 = time.time()
        strategies = game.backward()
        backward_house_to_dot(game,strategies, "dotdotdot___dot.dot")
        #SPE = HousesGame.get_SPE(strategies, nb_interval, nb_houses)
        SPE = game.get_SPE_until_last_reach(copy.deepcopy(strategies),nb_interval,nb_houses)
        print SPE
        all_SPE = game.get_all_SPE_until_last_reach(strategies,nb_interval,nb_houses)
        for e in all_SPE:
            print e
        time_1_end = time.time() - time_1
        print "time_backward", time_1_end
        file.write("time backward "+ str(time_1_end) + "\n")
        file.write("SPE "+ str(SPE)+"\n")


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
        file.write("time a_star "+ str(time_2_end)+"\n")
        file.write("a_star "+ str(a_star)+"\n")
        file.close()
        present  = False
        while len(all_SPE) != 0:
            if tuple(a_star) == all_SPE.pop():
                present = True
        self.assertTrue(present)

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
            SPE = game.get_SPE_until_last_reach(strategies, nb_interval, nb_houses)
            print SPE[0]
            SPE_index = ReachabilityGame.path_vertex_to_path_index(SPE)
            time_1_end = time.time() - time_1
            print "time_backward", time_1_end
            print SPE

            backward_house_to_dot(game,strategies, "houses_2_backward.dot")

            mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
            game.graph.mat = mat

            list_pred = Graph.matrix_to_list_pred(mat)
            game.graph.pred = list_pred
            (cout_SPE, reach_SPE) = game.get_info_path(SPE)
            print "cout " + str(cout_SPE) + " | atteints :" + str(reach_SPE)

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
            print a_star_index

            (cout_A, reach_A) = game.get_info_path(a_star)
            print "cout " + str(cout_A) + " | atteints :" + str(reach_A)

            self.assertEqual(SPE_index, a_star_index)

    def test_A_star_vs_backward(self):

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

        graph_house_to_dot(game, "a_star_vs_backward.dot")

        a_star = game.best_first_search(ReachabilityGame.a_star_negative,None,float("infinity"),True, True, None)
        print a_star

        strategies = game.backward()
        SPE = game.get_all_SPE_until_last_reach(strategies,1,2)
        for e in SPE:
            print list(e)