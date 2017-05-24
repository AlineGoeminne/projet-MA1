import os
import sys
import unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))


from GraphToDotConverter import *
from Value import *
import numpy as np





class TestValueWithNegWeight(unittest.TestCase):

    def test_example_from_the_article(self):
        W = 10

        v0 = Vertex(0, 2)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 2)

        vertex = [v0, v1, v2]

        succ0 = [(1, -1), (2, -W)]
        succ1 = [(0, 0), (2, 0)]
        succ2 = [(2, 0)]

        succ = [succ0, succ1, succ2]

        graph = Graph(vertex, None, None, succ, W)

        target = {2}

        res = compute_value_with_negative_weight(graph, target)

        game = ReachabilityGame(2, graph, None, target, None)

        minMax_graph_to_dot(game, res[0], "test.dot")

        self.assertEqual(res[0], [-W,-W,0])


    def test_other_example(self):

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 2)

        vertex = [v0, v1, v2, v3]

        succ1 = [(1, 0), (3, 0)]
        succ2 = [(0, -1), (2, 10)]
        succ3 = [(2, 0)]
        succ4 = [(3, 0)]

        succ = [succ1, succ2, succ3, succ4]

        graph = Graph(vertex, None, None, succ, 10)

        target = {2}

        res = compute_value_with_negative_weight(graph, target)

        game = ReachabilityGame(2,graph,None,target,None)

        minMax_graph_to_dot(game,res[0], "test2.dot")

        self.assertEqual(res[0],[-np.inf,-np.inf,0,np.inf])


    def test_example_from_master_thesis(self):

        v0 = Vertex(0,1)
        v1 = Vertex(1,2)
        v2 = Vertex(2,1)
        v3 = Vertex(3,1)
        v4 = Vertex(4,1)
        v5 = Vertex(5,2)
        v6 = Vertex(6,1)
        v7 = Vertex(7,1)

        vertex = [v0, v1, v2, v3, v4, v5, v6, v7]

        succ0 = [(1,1)]
        succ1 = [(4, 1), (2, 1)]
        succ2 = [(2, 0)]
        succ3 = [(5, -1)]
        succ4 = [(6, -1)]
        succ5 = [(6,1),(7,-1)]
        succ6 = [(7,1),(4,0)]
        succ7 = [(7,0)]

        succ = [succ0,succ1, succ2, succ3, succ4, succ5, succ6, succ7]

        graph = Graph(vertex, None, None, succ, 1)

        target = {7}

        res = compute_value_with_negative_weight(graph, target)

        game = ReachabilityGame(2, graph, None, target, None)

        minMax_graph_to_dot(game, res[0], "from_master_thesis_example.dot")


        self.assertEqual(res[0],[np.inf,np.inf,np.inf,-2,-np.inf,-1,-np.inf,0])


    def test_example_from_master_thesis_modify(self):

        v0 = Vertex(0,1)
        v1 = Vertex(1,2)
        v2 = Vertex(2,1)
        v3 = Vertex(3,1)
        v4 = Vertex(4,1)
        v5 = Vertex(5,2)
        v6 = Vertex(6,1)
        v7 = Vertex(7,1)

        vertex = [v0, v1, v2, v3, v4, v5, v6, v7]

        succ0 = [(1,1)]
        succ1 = [(4, 1), (2, 1)]
        succ2 = [(2, 0)]
        succ3 = [(5, 1)]
        succ4 = [(6, 1)]
        succ5 = [(6,1),(7,1)]
        succ6 = [(7,1),(4,1)]
        succ7 = [(7,1)]

        succ = [succ0,succ1, succ2, succ3, succ4, succ5, succ6, succ7]

        graph = Graph(vertex, None, None, succ, 1)

        target = {7}

        res = compute_value_with_negative_weight(graph, target)

        self.assertEqual(res[0],[np.inf,np.inf,np.inf,3,2,2,1,0])



        game = ReachabilityGame(2, graph, None, target, None)

        minMax_graph_to_dot(game, res[0], "from_master_thesis_example_bis.dot")

    def test_example_with_tuple(self):

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 1)
        v5 = Vertex(5, 2)
        v6 = Vertex(6, 1)
        v7 = Vertex(7, 1)

        vertex = [v0, v1, v2, v3, v4, v5, v6, v7]

        succ0 = [(1, (1,1))]
        succ1 = [(4, (1,1)), (2, (1,1))]
        succ2 = [(2, (0,1))]
        succ3 = [(5, (-1,1))]
        succ4 = [(6, (-1,1))]
        succ5 = [(6, (1,1)), (7, (-1,1))]
        succ6 = [(7, (1,1)), (4, (0,1))]
        succ7 = [(7, (0,1))]

        succ = [succ0, succ1, succ2, succ3, succ4, succ5, succ6, succ7]
        W = (1,1)
        graph = Graph(vertex, None, None, succ, W)

        target = {7}

        res1 = compute_value_with_negative_weight(graph, target,True,0)
        res2 = compute_value_with_negative_weight(graph, target,True,1)


        self.assertEqual(res1[0],[np.inf,np.inf,np.inf,-2,-np.inf,-1,-np.inf,0])

        self.assertEqual(res2[0],[np.inf,np.inf,np.inf,3,2,2,1,0])


        #game = ReachabilityGame(2, graph, None, target, None)

        #minMax_graph_to_dot(game, res1[0], "from_master_thesis_example_uple.dot")

    def test_from_simon_example(self):
        v0 = Vertex(0, 2)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 1)

        vertex = [v0, v1, v2]

        succ1 = [(1, -1), (2, -25)]
        succ2 = [(0, 0), (2, 0)]
        succ3 = []

        succ = [succ1, succ2, succ3]

        graph = Graph(vertex, None, None, succ, 25)

        target = {2}

        res = compute_value_with_negative_weight(graph, target)

        self.assertEqual(res[0],[-25,-25,0])

        game = ReachabilityGame(2, graph, None, target, None)

        minMax_graph_to_dot(game, res[0], "test3.dot")


    def test_is_there_infinite_value_test(self):

        v0 = Vertex(0,1)
        v1 = Vertex(1,1)
        v2 = Vertex(2,1)
        v3 = Vertex(3,2)
        v4 = Vertex(4,1)
        v5 = Vertex(5,1)

        vertex = [v0, v1, v2, v3, v4, v5]

        succ0 = [(1,1),(2,1)]
        succ1 = [(1,0)]
        succ2 = [(3,-5)]
        succ3 = [(5,50),(4,-1)]
        succ4 = [(2,-1),(5,1)]
        succ5 = [(5,0)]

        succ = [succ0, succ1, succ2, succ3, succ4, succ5]

        graph = Graph(vertex, None, None, succ, 50)

        target = {5}

        res = compute_value_with_negative_weight(graph, target)


        game = ReachabilityGame(2, graph, None, target, None)

        minMax_graph_to_dot(game, res[0], "no_inifinity.dot")

        self.assertEqual(res[0],[46,np.inf,45,50,1,0])


if __name__ == '__main__':

    unittest.main()

