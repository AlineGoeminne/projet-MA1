import unittest

from DijkstraMinMax import convertPred2NbrSucc
from DijkstraMinMax import dijkstraMinMax
from DijkstraMinMax import print_result
from AtteignabilityGame import Graph
from AtteignabilityGame import Vertex




class TestDijkstraMinMax(unittest.TestCase):

    def test_convertPred2NbrSuccTest(self):

        v0_pred = [(1, 42), (2, 42), (3, 42)]
        v1_pred = [(2, 42)]
        v2_pred = [(1, 42), (2, 42), (0, 42)]

        pred = [v0_pred, v1_pred, v2_pred, []]

        nbrSucc = convertPred2NbrSucc(pred)

        self.assertEqual(nbrSucc[3], 1)
        self.assertEqual(nbrSucc[0], 1)
        self.assertEqual(nbrSucc[1], 2)
        self.assertEqual(nbrSucc[2], 3)

    def test_recherche_valeurs_1(self):
        v0 = Vertex(0, 2)
        v1 = Vertex(1, 1)
        v2 = Vertex(2, 2)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 1)
        v5 = Vertex(5, 2)
        v6 = Vertex(6, 1)
        v7 = Vertex(7, 2)

        vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

        pred0 = [(0, 1), (1, 1), (2, 1), (3, 5)]
        pred1 = [(2, 1)]
        pred2 = [(3, 1), (4, 5)]
        pred3 = [(4, 1)]
        pred4 = [(5, 1)]
        pred5 = []
        pred6 = [(5, 1), (7, 1)]
        pred7 = [(6, 1)]

        list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7]

        graph = Graph(vertices, None, list_pred, None)
        goal = set([0])

        T = dijkstraMinMax(graph, goal)

        for v in T:

            key = v.key
            id = v.id

            if id == 0:
                self.assertEqual(key, 0)

            if id == 1:
               self.assertEqual(key, 1)
            if id == 2:
                self.assertEqual(key, 2)
            if id == 3:
                self.assertEqual(key, 3)
            if id == 4:
                self.assertEqual(key, 4)
            if id == 5:
                self.assertEqual(key, float("infinity"))
            if id == 6:
                self.assertEqual(key, float("infinity"))
            if id == 7:
                self.assertEqual(key, float("infinity"))


    def test_recherche_valeurs_2(self):

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 1)
        v4 = Vertex(4, 2)
        v5 = Vertex(5, 2)
        v6 = Vertex(6, 1)

        vertices = [v0, v1, v2, v3, v4, v5, v6]

        pred0 = [(1, 1), (2, 9)]
        pred1 = [(3, 1)]
        pred2 = [(5, 1)]
        pred3 = [(5,3), (2, 1)]
        pred4 = [(6, 1)]
        pred5 = [(4, 4)]
        pred6 = [(5, 2)]

        list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6]

        graph = Graph(vertices, None, list_pred, None)
        goal = set([0])

        T = dijkstraMinMax(graph, goal)

        for v in T:

            key = v.key
            id = v.id

            if id == 0:
                self.assertEqual(key, 0)

            if id == 1:
                self.assertEqual(key, 1)
            if id == 2:
                self.assertEqual(key, 3)
            if id == 3:
                self.assertEqual(key, 2)
            if id == 4:
                self.assertEqual(key, float("infinity"))
            if id == 5:
                self.assertEqual(key, float("infinity"))
            if id == 6:
                self.assertEqual(key, float("infinity"))


    def test_recherche_valeurs_2obj(self):

            v0 = Vertex(0, 2)
            v1 = Vertex(1, 1)
            v2 = Vertex(2, 2)
            v3 = Vertex(3, 1)
            v4 = Vertex(4, 1)
            v5 = Vertex(5, 2)
            v6 = Vertex(6, 1)
            v7 = Vertex(7, 2)

            vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

            pred0 = [(0, 1), (1, 1), (2, 1), (3, 5)]
            pred1 = [(2, 1)]
            pred2 = [(3, 1), (4, 5)]
            pred3 = [(4, 1)]
            pred4 = [(5, 1)]
            pred5 = []
            pred6 = [(5, 1), (7, 1)]
            pred7 = [(6, 1)]

            list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7]

            graph = Graph(vertices, None, list_pred, None)
            goal = set([0, 7])

            T = dijkstraMinMax(graph, goal)
            print_result(T,goal)

            for v in T:

                key = v.key
                id = v.id

                if id == 0:
                    self.assertEqual(key, 0)

                if id == 1:
                    self.assertEqual(key, 1)
                if id == 2:
                    self.assertEqual(key, 2)
                if id == 3:
                    self.assertEqual(key, 3)
                if id == 4:
                    self.assertEqual(key, 4)
                if id == 5:
                    self.assertEqual(key, 5)
                if id == 6:
                    self.assertEqual(key, 1)
                if id == 7:
                    self.assertEqual(key, 0)