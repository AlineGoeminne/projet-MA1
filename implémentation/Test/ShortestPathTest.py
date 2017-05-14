import os
import sys
import numpy as np
import unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from GraphGame import Graph
from GraphGame import NegativeCircuitError


class TestShortestPath(unittest.TestCase):

    def test_shortest_path_with_negative_circuit(self):

        mat = [[np.inf, 0, 4, 1], [ np.inf, np.inf,np.inf,1], [1, np.inf,np.inf,2],[-5,np.inf,np.inf,np.inf]]

        graph = Graph(None,mat,None,None,None)



        with self.assertRaises(NegativeCircuitError):
            graph.floyd_warshall()


    #Exemple de la page wiki sur Floyd-Warshall
    def  test_shortest_path_with_negative_weight(self):

        mat = [[np.inf,np.inf,-2,np.inf], [4,np.inf,3,np.inf], [np.inf,np.inf,np.inf,2],[np.inf,-1,np.inf,np.inf]]

        graph = Graph(None, mat, None, None, None)

        self.assertEqual(graph.floyd_warshall(),[[0,-1,-2,0],[4,0,2,4],[5,1,0,2],[3,-1,1,0]])


if __name__ == '__main__':

    unittest.main()