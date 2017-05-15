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

    def test_shortest_path_with_tuple(self):

        mat = [[(np.inf,np.inf),(np.inf,0),(-2,4),(np.inf,1)], [(4,np.inf),(np.inf,np.inf),(3,np.inf),(np.inf,1)],
               [(np.inf,1),(np.inf,np.inf),(np.inf,np.inf),(2,2)],[(np.inf,-5),(-1,np.inf),(np.inf,np.inf),(np.inf,np.inf)]]

        graph = Graph(None, mat, None, None, None)

        #on verifie que cela fonctionne si on calcule pour une seule composante du tuple
        self.assertEqual(graph.floyd_warshall(True,0),[[0,-1,-2,0],[4,0,2,4],[5,1,0,2],[3,-1,1,0]])
        with self.assertRaises(NegativeCircuitError):
            graph.floyd_warshall(True,1)
        #on verifie que le calcul des projections fonctionne bien
        self.assertEqual(Graph.mat_proj_componant(mat,0),[[np.inf,np.inf,-2,np.inf], [4,np.inf,3,np.inf], [np.inf,np.inf,np.inf,2],[np.inf,-1,np.inf,np.inf]])
        self.assertEqual(Graph.mat_proj_componant(mat,1),[[np.inf, 0, 4, 1], [ np.inf, np.inf,np.inf,1], [1, np.inf,np.inf,2],[-5,np.inf,np.inf,np.inf]])

if __name__ == '__main__':

    unittest.main()