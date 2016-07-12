import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from MinHeap import MinHeap
import unittest

class TestTasMin(unittest.TestCase):

    def test_insert_min(self):

        tas = MinHeap([4, 6, 4, 7, 9, 5, 9, 10, 11], 9)
        taille = tas.size
        tas.insertion(2)

        self.assertEqual(tas.tab[0], 2)
        self.assertEqual(tas.size, taille + 1)

    def test_insert_middle(self):
        tas = MinHeap([4, 6, 4, 7, 9, 5, 9, 10, 11], 9)
        tas.insertion(8)
        tas_voulu = MinHeap([4, 6, 4, 7, 8, 5, 9, 10, 11, 9], 10)
        self.assertEqual(tas, tas_voulu)

    def test_insert_middle_bis(self):
        tas = MinHeap([4, 6, 4, 7, 9, 5, 9, 10, 11], 9)
        tas.insertion(5)
        tas_voulu = MinHeap([4, 5, 4, 7, 6, 5, 9, 10, 11, 9], 10)
        self.assertEqual(tas, tas_voulu)

    def test_heapify(self):
        tas = MinHeap([11, 6, 4, 7, 9, 5, 9, 10, 4], 9)
        tas.heapify(0)
        tas_voulu = MinHeap([4, 6, 5, 7, 9, 11, 9, 10, 4], 9)
        self.assertEqual(tas, tas_voulu)

    def test_delete_min(self):
        tas = MinHeap([4, 6, 4, 7, 9, 5, 9, 10, 11], 9)
        tas.delete_min()
        tas_voulu = MinHeap([4, 6, 5, 7, 9, 11, 9, 10], 8)
        self.assertEqual(tas, tas_voulu)



if __name__ == '__main__':

    unittest.main()
