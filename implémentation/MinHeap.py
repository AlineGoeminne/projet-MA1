"""
Le module TasMin
=============

Ce module permet d'implementer la structure de donnees qu est le tas. Un tas stocker de maniere efficace un ensemble
de donnees ordonnees. On peut se representer un tas comme un arbre binaire equilibre tq pour tout noeud p et
ses deux fils f1 et f2 p <= f1 et p <= f2.

La particularite de cette structure est qu elle implemente les fonctions suivantes:
Soit n le nombre d'elements du tas,

- lecture de la plus grande valeur du tas en O(1)
- ajout d'un element dans le tas en O(log n)
- suppression d'un element en O(log n)
- modification de la clef par laquelle l'element est stocke en O(log n)

"""


class MinHeap(object):

    def __init__(self, tab=None, size=0):
        if tab is None:
            self.tab = []
        else:
            self.tab = tab
        self.size = size

    def __eq__(self, other):

        if id(self) == id(other):
            return True

        if self.size != other.size:
            return False
        else:
            for i in range(0, self.size - 1):
                if self.tab[i] != other.tab[i]:
                    return False
            return True

    def __repr__(self):

        if self.size == 0:
            return "[ ]"
        else:

            result = "[ "
            for i in range(0, self.size-1):
                result += repr(self.tab[i]) + ", "
            return "Tas: " + result + repr(self.tab[self.size-1]) + "]"

    def __sizeof__(self):
        return len(self.tab)

    def is_empty(self):

        if self.size == 0:
            return True
        else:
            return False

    def find_index_father(self, i):

        """
        Cette fonction permet de retrouver l'indice dans le tableau correspondant au pere de l element stocke en tab[i]

         :param i : indice de l'element dont on checrhe l indice du pere
         :type i : int

        """

        j = i+1
        p = j/2 -1
        return p

    def find_index_l_son(self, i):

        j = i+1
        f = 2*j -1
        return f

    def find_index_r_son(self, i):

        j = i+1
        f = 2*j
        return f

    def read_min(self):
        return self.tab[0]

    def insertion(self, item, modeInd=False):

        i = self.size
        self.tab.append(item)
        self.size += 1

        if modeInd:
            item.index = i

        while i >= 1 and self.tab[self.find_index_father(i)] > self.tab[i]:

            f_index = self.find_index_father(i)
            f_val = self.tab[f_index]

            self.tab[i] = f_val
            self.tab[f_index] = item

            if (modeInd):
                self.tab[i].index = i
                self.tab[f_index].index = f_index

            i = f_index

    def heapify(self, i, modeInd=False):

        """
            Cet algorithme reetablit l ordre sur le tas, si l element a l indice i est trop grand
            :param i: indice pour lequel tas[i] est eventuellement trop grand
        """

        l = self.find_index_l_son(i)
        r = self.find_index_r_son(i)

        size = self.size

        if (l <= size - 1) and (self.tab[l] < self.tab[i]):
            smallest = l
        else:
            smallest = i

        if (r <= size -1) and (self.tab[r] < self.tab[smallest]):
            smallest = r
        if i != smallest:

            inv1 = self.tab[smallest]
            inv2 = self.tab[i]

            self.tab[i] = inv1
            self.tab[smallest] = inv2

            if modeInd:
                self.tab[i].index = i
                self.tab[smallest].index = smallest

            self.heapify(smallest, modeInd)

    def delete_min(self, modeInd = False):

        d = self.tab[0]
        if modeInd:
            d.index = None

        if self.size == 1:
            self.tab = []
            self.size -= 1

        else:
            self.tab[0] = self.tab[self.size -1]
            if modeInd:
                self.tab[0].index = 0
            self.size -= 1
            del (self.tab[self.size])  #TODO: necessaire?

            self.heapify(0, modeInd)

        return d

    def heap_decrease_key(self, i, key, modeInd=False):

        """
            A partir de l'indice i d'un item diminue la clef correspondant a cet item.
            De plus, un parametre optionel permet d indiquer si l item modifie est un objet tel qu il possede un
            attribut permettant de stocker sa position dans le tas.

        """
        if ((not modeInd) and (key > self.tab[i].key)) or ((modeInd) and (key > self.tab[i].key)):
            raise ValueError("key doit etre plus petit que la valeur sctockee actuellement")
        else:
            self.tab[i].key = key

            while i > 0 and self.tab[self.find_index_father(i)] > self.tab[i]:
                pere = self.tab[self.find_index_father(i)]

                if(modeInd):
                    (pere.index,  self.tab[i].index) = (self.tab[i].index ,pere.index)

                (self.tab[self.find_index_father(i)],  self.tab[i]) = ( self.tab[i],self.tab[self.find_index_father(i)])

                i = self.find_index_father(i)


    def heap_increase_key(self, i, key, modeInd=False):

        """
            A partir de l'indice i d'un item augmente la clef correspondant a cet item.
            De plus, un parametre optionel permet d indiquer si l item modifie est un objet tel qu il possede un
            attribut permettant de stocker sa position dans le tas.

        """

        if ((not modeInd) and (key < self.tab[i].key)) or ((modeInd) and (key < self.tab[i].key)):
            raise ValueError("key doit etre plus grande que la valeur sctockee actuellement")
        else:
            self.tab[i].key = key
            self.heapify(i, modeInd)

    def sort(self):

        ret = []

        while self.size > 0:
            ret.append(self.delete_min())

        return ret



def test():

        tas = MinHeap([4, 6, 4, 7, 9, 5, 9, 10, 11], 9)
        pere = tas.find_index_father(1)
        filsgauche = tas.find_index_l_son(1)
        filsdroit = tas.find_index_r_son(1)
        #tas.insertion(10)
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        min = tas.delete_min()
        print tas
        print tas.is_empty()

        print "min ", min
        print "size ",tas.size
        print tas


        tas.insertion(4)
        print tas
        tas.insertion(43)
        print tas
        tas.delete_min()
        print tas



if __name__=='__main__':
    test()



