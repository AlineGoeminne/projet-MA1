from AtteignabilityGame import Vertex,Graph

from MinHeap import MinHeap

class VertexDijk(object):

    def __init__(self, id, player, key = float("infinity"), index=0):

        self.id = id        #represente l'ID du sommet
        self.player = player #represente le joueur auquel le noeud appartient
        self.key = key      #correspondra a la valeur du noeud apres l execution de l algo, clef du tas Q
        self.index = index  #index dans Q

    def __eq__(self, other):

        if id(self) == id(other):
            return True
        else:

            return self.id == other.id and self.key == other.key and self.index == other.index and self.S == other.S

    def __ge__(self, other):
        return self.key >= other.key

    def __gt__(self, other):
        return self.key > other.key

    def __le__(self, other):
        return self.key <= other.key

    def __lt__(self, other):
        return self.key < other.key

    def __repr__(self):
        return "( v" + str(self.id) + ", " + str(self.key) + ")"


class VertexDijkPlayerMin(VertexDijk):

    def __init__(self, id, player = 1, key =float("infinity"), min = None, index=0 ):

        VertexDijk.__init__(self, id, player, key, index)
        self.S = min


class VertexDijkPlayerMax(VertexDijk):

    def __init__(self, id, nbrSucc, player = 2, key = float("infinity"), heap = None, index = 0):

        VertexDijk.__init__(id, player, key, index)
        self.S = heap  # represente un tas pour les noeuds du joueur max
        self.nbrSucc = nbrSucc



"""

Cette classe represente les objets qui seront stockes dans chaque tas S de chaque vertex.
Pour un certain vertex v, on possede un tas S composes d objets successeurs.
Un objet successeur possede un attribut ID qui represente un successeur s de v et un attribut key qui represente la valeur
si en v on passe par s.

"""


class Successor(object):

    def __init__(self, id, key):

        self.id = id
        self.key = key

    def __repr__(self):
        return "( v" + str(self.id) + ", " + str(self.key) + ")"

    def __eq__(self, other):

        if id(self) == id(other):
            return True
        else:

            return self.id == other.id and self.key == other.key

    def __ge__(self, other):
        return self.key >= other.key

    def __gt__(self, other):
        return self.key > other.key

    def __le__(self, other):
        return self.key <= other.key

    def __lt__(self, other):
        return self.key < other.key

"""
    Initialise pour chaque noeud du graphe la structure de donnee permettant de tenir a jour la valeur future de Val(v)
    :param graph : un graph represente par sa liste de predecesseurs

"""

def initS(graph, goal):

    vertices = graph.vertex

    for v in vertices :
        if v.id in goal:
            v.S = Successor(None,0)

        else:
            if v.player == 1 :  # joueur Min
                v.S = Successor(None, float("infinity"))
            else:  # joueur Max
                S = MinHeap()
                v.S = S
                v.S.tab[0] = Successor(None, float("infinity"))

"""
Initialise le tas Q de l'algorithme DijkstraMinMax
"""


def initQ(graph,goal):

    Q = MinHeap()
    i = 0
    for o in goal:
        v = graph.vertex[o]
        v.key = 0
        Q.tab[i] = v
        i += 1

    vertices = graph.vertex
    for j in range(0,len(vertices)-2):
        if not(vertices[j] in goal):
            v = vertices[j]
            v.key = float("infinity")
            Q.tab[i] = v




def relaxation(p, s, w, Q):

    """
    Relaxation du predecesseur p de s sachant que l arc (p,s) est des poids w
    """

    sVal = s.S.read_min()
    pValNew = w + sVal
    pValOld = p.S.read_min()
    p.S.insertion()

    if pValNew < pValOld: #alors il faut changer la valeur dans Q

        Q.heap_decrease_key(p.index, pValNew, True) #TODO: retirer le none

        if p.player == 1:  # Alors il faut changer la valeur stockee
            p.key = pValNew
    if p.player == 2 :  # S'il s'agit d un noeud du joueur Max alors on l ajoute a ses possibilites de chemin
        p.S.insertion(Successor(s.id, pValNew))

def block_max(s,Q):

    s.S.delete_min()
    new_min = s.S.read_min()

    Q.heap_increase_key(s.index, new_min.key, True)
    s.nbSucc -= 1

    # vu que pas besoin de retrouver les strategies optimales dans mon cas, je ne stocke pas les arcs deja bloques

def convertPred2NbrSucc(pred):

    """
    A partir de la liste des predecesseurs calcule le nombre de successeurs de chaque noeud.

    """
    nbrSucc = [0]*len(pred)
    for i in range(0,len(pred)):
        pred_i = pred[i]
        if pred_i is not None:
            for j in range(0, len(pred_i)):
                (v, w) = pred[i][j]
                nbrSucc[v] += 1

    return nbrSucc



def graph_transformer(graph):
    """
    A partir des sommets "normaux" , construit les sommets que nous utiliserons dans l algorithme de Dijkstra

    """

    vertices = graph.vertex
    newVertices = []

    nbrSucc = convertPred2NbrSucc(graph.pred)

    for i in range (0, len(vertices)):
        oldVert = vertices[i]
        print "i = ", i, "et id = ", oldVert.id
        if oldVert.player == 1 : #Noeud du joueur Min

            dijkVert = VertexDijkPlayerMin(oldVert.id)

        else : # Noeud du joueur Max, il faut aussi recuperer son nombre de sucesseurs
            dijkVert = VertexDijkPlayerMax(oldVert.id, nbrSucc[oldVert.id])

        newVertices[i] = dijkVert

    return Graph(newVertices,graph.mat, graph.pred, graph.succ)


def test():

    v1 = VertexDijk(1, 1, 4)
    v2 = VertexDijk(2, 1, 7)
    v3 = VertexDijk(3, 1, 12)
    v4 = VertexDijk(4, 1, 2)

    tas = MinHeap()

    tas.insertion(v1,True)
    tas.insertion(v2,True)
    tas.insertion(v3,True)
    tas.insertion(v4,True)

    #print "index v3", v3.index
    #print "index v1", v1.index
    #print "index v2", v2.index
    #print "index v4", v4.index
    #TODO: probleme de la modification des indices lors de l insertion
    vnew = VertexDijk(3, 1, 1)
    tas.heap_decrease_key(v3.index, vnew, True, v3)

    print tas
    print "index v3", v3.index
    print "index v1", v1.index
    print "index v2", v2.index
    print "index v4", v4.index

    tas.heap_increase_key(v1.index, VertexDijk(1, 1, 12), True, v1)
    print tas
    print "index v3", v3.index
    print "index v1", v1.index
    print "index v2", v2.index
    print "index v4", v4.index

if __name__ == "__main__":
    #test()

    v1 = VertexDijkPlayerMin(1)
    print v1