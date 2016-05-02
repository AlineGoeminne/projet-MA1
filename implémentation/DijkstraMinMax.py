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

        VertexDijk.__init__(self, id, player, key, index)
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

    for v in vertices:
        if v.id in goal:
            v.S = Successor(None,0)

        else:
            if v.player == 1:  # joueur Min
                v.S = Successor(None, float("infinity"))
            else:  # joueur Max
                v.S = MinHeap()
                v.S.tab = [Successor(None, float("infinity"))]
                v.S.size += 1

"""
Initialise le tas Q de l'algorithme DijkstraMinMax
"""


def initQ(graph, goal):

    init_tab = [0]*len(graph.vertex)
    Q = MinHeap(init_tab, len(graph.vertex))
    i = 0
    for o in goal:
        v = graph.vertex[o]
        v.key = 0
        v.index = i
        Q.tab[i] = v
        i += 1

    vertices = graph.vertex
    for j in range(0, len(vertices)):
        if not(vertices[j].id in goal):
            v = vertices[j]
            v.key = float("infinity")
            v.index = i
            Q.tab[i] = v
            i += 1

    return Q



def relaxation(p, s, w, Q, goal):

    """
    Relaxation du predecesseur p de s sachant que l arc (p,s) est des poids w
    """
    if(s.player == 1) or (s.id in goal):
        sVal = s.S.key

    else:
        sVal = s.S.read_min().key

    pValNew = w + sVal

    if(p.player == 1) or (p.id in goal):
        pValOld = p.S.key

    else:
        pValOld = p.S.read_min().key

    if pValNew < pValOld: #alors il faut changer la valeur dans Q
        Q.heap_decrease_key(p.index, pValNew, True)

        if p.player == 1: # Noeud min: modification de la valeur stockee
            p.key = pValNew
            p.S.key = pValNew
            p.S.id = s.id

    if p.player == 2 and (p.id not in goal):  # Noeud max: on l ajoute a ses possibilites de chemin
        p.S.insertion(Successor(s.id, pValNew))


def block_max(s,Q):

    s.S.delete_min()
    new_min = s.S.read_min()

    Q.heap_increase_key(s.index, new_min.key, True)
    s.nbrSucc -= 1

    # vu que pas besoin de retrouver les strategies optimales dans mon cas, je ne stocke pas les arcs deja bloques

def convertPred2NbrSucc(pred):

    """
    A partir de la liste des predecesseurs calcule le nombre de successeurs de chaque noeud.

    """
    nbrSucc = [0]*len(pred)
    for i in range(0,len(pred)):
        pred_i = pred[i]
        for j in range(0, len(pred_i)):
            (v, w) = pred[i][j]
            nbrSucc[v] += 1

    return nbrSucc



def graph_transformer(graph):
    """
    A partir des sommets "normaux" , construit les sommets que nous utiliserons dans l algorithme de Dijkstra

    """

    vertices = graph.vertex
    newVertices = [0]*len(vertices)

    nbrSucc = convertPred2NbrSucc(graph.pred)

    for i in range(0, len(vertices)):
        oldVert = vertices[i]
        if oldVert.player == 1:  # Noeud du joueur Min

            dijkVert = VertexDijkPlayerMin(oldVert.id)

        else:  # Noeud du joueur Max, il faut aussi recuperer son nombre de sucesseurs
            dijkVert = VertexDijkPlayerMax(oldVert.id, nbrSucc[oldVert.id])

        newVertices[i] = dijkVert

    return Graph(newVertices,graph.mat, graph.pred, graph.succ)


def dijkstraMinMax(graph, goal):

    dijk_graph = graph_transformer(graph)
    Q = initQ(dijk_graph, goal)
    initS(dijk_graph, goal)

    T = set()

    while not(Q.is_empty()):

        min = Q.read_min()

        vertex_min_id = min.id
        val_min = dijk_graph.vertex[vertex_min_id].key

        if val_min == float("infinity"):
            u = Q.delete_min(True)
            T.add(u)

        else:
            if(min.player == 1 or min.id in goal) or (min.nbrSucc == 1):
                s = Q.delete_min(True)
                T.add(s)

                list_pred = dijk_graph.pred[s.id]
                for i in range(0,len(list_pred)):
                    (pred, w) = list_pred[i]
                    relaxation(dijk_graph.vertex[pred], s, w, Q, goal)
            else:
                block_max(min,Q)

    return T


def print_result(T, goal):

    print " -----------------------"

    for i in T:

        print "noeud :", i.id, " valeur : ", i.key

    # TODO: reste a gerer les noeuds qui pointent vers qqc de None
    """
    print "la strategie a mettre en oeuvre"

    for i in T:
        if i.player == 1 or i.id in goal:
            print "v"+str(i.id)," --> v"+str(i.S.id)
        else :
            print "v"+str(i.id)," --> v"+str(i.S.read_min().id)
    """
















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

    vnew = 0
    tas.heap_decrease_key(v3.index, vnew, True)

    print tas
    print "index v3", v3.index
    print "index v1", v1.index
    print "index v2", v2.index
    print "index v4", v4.index

    tas.heap_increase_key(v1.index, 4, True)
    print tas
    print "index v3", v3.index
    print "index v1", v1.index
    print "index v2", v2.index
    print "index v4", v4.index

def test2():
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
    pred7 = [(5, 1)]

    list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7]

    graph = Graph(vertices, None,list_pred, None)
    goal = set([0])

    T = dijkstraMinMax(graph, goal)
    print_result(T, goal)

def test_index_mis_a_jour():

    tas = MinHeap()

    v0 = VertexDijk(0, 1, 42)
    tas.insertion(v0, True)

    v1 = VertexDijk(1, 1, 3)
    tas.insertion(v1, True)

    v2 = VertexDijk(2, 1, 50)
    tas.insertion(v2, True)

    tas.delete_min(True)

    v4 = VertexDijk(4, 1, 43)

    tas.insertion(v4, True)


    print tas
    print "index de V0 ", v0.index
    print "index de V1", v1.index
    print "index de V2", v2.index
    print "index de V4", v4.index

def insertion_tas_sans_index():

    tas1 = MinHeap()
    tas2 = MinHeap()

    tas1.tab = [42]

    print tas1
    print tas2

   # super test sur un jeu maintenant

    v0 = VertexDijkPlayerMax(0,0)
    print v0.S is None

    pred = [[]]
    graph = Graph([v0], None, pred, None)
    initS(graph, set([1]))
    print v0.S.tab
    print v0.S


def test3(): # comportement bizarre avec ou sans none , si je mets tab = []
    heap = MinHeap([1, 2, 3], 3)
    heap.insertion(4)
    heap2 = MinHeap()
    print heap
    print heap2.tab

    v0 = VertexDijkPlayerMax(0, 0)
    print v0.S is None
    v1 = VertexDijkPlayerMax(1, 0)
    print v1.S is None

    pred = [[],[]]
    graph = Graph([v0, v1], None, pred, None)
    initS(graph, set([2]))
    print v0.S.tab
    print v0.S
    v0.S.insertion(Successor(2, 42))
    print v0.S.tab
    print v0.S
    print v1.S.tab
    print v1.S

if __name__ == "__main__":
    test3()