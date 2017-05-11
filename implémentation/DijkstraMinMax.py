from MinHeap import MinHeap




class VertexDijk(object):

    def __init__(self, id, player, key=float("infinity"), index=0):

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

    def __init__(self, id, player=1, key=float("infinity"), min = None, index=0 ):

        VertexDijk.__init__(self, id, player, key, index)
        self.S = min


class VertexDijkPlayerMax(VertexDijk):

    def __init__(self, id, nbrSucc, player=2, key = float("infinity"), heap = None, index = 0):

        VertexDijk.__init__(self, id, player, key, index)
        self.S = heap  # represente un tas pour les noeuds du joueur max
        self.nbrSucc = nbrSucc
        self.blocked = set()



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



def initS(graph, goal):

    """
        Initialise pour chaque noeud du graphe la structure de donnee permettant de tenir a jour la valeur future de Val(v)
        :param graph : un graph represente par sa liste de predecesseurs

    """

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



def initQ(graph, goal):

    """
    Initialise le tas Q de l'algorithme DijkstraMinMax
    """

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

    blocked = s.S.read_min()
    s.S.delete_min()
    new_min = s.S.read_min()

    Q.heap_increase_key(s.index, new_min.key, True)
    s.nbrSucc -= 1

    s.blocked.add(blocked.id)

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





def dijkstraMinMax(graph, goal):

    Q = initQ(graph, goal)
    initS(graph, goal)

    T = set()

    while not(Q.is_empty()):

        min = Q.read_min()

        vertex_min_id = min.id
        val_min = graph.vertex[vertex_min_id].key

        if val_min == float("infinity"):
            u = Q.delete_min(True)
            T.add(u)

        else:
            if(min.player == 1 or min.id in goal) or (min.nbrSucc == 1):
                s = Q.delete_min(True)
                T.add(s)

                list_pred = graph.pred[s.id]
                for i in range(0,len(list_pred)):
                    (pred, w) = list_pred[i]
                    relaxation(graph.vertex[pred], s, w, Q, goal)
            else:
                block_max(min,Q)

    return T


def print_result(T, goal, succ):

    print " -----------------------"

    for i in T:

        print "noeud :", i.id, " valeur : ", i.key

    print "la strategie a mettre en oeuvre"
    successors = get_succ_in_opti_strat(T, goal, succ)
    for i in range(0,len(successors)):
        print "v"+str(i), "----> v"+str(successors[i])



def get_succ_in_opti_strat(T, goal, succ):

    successor = [0]* len(T)
    for v in T:

        if v.player == 1 or v.id in goal:

            res = v.S.id
            if res is not None:
                successor[v.id] = v.S.id
            else:
                (res, w) = succ[v.id][0]
                successor[v.id] = res


        else:

            res = v.S.read_min().id

            if res is not None:
                successor[v.id] = res
            else:
                blocked = v.blocked
                list_succ = succ[v.id]

                notFind = True
                index = 0
                while(notFind):
                    (candidate, w) = list_succ[index]
                    if candidate not in blocked: # l'arc n'avait pas ete bloque
                        successor[v.id] = candidate
                        notFind = False
                    index += 1

    return successor

def get_all_values(T):

    """
    A partir de l'ensemble des resultats, reconstruit le tableau ID du noeud -> valeur du noeud
    """
    tab = [0]*len(T)
    for i in T:
        tab[i.id] = i.key

    return tab
