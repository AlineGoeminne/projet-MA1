from MinHeap import MinHeap
import copy

"""
Ce module permet de calculer les valeurs des reachability-price games.

dijkstraMinMax : /!\ uniquement si pas de poids negatifs ET si pas de tuple pour le poids des arcs
compute_value_with_negative_weight : peut etre utilises si les poids sont negatifs ET gere la presence de tuple sur les
arcs

"""


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





class Successor(object):
    """

    Cette classe represente les objets qui seront stockes dans chaque tas S de chaque vertex.
    Pour un certain vertex v, on possede un tas S composes d objets successeurs.
    Un objet successeur possede un attribut ID qui represente un successeur s de v et un attribut key qui represente la valeur
    si en v on passe par s.

    """

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

def convertSucc2NbrSucc(succ):
    """
    A partir de la liste des successuers retourne le nombre de successeurs de chaque noeud.
    """

    nbrSucc = [0] * len(succ)
    for i in range(0,len(succ)):
        nbrSucc[i] = len(succ[i])

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

    """
        Pour chaque noeud du graphe determine quel successeur est choisi selon les strategies optimales des joueurs Min
        et Max determinees par l algorithme dijkstraMinMax.

        :param T: resultat de l algorithme dijkstraMinMax
        :param goal: liste des ensembles objectifs
        :param succ: liste de listes de successeurs pour chaque noeud du graphe du jeu
    """

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



def compute_value_with_negative_weight(graph, goal, tuple = False, compo=0):

    """
        Etant donne le graphe du jeu et les ensembles objectifs pour chaque joueur, calcule la valeur de chaque noeud.
        :param graph: graphe du jeu donne par sa matrice de successeurs
        :param goal: liste des ensembles objectifs de chaque joueur
        :param tuple_: boolean permettant de savoir s'il y a des tuples pour ponderer les arcs
        :param compo : precise, si tuple_ est True, quelle composante on considere pour le calcul des valeurs

        :return (tab_value, min_1, min_2, max): tab_value -> tableau des valeurs de chaque noeud, min_1 et min_2 -> per-
        mettent de retrouver la strategie optimale pour le joueur Min, max -> permet de retrouver la strategie optimale pr
        le joueur Max

    """

    V = len(graph.vertex)

    if tuple:
        W = graph.max_weight[compo]
    else:
        W = graph.max_weight

    tab_value = [float("infinity")] * V
    min_1 = {}
    min_2 = {}
    max = {}

    for t in goal:
        tab_value[t] = 0

    iter = 0
    old_tab_value = []
    while(old_tab_value != tab_value):
        #print tab_value
        old_tab_value = copy.deepcopy(tab_value)
        iter +=1

        for v in graph.vertex:

            if not (v.id in goal):



                if v.player == 1: # on cherche a minimiser
                    compute_min = min_succ_value(graph.succ[v.id], old_tab_value,tuple,compo)
                    tab_value[v.id] = compute_min[0]
                    if tab_value[v.id] != old_tab_value[v.id]:
                        min_1[v.id] = compute_min[1]
                        if old_tab_value[v.id] == float("infinity"):
                            min_2[v.id] = compute_min[1]



                else: # on cherche a maximiser
                    compute_max = max_succ_value(graph.succ[v.id], old_tab_value,tuple,compo)
                    tab_value[v.id] = compute_max[0]
                    max[v.id] = compute_max[1]

                if tab_value[v.id] < -(V - 1) * W:
                    tab_value[v.id] = -float("infinity")



    return (tab_value, min_1, min_2, max)








def min_succ_value(succ, tab_value, tuple=False, compo=0):

    min = float("infinity")
    arg_min = None

    for p in succ:

        temp = p[1][compo] if tuple else p[1]
        temp += tab_value[p[0]]

        if temp < min:
            min = temp
            arg_min = p[0]
    return (min, arg_min)

def max_succ_value(succ, tab_value, tuple=False, compo=0):


    max = - float("infinity")
    arg_max = None

    for p in succ:

        temp = p[1][compo] if tuple else p[1]
        temp += tab_value[p[0]]
        if temp > max:
            max = temp
            arg_max = p[0]
    return (max , arg_max)

