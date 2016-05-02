
class Vertex(object):
    """
        Classe representant un sommet d'un graphe, on considere que ces sommets sont entierement determines par leur
        "nom" qui est en fait un entier.
    """
    def __init__(self, id, player):
        self.id = id
        self.player = player

    def __eq__(self, other):

        if id(self) == id(other):
            return True
        else:
            return self.id == other.name


class Graph(object):

    """

    :param vertex : liste des sommets du jeu classe par leur ID
    :type vertex :  list

    :param mat : matrice d adjacence du jeu
    :type mat : tableau |vertex|*|vertex|

    :param pred: liste de predecesseurs. Pour chaque noeud v, on associe une liste de tuples (p,w) ou p est un de ces
    predecesseurs et w est le poid de l'arc (p,v)
    :type pred: liste de listes de tuples

    :param succ: liste de successeurs. Pour chauqe noeud v, on associe une liste de tuples (s,w) ou s est un ces succ
    et w est le poids de l'arc (v,s)
    :type succ: liste de listes de tuples

    """

    def __init__(self, vertex, mat , pred, succ):

        self.vertex = vertex

        self.mat = mat
        self.pred = pred
        self. succ = succ


class ReachabilityGame(object):

    def __init__(self, player, graph, goal, partition):

        self.player = player
        self.graph = graph
        self.goal = goal
        self.part = partition


