
import random

class ArenaError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


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
        self.succ = succ


class ReachabilityGame(object):

    def __init__(self, player, graph, goal, partition):

        self.player = player
        self.graph = graph
        self.goal = goal
        self.part = partition


def generate_complete_graph(nbr_edge, type, a, b):

    """
    Genere un graphe complet dont les poids sont generes aleatoirement dans l'intervalle [a,b]

    :param nbr_edge : le nombre d'arc du graphe
    :type nbr_edge: int

    :param type: le type de representation du graphe que l'on souhaite (liste de predecesseurs, liste de sucesseurs,
    matrice d'adjacence
    :type  type : string

    :param a : plus petite valeur que peut prendre le poids d'un arc
    :type a : int

    :param b : la plus grande valeur que peut prendre le poids d'un arc
    :type b : int

    """
    res = []

    if type == "matrix":

        for i in range(0, nbr_edge):

            line = []
            for j in range(0, nbr_edge):

                weight = random.randint(a, b)
                weight_decimal = random.random()

                if i != j:
                    line.append(weight + weight_decimal)
                else:
                    line.append(0)

            res.append(line)
        return res
    if type == "succ":

        for i in range(0, nbr_edge):
            new_succ = []
            for j in range(0, nbr_edge):
                weight = random.randint(a, b)
                weight_decimal = random.random()

                if i != j:
                    new_succ.append((j, weight + weight_decimal))
                
            res.append(new_succ)
        return res

    if type == "pred":
        for i in range(0, nbr_edge):
            new_pred = []
            for j in range(0, nbr_edge):
                weight = random.randint(a, b)
                weight_decimal = random.random()

                if i != j:
                    new_pred.append((j, weight + weight_decimal))

            res.append(new_pred)
        return res


    else:
        # TODO : est-ce bien un ValueError qui est adequat?
        raise ValueError()



def matrix_to_list_succ(mat):

    "Transforme une matrice d'adjancence en liste de successeurs"

    if len(mat) != len(mat[0]):
        raise ArenaError("La matrice d'adjacence n'est pas carree")

    nbr_edge = len(mat)

    list_succ = []

    for i in range(0, nbr_edge):
        new_succ = []

        for j in range(0, nbr_edge):

            weight = mat[i][j]
            if weight != 0:
                new_succ.append((j, weight))

        list_succ.append(new_succ)

    return list_succ

def matrix_to_list_pred(mat):

    "Transforme une matrice d'adjancence en liste de predecesseurs"

    if len(mat) != len(mat[0]):
        raise ArenaError("La matrice d'adjacence n'est pas carree")

    nbr_edge = len(mat)

    list_pred = []

    for j in range(0, nbr_edge):
        new_pred = []

        for i in range(0, nbr_edge):

            weight = mat[i][j]
            if weight != 0:
                new_pred.append((i, weight))

        list_pred.append(new_pred)

    return list_pred

if __name__ == '__main__':

    #res = generate_complete_graph(4, "matrix", 1, 100)

    #print res

    #print matrix_to_list_succ(res)
    #print matrix_to_list_pred(res)

    res = generate_complete_graph(5, "matrix", 0, 10)
    print res

    res1 = generate_complete_graph(5, "pred", 0, 10)
    print res1

    res2 = generate_complete_graph(5, "succ", 0 , 10)
    print res2

