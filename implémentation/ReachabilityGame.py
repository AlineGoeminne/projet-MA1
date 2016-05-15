
import random

from DijkstraMinMax import dijkstraMinMax
from DijkstraMinMax import VertexDijkPlayerMax
from DijkstraMinMax import VertexDijkPlayerMin
from DijkstraMinMax import convertPred2NbrSucc
from DijkstraMinMax import set_to_tab_result

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

    @staticmethod
    def generate_complete_graph(nbr_vertex, type, a, b):

        """
        Genere un graphe complet dont les poids sont generes aleatoirement dans l'intervalle [a,b]

        :param nbr_vertex : le nombre d'arc du graphe
        :type nbr_vertex: int

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

            for i in range(0, nbr_vertex):

                line = []
                for j in range(0, nbr_vertex):

                    weight = random.randint(a, b)
                    weight_decimal = random.random()

                    if i != j:
                        line.append(weight + weight_decimal)
                    else:
                        line.append(0)

                res.append(line)
            return res
        if type == "succ":

            for i in range(0, nbr_vertex):
                new_succ = []
                for j in range(0, nbr_vertex):
                    weight = random.randint(a, b)
                    weight_decimal = random.random()

                    if i != j:
                        new_succ.append((j, weight + weight_decimal))

                res.append(new_succ)
            return res

        if type == "pred":
            for i in range(0, nbr_vertex):
                new_pred = []
                for j in range(0, nbr_vertex):
                    weight = random.randint(a, b)
                    weight_decimal = random.random()

                    if i != j:
                        new_pred.append((j, weight + weight_decimal))

                res.append(new_pred)
            return res


        else:
            # TODO : est-ce bien un ValueError qui est adequat?
            raise ValueError()


    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def list_pred_to_list_succ(pred):
        result = []
        for i in range(0, len(pred)):
            result.append([])
        for i in range(0, len(pred)):
            pred_i = pred[i]
            for j in range(0, len(pred_i)):
                (v, w) = pred[i][j]
                result[v].append((i,w))

        return result

    @staticmethod
    def get_weight_pred(current, pred, list_pred):

        """ Etant donne un noeud courant, un certain predecesseur et la liste des predecesseurs, donne le poids entre
        les deux noeuds consideres"""

        pred_current = list_pred[current.id]
        weight = None
        for i in pred_current:
            (_pred, w) = i
            if _pred == pred.id:
                weight = w

        return weight


# ------------------
# Class ReachabilityGame
# ------------------


class ReachabilityGame(object):

    def __init__(self, player, graph, init, goal, partition):

        self.player = player   # nbr de joueurs
        self.graph = graph     # graphe represantant le jeu
        self.init = init       # noeud initial du jeu
        self.goal = goal       # tab avec pour chaque joueur ensemble des noeuds objectifs
        self.part = partition  # tab avec pour chaque joueur ensemble des noeuds lui appartenant

    def get_vertex_player(self, id_player):

        return self.partition[id_player-1]

    def get_goal_player(self, id_player):
        return self.goal[id_player-1]

    def is_a_goal(self, v):

        """
        Teste si le noeud teste correspond a l'objectif recherche.
        Si oui renvoie (True, player) ou player est le joueur ayant atteint son objectif
        Si non renvoie (False, None)
        """

        for player in range(0, len(self.goal)):

            goal_set = self.goal[player]

            if v.id in goal_set:
                return True , player+1

        return False, None


    @staticmethod
    def generate_vertex_player_uniform(nbr_player, nbr_vertex):

        player_univers = range(1,nbr_player+1)
        vertex = []
        partition = []

        for i in range(0, nbr_player):
            partition.append(set())

        for i in range(0, nbr_vertex):
            player = random.choice(player_univers)
            v = Vertex(i, player)
            vertex.append(v)
            partition[player-1].add(i)

        return vertex, partition

    @staticmethod
    def generate_game(nbr_player, nbr_vertex, init, goal, a, b):

        pred = Graph.generate_complete_graph(nbr_vertex, "pred", a, b)
        succ = Graph.list_pred_to_list_succ(pred)
        (vertex, partition) = ReachabilityGame.generate_vertex_player_uniform(nbr_player, nbr_vertex)

        graph = Graph(vertex, None, pred, succ)

        return ReachabilityGame(nbr_player, graph, init, goal, partition)

    @staticmethod
    def graph_transformer(graph, min_player):

        """
        A partir d'un graphe, modelise le graphe du jeu min-max tel que le joueur voulant minimiser est min_player

        """

        vertices = graph.vertex
        newVertices = [0] * len(vertices)

        nbrSucc = convertPred2NbrSucc(graph.pred)

        for i in range(0, len(vertices)):
            oldVert = vertices[i]
            if oldVert.player == min_player:  # Noeud du joueur Min

                dijkVert = VertexDijkPlayerMin(oldVert.id)

            else:  # Noeud du joueur Max, il faut aussi recuperer son nombre de sucesseurs
                dijkVert = VertexDijkPlayerMax(oldVert.id, nbrSucc[oldVert.id])

            newVertices[i] = dijkVert

        return Graph(newVertices, graph.mat, graph.pred, graph.succ)

    def cost_for_all_players(self, path):

        cost = [float("infinity")] * self.player
        weight = 0

        reach_goals = set()
        (goal, player) = self.is_a_goal(path[0])
        if goal:
            cost[player-1] = weight

        for i in range(1,len(path)):

            v = path[i]
            pred_v = path[i-1]
            weight += Graph.get_weight_pred(v, pred_v, self.graph.pred)

            # on teste si on ne vient pas d atteindre un etat objectif
            if v.id not in reach_goals:
                (goal, player) = self.is_a_goal(v)
                if goal:
                    cost[player -1] = weight
                    reach_goals.add(v.id)

        return cost


    @staticmethod
    def respect_property(val, epsilon, cost):
        if val + epsilon >= cost:
            return True# on respecte les conditions de la propriete pour etre un EN

        else:
            return False # on ne respcte pas la condition pour au moins un noeud -> pas un EN


    def is_a_Nash_equilibrium(self, path, coalitions = None):

        """
        A partir d un chemin (path), determine s il sagit de l outcome d un EN.
        Coalition contient, si elles ont deja ete calculees, les valeurs des noeuds pour les jeux de coalition

        """

        path_cost = self.cost_for_all_players(path) #calcule les couts de tous les joueurs

        if coalitions is None: coalitions = {}
        epsilon = 0 # poids du chemin jusqu'au noeud courant
        nash = True # le chemin est un equilibre de Nash
        ind = 0 # indice du noeud courant dans le chemin

        current = path[ind]
        pred = None
        # dans un premier temps supposons que chaque element du path est un noeud (id,player)
        while(nash and ind < len(path)):
            if ind != 0:
                pred = current
                current = path[ind]
                epsilon += Graph.get_weight_pred(current, pred, self.graph.pred)
            if current.player in coalitions:  #on a deja calcule les valeurs du jeu ou player joue contre la collation Pi\{player}
                val = coalitions[current.player][current.id]

                if ReachabilityGame.respect_property(val, epsilon, path_cost[current.player - 1]):
                    ind += 1 # on respecte les conditions de la propriete pour etre un EN

                else:
                    nash = False # on ne respcte pas la condition pour au moins un noeud -> pas un EN

            else:  # il faut au prealable calculer les valeurs du jeu min-max associe

                graph_min_max = ReachabilityGame.graph_transformer(self.graph, current.player)
                result_dijk_min_max = dijkstraMinMax(graph_min_max, self.goal[current.player - 1])
                tab_result = set_to_tab_result(result_dijk_min_max)
                coalitions[current.player] = tab_result

                val = coalitions[current.player][current.id]
                if ReachabilityGame.respect_property(val, epsilon, path_cost[current.player - 1]):
                    ind += 1  # on respecte les conditions de la propriete pour etre un EN

                else:
                    nash = False  # on ne respcte pas la condition pour au moins un noeud -> pas un EN

        return nash











    def print_partition(self):

        partition = self.part

        for i in range(0,len(partition)):
            print "Noeud(s) du joueur ", i+1,": ", partition[i]

    def print_goal(self):

        goal = self.goal

        for i in range(0, len(goal)):
            print "Objectif(s) du joueur ", i+1, ": ", goal[i]


    def print_sucesseur(self):
        succ = self.graph.succ
        for i in range(0,len(succ)):
            print "succ de v",i," : ", succ[i]

    def print_info_game(self):

        print "Nombre de joueurs :", self.player
        print "Nombre de noeuds de l'arene :", len(self.graph.pred)
        print "Parition "
        self.print_partition()
        print "*******"
        print "Objectif(s) :"
        self.print_goal()
        print "*******"

        print "noeud initial : v"+repr(self.init)
        print "*******"

        print "Liste des sucesseurs :"
        self.print_sucesseur()




def test_pred_to_succ():
    pred_0 = [(1, 4), (0, 2)]
    pred_1 = [(2, 4)]
    pred_2 = []

    list_pred = [pred_0, pred_1, pred_2]
    list_succ = Graph.list_pred_to_list_succ(list_pred)
    print list_succ

def test_generate_game():

    game = ReachabilityGame.generate_game(3, 40, 2, [set([2,3]), set([2]), set([5])], 1, 100)
    game.print_info_game()


def test_path_cost():
     succ0 = [(1, 1), (2, 5)]
     succ1 = [(4, 2)]
     succ2 = [(3, 4)]
     succ3 = [(4, 3)]
     succ4 = []

     list_succ = [succ0, succ1, succ2, succ3, succ4]

     list_pred = [[], [(0, 1)], [(0, 5)], [(2, 4)], [(1, 2), (3, 3)]]

     v0 = Vertex(0, 1)
     v1 = Vertex(1, 2)
     v2 = Vertex(2, 1)
     v3 = Vertex(3, 1)
     v4 = Vertex(4, 2)

     vertex = [v0, v1, v2, v3, v4]
     graph = Graph(vertex,None, list_pred, list_succ)

     game = ReachabilityGame(2, graph, 0, [set([3]), set([1])], None)

     path = [v0, v2, v3, v4]

     tab_cost = game.cost_for_all_players(path)

     print tab_cost[0]
     print tab_cost[1]


def test_nash_equilibrium():

    v0 = Vertex(0, 1)
    v1 = Vertex(1, 1)
    v2 = Vertex(2, 2)
    v3 = Vertex(3, 1)
    v4 = Vertex(4, 1)
    vertex = [v0, v1, v2, v3, v4]

    pred0 = [(1, 1), (3, 1)]
    pred1 = [(0, 1)]
    pred2 = [(1, 1), (4, 2)]
    pred3 = [(2, 1), (4, 1)]
    pred4 = [(2, 4), (3, 1)]

    list_pred = [pred0, pred1, pred2, pred3, pred4]
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    graph = Graph(vertex, None, list_pred, list_succ)

    goals = [set([3]), set([0])]
    init = 1

    game = ReachabilityGame(2, graph, init, goals, None)

    path1 = [v1, v2, v3, v4, v3, v4, v3, v4] # EN ou J1 voit son objectif
    path2 = [v1, v2, v3, v0, v1, v0 , v1, v0, v1 ] # En ou les deux joueurs voient leur objectif
    path3 = [v1, v2, v4, v2, v4, v2, v4, v2, v4, v2, v4] # pas un EN

    print game.is_a_Nash_equilibrium(path1)
    #print game.is_a_Nash_equilibrium(path2)
    #print game.is_a_Nash_equilibrium(path3)






if __name__ == '__main__':

    #test_generate_game()

    #test_path_cost()

    test_nash_equilibrium()