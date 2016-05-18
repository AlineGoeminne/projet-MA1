
import random
import heapq

from DijkstraMinMax import dijkstraMinMax
from DijkstraMinMax import VertexDijkPlayerMax
from DijkstraMinMax import VertexDijkPlayerMin
from DijkstraMinMax import convertPred2NbrSucc
from DijkstraMinMax import get_all_values
from DijkstraMinMax import get_succ_in_opti_strat

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

    def __init__(self, vertex, mat , pred, succ, max_weight=None):

        self.vertex = vertex

        self.mat = mat
        self.pred = pred
        self.succ = succ
        self.max_weight= max_weight

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

    @staticmethod
    def get_weight_succ(current, succ, list_succ):
        succ_current = list_succ[current.id]
        weight = None
        for i in succ_current:
            (_succ, w) = i
            if _succ == succ.id:
                weight = w
        return weight



# ----------------
# Classe utile pour encapsuler les objets pour la recherche best_first_search
# -----------------

class Node(object):

    def __init__(self, current, parent, eps, cost, value = float("infinity")):

        self.current = current
        self.parent = parent
        self.eps = eps
        self.cost = cost
        self.value = value


    def __eq__(self, other):

        return self.value == other

    def __ge__(self, other):
        return self.value >= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __lt__(self, other):
        return self.value < other.value



# ------------------
# Class ReachabilityGame
# ------------------


class ReachabilityGame(object):

    def __init__(self, player, graph, init, goal, partition, dic):

        self.player = player   # nbr de joueurs
        self.graph = graph     # graphe represantant le jeu
        self.init = init       # noeud initial du jeu
        self.goal = goal       # tab avec pour chaque joueur ensemble des noeuds objectifs
        self.part = partition  # tab avec pour chaque joueur ensemble des noeuds lui appartenant
        self.id_to_player = dic #dictionnaire permettant a partir de l ID d un noeud de retrouver son joueur

    def get_vertex_player(self, id_player):

        return self.partition[id_player-1]

    def get_goal_player(self, id_player):
        return self.goal[id_player-1]

    def is_a_goal(self, v):

        # TODO : ici je considere que les ensembles objectifs sont disjoints

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
    def path_vertex_to_path_index(path):
        res = []
        for i in path:
            res.append(i.id)
        return res

    @staticmethod
    def generate_vertex_player_uniform(nbr_player, nbr_vertex):

        player_univers = range(1,nbr_player+1)
        vertex = []
        partition = []
        id_to_player = {}

        for i in range(0, nbr_player):
            partition.append(set())

        for i in range(0, nbr_vertex):
            player = random.choice(player_univers)
            v = Vertex(i, player)
            vertex.append(v)
            id_to_player[i] = player
            partition[player-1].add(i)

        return vertex, partition, id_to_player

    @staticmethod
    def generate_game(nbr_player, nbr_vertex, init, goal, a, b):

        pred = Graph.generate_complete_graph(nbr_vertex, "pred", a, b)
        succ = Graph.list_pred_to_list_succ(pred)
        (vertex, partition, dic) = ReachabilityGame.generate_vertex_player_uniform(nbr_player, nbr_vertex)

        graph = Graph(vertex, None, pred, succ)

        return ReachabilityGame(nbr_player, graph, graph.vertex[init], goal, partition, dic)




    def random_path(self, length):
        """
        A partir d un jeu, construit un chemin de longueur lenght
        """

        path = [self.init]

        current = self.init
        for i in range(0,length - 1):
            list_succ_current = self.graph.succ[current.id]
            (succ, weight) = random.choice(list_succ_current)
            current = Vertex(succ, self.id_to_player[succ])
            path.append(current)
        return path




    def test_random_path(self, nbr, length):

        """
        A partir d'un jeu, construit nbr chemins de longueur length et retient les EN
        """
        en_path = []
        coalitions = {}
        for i in range(0, nbr):
            new_path = self.random_path(length)
            (is_Nash, coalitions) = self.is_a_Nash_equilibrium(new_path, coalitions)
            if is_Nash:
                en_path.append(new_path)

        return en_path




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


    @staticmethod
    def same_paths(path1, path2):
        """
        Teste si deux chemins sont les memes
        """
        if id(path1) == id(path2):
            return True

        if len(path1) != len(path2):
            return False

        for i in range(0,len(path1)):
            if path1[i].id != path2[i].id:
                return False
        return True

    def parcours_d_arbre(self, deep):

        path_init = [self.init]
        deep = deep - 1
        coalitions = {}
        reach_player = []
        result = []

        res =  self.parcours_d_arbre_aux(path_init, deep, coalitions, reach_player, result)
        return res
    def parcours_d_arbre_aux(self, path, deep, coalitions, reach_player, result):

        """
        Fonction auxiliaire permettant de parcourir un arbre sur une hauteur de "deep"

        @param path: le chemin courant
        @param deep: la profondeur maximale de parcourt de l'arbre
        @param coalitions: dictionnaire (joueur -> tableau de valeurs)
        @param cost : pour chaque joueur le cout du chemin
        @param eps: cout du prefixe du chemin
        @reach_player: ensemble des joueur ayant atteint leur objectif
        @param result: ensemble des equilibres de Nash retournes

        """

        if deep == 0:
            (nash, coalitions) = self. is_a_Nash_equilibrium(path,coalitions)
            if nash:
                result.append(path)
            return result


        else:
            last_vertex = path[len(path) - 1]
            succ_current = self.graph.succ[last_vertex.id]
            for i in range(0, len(succ_current)):

                (succ, w) = succ_current[i]
                succ_vertex = self.graph.vertex[succ]
                new_path = []
                new_path[0:len(path)] = path
                new_path.append(succ_vertex)
                (goal, player_new_goal) = self.is_a_goal(succ_vertex)
                if goal and player_new_goal not in reach_player:
                    _reach_player = []
                    _reach_player[0:len(reach_player)] = reach_player
                    _reach_player.append(player_new_goal)
                    (nash, coalitions) = self.is_a_Nash_equilibrium_one_player(new_path, player_new_goal, coalitions)
                    if nash:
                        if len(_reach_player) == self.player:
                            result.append(new_path)
                            return result
                        else:
                            self.parcours_d_arbre_aux(new_path, deep - 1, coalitions, _reach_player, result)
                    else:
                        return result
                else:
                    self.parcours_d_arbre_aux(new_path, deep - 1 , coalitions, reach_player, result)

            return result



    #test d'un parcours d'arbre un peu plus intelligent

    def best_first_search(self):
        # todo

        pass

    def generate_successor(self, current, border, heuristic):

        path = current.current # on recupere le veritable noeud courant

        # on ne continue l'exploration que si on a pas atteint la longueur maximale de chemin sur lequel on veut tester
        if len(path) < (self.player +1) * self.graph.max_weight * len(self.graph.vertex):
            last_vertex = path[-1]

            list_succ_current = self.graph.succ[last_vertex.id]

            for i in range(0, len(list_succ_current)):

                cost = current.cost
                (succ, w) = list_succ_current[i]
                epsilon = current.parent.eps + w

                # on regarde si un joueur a atteint son objectif pour la premiere fois

                succ_vertex = self.graph.vertex[succ]
                (goal, player) = self.is_a_goal(succ_vertex)

                if goal and player not in current.cost:

                    cost[player] = epsilon

                    new_path = []
                    new_path[0:len(path)] = path
                    new_path.append(succ_vertex)
                    # on teste si jusque la, pour ce joueur la, l outcome correspond a un EN
                    print "player", player
                    (nash, coalitions) = self.is_a_Nash_equilibrium_one_player(path, player)

                    if nash:
                        value = heuristic(cost, self.player, self.graph.max_weight, len(self.graph.vertex))
                        new = Node(new_path, current, epsilon, cost, value)
                        heapq.heappush(border,new)

                    #si on rompt deja le critere d'etre un en, alors ca en sert a rien de continuer sur ce chemin
                else:
                    # on a pas atteint de nouveau objectif, on continue l exploration

                    value = heuristic(cost, self.player, self.graph.max_weight, len(self.graph.vertex))
                    new = Node(new_path, current, epsilon, cost, value)
                    heapq.heappush(border, new)


    @staticmethod
    def heuristic(cost, nb_player, max_weight, nb_vertex):

        nb_reached = len(cost)
        max_length_path = (nb_player + 1) * nb_vertex

        value = 0

        if nb_reached == 0:
            return float("infinity")
        else:
            keys = cost.keys()
            for p in keys:
                value += cost[p]
            penality = (nb_player - nb_reached)*max_weight * max_length_path
            return value + penality





    def cost_for_all_players(self, path):

        cost = [float("infinity")] * self.player
        weight = 0

        reach_goals_player = set()
        reach_goals = set()
        (goal, player) = self.is_a_goal(path[0])
        if goal:
            cost[player-1] = weight
            reach_goals_player.add(player)
            reach_goals.add(path[0].id)

        for i in range(1, len(path)):

            v = path[i]
            pred_v = path[i-1]
            weight += Graph.get_weight_pred(v, pred_v, self.graph.pred)

            # on teste si on ne vient pas d atteindre un nouvel etat objectif
            if v.id not in reach_goals:
                (goal, player) = self.is_a_goal(v)
                if goal and player not in reach_goals_player:
                    cost[player -1] = weight
                    reach_goals_player.add(player)
                    reach_goals.add(v.id)

        return cost

    def cost_for_one_player(self, path, player):

        cost = float("infinity")
        weight = 0
        goal_player = self.goal[player-1]

        if path[0].id in goal_player:
            cost = 0
            return cost

        for i in range(1, len(path)):

            v = path[i]
            pred_v = path[i - 1]
            weight += Graph.get_weight_pred(v, pred_v, self.graph.pred)

            # on teste si on ne vient pas d atteindre un objectif
            if v.id in goal_player:
                return weight

        return cost








    @staticmethod
    def respect_property(val, epsilon, cost):
        if val + epsilon >= cost:
            return True# on respecte les conditions de la propriete pour etre un EN

        else:
            return False# on ne respcte pas la condition pour au moins un noeud -> pas un EN


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
        while nash and ind < len(path):

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
                tab_result = get_all_values(result_dijk_min_max)
                coalitions[current.player] = tab_result

                val = coalitions[current.player][current.id]

                if ReachabilityGame.respect_property(val, epsilon, path_cost[current.player - 1]):
                    ind += 1  # on respecte les conditions de la propriete pour etre un EN


                else:
                    nash = False  # on ne respcte pas la condition pour au moins un noeud -> pas un EN


        return nash, coalitions

    def is_a_Nash_equilibrium_one_player(self, path, player, coalitions=None):

        """
        A partir d un chemin (path), determine si la condition d'etre un EN est respectue sur le chemin "path" pour
        un certain joueur "player".
        Coalition contient, si elles ont deja ete calculees, les valeurs des noeuds pour les jeux de coalition

        """
        path_cost = self.cost_for_one_player(path, player)  # calcule les couts de tous les joueurs
        if coalitions is None: coalitions = {}
        if player in coalitions:
            values_player = coalitions[player]
        else:
            graph_min_max = ReachabilityGame.graph_transformer(self.graph, player)
            result_dijk_min_max = dijkstraMinMax(graph_min_max, self.goal[player - 1])
            values_player = get_all_values(result_dijk_min_max)
            coalitions[player] = values_player

        epsilon = 0  # poids du chemin jusqu'au noeud courant
        nash = True  # le chemin est un equilibre de Nash
        ind = 0  # indice du noeud courant dans le chemin

        current = path[ind]
        while nash and ind < len(path):

            if ind != 0:
                pred = current
                current = path[ind]
                epsilon += Graph.get_weight_pred(current, pred, self.graph.pred)

            if current.player == player :
                val = values_player[current.id]

                if ReachabilityGame.respect_property(val, epsilon, path_cost):
                    ind += 1  # on respecte les conditions de la propriete pour etre un EN
                else:
                    nash = False  # on ne respecte pas la condition pour au moins un noeud -> pas un EN
            else:
                ind += 1
        return nash, coalitions

    def get_info_path(self, path):

        player_goal = []

        cost = self.cost_for_all_players(path)

        for i in range(len(cost)):

            if cost[i] != float("infinity"):
                player_goal.append(i+1)

        return (cost, player_goal)


    def filter_best_result(self, result):

        best_cost = None
        value_best_cost = float("infinity")
        best_reached = None
        value_best_reached = 0
        partial_value_best_cost = float("infinity")

        for res in result:
            (cost, player_goal) = self.get_info_path(res)

            sum_cost = sum(cost)

            if sum_cost != float("infinity") and value_best_cost > sum_cost:
                best_cost = res
                value_best_cost = sum_cost
            else:
                nb_player_goal = len(player_goal)
                if value_best_reached < nb_player_goal:
                    best_reached = res
                    value_best_reached = best_reached
                else:
                    partial_cost = ReachabilityGame.compute_partial_cost(player_goal, cost)
                    if value_best_reached == nb_player_goal and partial_value_best_cost > partial_cost:
                        partial_value_best_cost = partial_cost
                        best_reached = res


        if value_best_cost != float("infinity"):
            return best_cost
        else:
            return best_reached

    @staticmethod
    def compute_partial_cost(player_reached, all_cost):

        cost = 0

        for p in player_reached:

            cost += all_cost[p-1]
        return cost









    # *******
    # afficheurs d'informations
    # *******
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

    game = ReachabilityGame(2, graph, init, goals, None, None)

    path1 = [v1, v2, v3, v4, v3, v4, v3, v4] # EN ou J1 voit son objectif
    path2 = [v1, v2, v3, v0, v1, v0 , v1, v0, v1 ] # En ou les deux joueurs voient leur objectif
    path3 = [v1, v2, v4, v2, v4, v2, v4, v2, v4, v2, v4] # pas un EN

    print game.is_a_Nash_equilibrium(path1)
    print game.is_a_Nash_equilibrium(path2)
    print game.is_a_Nash_equilibrium(path3)






if __name__ == '__main__':

    #test_generate_game()

    #test_path_cost()

    test_nash_equilibrium()