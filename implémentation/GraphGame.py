
import random
import heapq
import copy
import time


from Value import dijkstraMinMax
from Value import VertexDijkPlayerMax
from Value import VertexDijkPlayerMin
from Value import convertPred2NbrSucc
from Value import get_all_values
from Value import print_result
from Value import get_succ_in_opti_strat


class ArenaError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class BestFirstSearchError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class NegativeCircuitError(Exception):
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class GraphError(Exception):
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
            return self.id == other.id

    def __repr__(self):
        str = "v"+repr(self.id)
        return str


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
        self.max_weight = max_weight

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

                    if i != j:
                        line.append(weight)
                    else:
                        line.append(0)

                res.append(line)
            return res
        if type == "succ":

            for i in range(0, nbr_vertex):
                new_succ = []
                for j in range(0, nbr_vertex):
                    weight = random.randint(a, b)


                    if i != j:
                        new_succ.append((j, weight))

                res.append(new_succ)
            return res

        if type == "pred":
            for i in range(0, nbr_vertex):
                new_pred = []
                for j in range(0, nbr_vertex):
                    weight = random.randint(a, b)


                    if i != j:
                        new_pred.append((j, weight))

                res.append(new_pred)
            return res


        else:
            raise ValueError()


    @staticmethod
    def matrix_to_list_succ(mat):

        "Transforme une matrice d'adjacence en liste de successeurs"

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

    def list_succ_to_mat(list_succ):
        mat = []

        for i in range(0, len(list_succ)):
            succ_i = list_succ[i]
            line = [0] * len(list_succ)
            for j in range(0, len(succ_i)):
                (succ,w) = succ_i[j]
                line[succ] = w
            mat.append(line)

        return mat

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

    @staticmethod
    def mat_to_string(mat):
        string = ""
        nb_vertex = len(mat)

        for i in range(0, nb_vertex):
            string = string + str(mat[i]) + "\n"
        return string


    def floyd_warshall(self):

        """ Calcule le plus court chemin entre chaque paire de sommet du graphe."""

        if self.mat == None:
            raise GraphError(" Pas de representation matricielle du graphe")

        M = self.mat
        n = len(M)

        for i in xrange(n):
           M[i][i] = 0

        for k in xrange(n):
            for i in xrange(n):
                for j in xrange(n):
                    M[i][j]= min(M[i][j], M[i][k]+ M[k][j])
                    if i == j and M[i][j] < 0:
                        raise NegativeCircuitError("Il y a un circuit de poids strictement negatif dans le graphe")


        return M






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

    def __repr__(self):
        str = ""+repr(self.current)+" ; "+ repr(self.value)
        return str



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

        return self.part[id_player-1]

    def get_goal_player(self, id_player):
        return self.goal[id_player-1]

    def is_a_goal(self, v):

        # /!\ ici on suppose que les ensembles Goal_i sont disjoints
        """
        Teste si le noeud teste correspond a un etat objectif.
        Si oui renvoie (True, player) ou player est le joueur ayant atteint son objectif
        Si non renvoie (False, None)
        """

        for player in range(0, len(self.goal)):

            goal_set = self.goal[player]

            if v.id in goal_set:
                return True, player + 1

        return False, None



    def compute_max_length(self):

        """
        Calcule la longueur maximale de l outcome d un EN
        """
        sum = 0
        adj = self.graph.succ

        for i in range(0, len(adj)):
            list_succ = adj[i]
            for j in range(0,len(list_succ)):
                (succ,w) = list_succ[j]
                sum = sum + w - 1

        return (self.player + 1)*(len(self.graph.vertex) + sum)

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


        """

        :param nbr_player: nombre de joueurs dans le jeu
        :param nbr_vertex: nombre de sommets du jeu
        :param init: index du sommet initial
        :param goal: objectif pour chaque joueur
        :param a: poids min sur les arcs du graphe
        :param b: poids max sur les arcs du graphe
        :return: un jeu  dont le graphe est genere aleatoirement
        """

        pred = Graph.generate_complete_graph(nbr_vertex,"pred", a, b)
        succ = Graph.list_pred_to_list_succ(pred)
        (vertex, partition) = ReachabilityGame.generate_vertex_player_uniform(nbr_player, nbr_vertex)

        graph = Graph(vertex, None, pred, succ, b)

        return ReachabilityGame(nbr_player, graph, graph.vertex[init], goal, partition)




    def random_path(self, length):

        """
        A partir d un jeu, construit un chemin de longueur lenght
        """

        path = [self.init]

        current = self.init
        for i in range(0,length - 1):
            list_succ_current = self.graph.succ[current.id]
            (succ, weight) = random.choice(list_succ_current)
            current = Vertex(succ, self.graph.vertex[succ].player)
            path.append(current)
        return path




    def test_random_path(self, nbr, length, coalitions = None):

        """
        A partir d'un jeu, construit nbr chemins de longueur length et retient les EN
        """
        en_path = []
        coalitions = {}
        for i in range(0, nbr):
            new_path = self.random_path(length)
            (is_Nash, coalitions) = self.is_a_Nash_equilibrium(new_path, None, coalitions)
            if is_Nash:
                en_path.append(new_path)
        return en_path




    @staticmethod
    def graph_transformer(graph, min_player):

        """
        A partir d'un graphe, modelise le graphe du jeu min-max tel que le joueur voulant minimiser est min_player
        :param grap: un graphe
        :param min_player: le joueur que l on veut transformer en joueur Min
        :return  le graphe du jeu Min Max ou le joueur min_player est le joueur Max, les autres joueurs sont rassembles
        en une coalition

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
    def graph_transformer_real_dijkstra(graph):
        """
         A partir d'un graphe, modelise le graphe du jeu ou tous les noeuds appartiennent au joueur Min.

        """

        vertices = graph.vertex
        newVertices = [0] * len(vertices)

        for i in range(0, len(vertices)):
            oldVert = vertices[i]
            dijkVert = VertexDijkPlayerMin(oldVert.id)
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



    def breadth_first_search(self, first=True, allowed_time= float("infinity")):

        start = time.time()

        max_length = self.compute_max_length()

        if first:
            res = None
        else:
            res =[]

        parent = Node([], None, 0, {})
        initial_node = Node([self.init], parent, 0, {})

        (goal, player) = self.is_a_goal(self.init)

        if goal:
            initial_node.cost[player] = 0

        frontier = []
        frontier.append(initial_node)

        while True and time.time() - start < allowed_time:

            if len(frontier) == 0:
                raise BestFirstSearchError(" Plus d'elements dans la frontiere")

            candidate_node = frontier.pop(0)
            candidate_path = candidate_node.current


            if len(candidate_node.cost) == self.player:


                # Alors on sait qu'il s'agit d'un equilibre de Nash car tous les cost ont ete initialises
                if first:
                    return candidate_path
                else:
                    res.append(candidate_path)

            if len(candidate_path) == max_length:
                # Il se peut que ce soit un equilibre de Nash tel que tous les joueurs n'ont pas vu leur objectif mais
                # tel que la longueur max est atteinte: il faut donc tester pour ceux qui n'ont pas encore atteint leur obj

                (nash, coalition) = self.is_a_Nash_equilibrium(candidate_path, candidate_node.cost.keys())
                if nash:
                    if first:
                        return candidate_path
                    else:
                        res.append(candidate_path)

            else:  # len(candidate_path) < max_length et len(candidate_node.cost) != self.player

                last_vertex = candidate_node.current[-1]
                succ_last_vertex = self.graph.succ[last_vertex.id]
                random.shuffle(succ_last_vertex)

                for i in range(0, len(succ_last_vertex)):

                    (succ, w) = succ_last_vertex[i]

                    epsilon = candidate_node.eps + w

                    new_path = []
                    new_path[0: len(candidate_path)] = candidate_path

                    succ_vertex = self.graph.vertex[succ]
                    new_path.append(succ_vertex)


                    (goal, player) = self.is_a_goal(succ_vertex)

                    if goal and player not in candidate_node.cost:

                        (nash, coalitions) = self.is_a_Nash_equilibrium_one_player(new_path, player)

                        if nash:
                            new_cost = copy.deepcopy(candidate_node.cost)
                            new_cost[player] = epsilon

                            new_node = Node(new_path, candidate_node, epsilon, new_cost)

                            frontier.append(new_node)

                    else:

                        new_node = Node(new_path, candidate_node, epsilon, candidate_node.cost)
                        frontier.append(new_node)

        return res



    @staticmethod
    def find_loop(path, index):
        counter = set([])
        loop = False
        res = None

        while(not loop and index<len(path)):

            vertex = path[index]

            if vertex in counter:
                loop = True
                res = path[0:index+1]

            else:
                counter.add(vertex)

            index += 1

        return res

    # test d'un parcours d'arbre un peu plus intelligent


    def best_first_search(self, heuristic, frontier = None, allowed_time = float("infinity")):

        if heuristic is ReachabilityGame.a_star:
            all_dijk = self.compute_all_dijkstra()
        else:
            all_dijk = None

        start = time.time()

        max_length = self.compute_max_length()

        last_goal_index = 0


        if frontier is None:
            parent = Node([], None, 0, {})
            initial_node = Node([self.init], parent, 0, {})

            (goal, player) = self.is_a_goal(self.init)

            if goal:
                initial_node.cost[player] = 0
            value = heuristic(self, initial_node.cost,0, 1, self.init, all_dijk)
            initial_node.value = value
            frontier = []
            heapq.heappush(frontier, initial_node)

        while True and time.time() - start < allowed_time:

            if len(frontier) == 0:
                raise BestFirstSearchError(" Plus d'elements dans la frontiere")

            candidate_node = heapq.heappop(frontier)
            candidate_path = candidate_node.current

            if len(candidate_node.cost) == self.player:

                #Alors on sait qu'il s'agit d'un equilibre de Nash car tous les cost ont ete initialises
                return candidate_path

            if len(candidate_path) == max_length :
                #Il se peut que ce soit un equilibre de Nash tel que tous les joueurs n'ont pas vu leur objectif mais
                #tel que la longueur max est atteinte: il faut donc tester pour ceux qui n'ont pas encore atteint leur obj

                (nash,coalition) = self.is_a_Nash_equilibrium(candidate_path, candidate_node.cost.keys())
                if nash:

                    return candidate_path

            else: #len(candidate_path) < max_length et len(candidate_node.cost) != self.player

                last_vertex = candidate_node.current[-1]
                succ_last_vertex = self.graph.succ[last_vertex.id]
                random.shuffle(succ_last_vertex)

                for i in range(0, len(succ_last_vertex)):

                    (succ, w) = succ_last_vertex[i]

                    epsilon = candidate_node.eps + w

                    new_path = []
                    new_path[0 : len(candidate_path)] = candidate_path

                    succ_vertex = self.graph.vertex[succ]
                    new_path.append(succ_vertex)

                    (goal, player) = self.is_a_goal(succ_vertex)


                    if goal and player not in candidate_node.cost:

                        (nash, coalitions) = self.is_a_Nash_equilibrium_one_player(new_path, player)

                        if nash:

                            new_cost = copy.deepcopy(candidate_node.cost)
                            new_cost[player] = epsilon
                            value = heuristic(self, new_cost,epsilon, len(new_path), succ_vertex, all_dijk)

                            new_node = Node(new_path, candidate_node, epsilon, new_cost, value)

                            heapq.heappush(frontier, new_node)

                    else:
                        value = heuristic(self, candidate_node.cost,epsilon, len(new_path), succ_vertex, all_dijk)

                        new_node = Node(new_path, candidate_node, epsilon, candidate_node.cost, value)
                        heapq.heappush(frontier, new_node)

        return

    def compute_all_dijkstra(self):
        """
        Calcule la longueur du plus court chemin pour aller d un noeud a un etat objectif et ce pour tout objectif
        """
        result = {}
        union_goal = set([])
        for i in range(0, len(self.goal)):
            union_goal = union_goal.union(self.goal[i])

        for g in union_goal:
            graph_dijk = self.graph_transformer_real_dijkstra(self.graph)
            T = dijkstraMinMax(graph_dijk, {g})
            res = get_all_values(T)
            result[g] = res

        return result

    @staticmethod

    def a_star(game, cost, epsilon, length_path, last_vertex = None, all_dijk = None):


        nbr_player_notreached = game.player - len(cost)
        g_n = 0
        max_length = game.compute_max_length()
        max_weight = game.graph.max_weight
        for i in cost:
            g_n += cost[i]

        g_n = g_n + nbr_player_notreached * epsilon

        h_n = 0

        #on suppose un seul objectif par joueur

        for p in range(1, game.player+1):
            if p not in cost:
                goal_p = game.goal[p-1].pop()
                game.goal[p-1].add(goal_p)
                res = all_dijk[goal_p]
                h_n += min(res[last_vertex.id], max_length * max_weight + 1)

        return g_n + h_n



    @staticmethod
    def heuristic(game, cost, epsilon, length_path, last_vertex = None, all_dijk = None):

        nb_player = game.player
        max_weight = game.graph.max_weight
        nb_reached = len(cost)
        max_length_path = game.compute_max_length()

        value = 0

        if nb_reached == 0:
            return float("infinity")
        else:
            keys = cost.keys()
            for p in keys:
                value += cost[p]
            penality = (nb_player - nb_reached) * max_weight * (max_length_path - length_path) + length_path + (epsilon * (nb_player - nb_reached))
            return value + penality

    @staticmethod
    def short_path_evaluation(game, cost, epsilon, length_path, last_vertex = None, all_dijk = None):


        nb_player = game.player
        max_weight = game.graph.max_weight
        nb_reached = len(cost)
        max_length_path = game.compute_max_length()



        value = 0

        graph_real_dijk = ReachabilityGame.graph_transformer_real_dijkstra(game.graph)
        union_goal = game.goal[0]
        for i in range(1, len(game.goal)):
            union_goal = union_goal.union(game.goal[i])
        T = dijkstraMinMax(graph_real_dijk, union_goal)
        tab_result = get_all_values(T)
        if nb_reached == 0:
            return max_length_path * nb_player* max_weight+ tab_result[last_vertex.id] + epsilon * nb_player
        else:
            keys = cost.keys()
            for p in keys:
                value += cost[p]
            penality = (nb_player - nb_reached) * max_weight * (max_length_path - length_path) + length_path + (epsilon * (nb_player - nb_reached))

            return value + penality + tab_result[last_vertex.id]



    def init_search(self):

        """
        On va tester le chemin du noeud initial -> l'objectif le plus proche, si cela respecte tjs les conditions pour
        etre un EN , on continue la recherche a partir de ce chemin

        """
        graph_dijk = ReachabilityGame.graph_transformer_real_dijkstra(self.graph)

        union_goal = self.goal[0]
        for i in range(1, len(self.goal)):
            union_goal = union_goal.union(self.goal[i])

        T = dijkstraMinMax(graph_dijk, union_goal)
        successor = get_succ_in_opti_strat(T, union_goal, self.graph.succ)

        path = [self.init]
        goal_is_reached = False
        (goal, player) = self.is_a_goal(self.init)
        if goal:
            return path, player
        else:

            while not goal_is_reached:
                succ = self.graph.vertex[successor[path[-1].id]]
                path.append(succ)

                (goal, player) = self.is_a_goal(succ)

                if goal:
                    goal_is_reached = True

            return path, player

    def init_search_goal(self, goal):

        """
        A partir du noeud initial, reconstruit le plus court chemin jusqu'a l objectif goal
        """

        graph_dijk = ReachabilityGame.graph_transformer_real_dijkstra(self.graph)

        goal_id = goal.id
        T = dijkstraMinMax(graph_dijk, {goal_id})
        successor = get_succ_in_opti_strat(T,{goal_id}, self.graph.succ)
        goal_is_reached = False
        path = [self.init]

        if self.init.id == goal_id:
            return path
        else:

            while not goal_is_reached:
                succ = self.graph.vertex[successor[path[-1].id]]
                path.append(succ)

                if succ.id == goal_id:
                    goal_is_reached = True

            return path


    def best_first_search_with_init_path(self, evaluation, allowed_time=float("infinity")):

        (path, player) = self.init_search()

        (nash, coalition) = self.is_a_Nash_equilibrium_one_player(path, player)

        if nash:
            cost = self.cost_for_all_players(path,True)
            epsilon = self.compute_epsilon(path)
            value = evaluation(self, cost, epsilon, len(path), path[-1])
            eps = self.compute_epsilon(path)
            node = Node(path, Node([], None, 0, cost, value), eps, cost, value)
            frontier = [node]

            res = self.best_first_search(evaluation, frontier, allowed_time)
            return res

        else:
            return None

    def best_first_search_with_init_path_both_two(self, evaluation, allowed_time=float("infinity")):

        """
        on teste l initialisation du chemin , en testant un a un le plus court chemin vers un objectif"
        :param evaluation: la fonction d evaluation utilisee pour le best-first search
        :param allowed_time: le temps permis pour faire execution de best-first search
        :return: un equilibre de Nash, si best-first search en trouve un durant le temps imparti; None sinon
        """

        union_goal = self.goal[0]
        for i in range(1, len(self.goal)):
            union_goal = union_goal.union(self.goal[i])

        for i in range(0, len(union_goal)):
            goal = union_goal.pop()
            goal_vertex = self.graph.vertex[goal]
            res = self.best_first_search_with_init_path_both_two_aux(evaluation, goal_vertex , allowed_time)

            if res is not None:
                return res

        return None

    def best_first_search_with_init_path_both_two_aux(self, evaluation, goal, allowed_time=float("infinity")):

        if evaluation is ReachabilityGame.a_star:
            all_dijk = self.compute_all_dijkstra()
        else:
            all_dijk = None

        (is_goal,player) = self.is_a_goal(goal)
        path = self.init_search_goal(goal)

        (nash, coalition) = self.is_a_Nash_equilibrium_one_player(path, player)

        if nash:
            cost = self.cost_for_all_players(path, True)
            epsilon = self.compute_epsilon(path)
            value = evaluation(self, cost, epsilon, len(path), path[-1], all_dijk)
            eps = self.compute_epsilon(path)
            node = Node(path, Node([], None, 0, cost, value), eps, cost, value)
            frontier = [node]

            res = self.best_first_search(evaluation, frontier, allowed_time)
            return res

        else:
            return None


    def compute_epsilon(self, path):

        """
        Calcule le poids d'un chemin
        :param path: un chemin
        :return: poids du chemin path
        """
        eps = 0
        for i in range(1, len(path)):

            v = path[i]
            pred_v = path[i - 1]
            eps += Graph.get_weight_pred(v, pred_v, self.graph.pred)

        return eps






    def cost_for_all_players(self, path, only_reached=False):

        """
        Calcule le cout du chemin pour chaque jouer
        :param: path: un chemin
        :param: only_reached: cette option permet de ne calculer uniquement les couts pour les joueurs qui ont atteint
        leur objectif
        :return  le poids du chemin pour tous les joueurs ou le poids uniquement pour ceux qui ont atteint leur objectif
        si only_reached = True. Ce sous la forme d'un dictionnaire player -> cost

        """

        cost = {}
        weight = 0

        reach_goals_player = set()
        reach_goals = set()
        (goal, player) = self.is_a_goal(path[0])
        if goal:
            cost[player] = weight
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
                    cost[player] = weight
                    reach_goals_player.add(player)
                    reach_goals.add(v.id)

        if (not only_reached):
            for p in range(1, self.player + 1):
                if p not in cost:
                    cost[p] = float("infinity")

        return cost

    def cost_for_one_player(self, path, player):

        """
        Calcule le cout d un chemin pour un certain joueur
        :param path: un chemin
        :param player: un joueur
        :return: cout du chemin path pour le joueur player
        """

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
        """
        Determine si en un sommet donne la propriete d etre un EN est respectee
        :param val: valeur du sommet courant
        :param epsilon: valeur d epsilon en le sommet courant
        :param cost: cout du chemin pour le joueur auquel le sommet courant appartient
        :return: True si la propriete est encore respectee; False sinon
        """

        if val + epsilon >= cost:
            return True

        else:
            return False


    def is_a_Nash_equilibrium(self, path, already_test = None, coalitions = None):

        """
        A partir d un chemin (path), determine s il sagit de l outcome d un EN.
        Already_test contient l'ensemble des joueurs pour lesquels on sait deja que cet outcome est un EN
        Coalition contient, si elles ont deja ete calculees, les valeurs des noeuds pour les jeux de coalition

        """

        path_cost = self.cost_for_all_players(path) #calcule les couts de tous les joueurs

        if coalitions is None:
            coalitions = {}
        if already_test is None:
            already_test = set([])


        epsilon = 0 # poids du chemin jusqu'au noeud courant
        nash = True # le chemin est un equilibre de Nash
        ind = 0 # indice du noeud courant dans le chemin

        (goal,player) = self.is_a_goal(path[0])
        if goal:
            path_cost[player] = 0

        current = path[ind]
        pred = None
        while nash and ind < len(path):

            if ind != 0:
                pred = current
                current = path[ind]
                epsilon += Graph.get_weight_pred(current, pred, self.graph.pred)

            if current.player not in already_test: #alors il faut tester pour ce joueur s'il s'agit d'un EN
                if current.player in coalitions:  #on a deja calcule les valeurs du jeu ou player joue contre la collation Pi\{player}
                    val = coalitions[current.player][current.id]

                    if ReachabilityGame.respect_property(val, epsilon, path_cost[current.player]):
                        ind += 1 # on respecte les conditions de la propriete pour etre un EN
                    else:
                        nash = False # on ne respcte pas la condition pour au moins un noeud -> pas un EN
                else:  # il faut au prealable calculer les valeurs du jeu min-max associe

                    graph_min_max = ReachabilityGame.graph_transformer(self.graph, current.player)
                    result_dijk_min_max = dijkstraMinMax(graph_min_max, self.goal[current.player - 1])
                    tab_result = get_all_values(result_dijk_min_max)
                    coalitions[current.player] = tab_result

                    val = coalitions[current.player][current.id]


                    if ReachabilityGame.respect_property(val, epsilon, path_cost[current.player]):
                        ind += 1  # on respecte les conditions de la propriete pour etre un EN


                    else:
                        nash = False  # on ne respcte pas la condition pour au moins un noeud -> pas un EN
            else:
                ind += 1
        return nash, coalitions

    def is_a_Nash_equilibrium_one_player(self, path, player, coalitions=None):

        """
        A partir d un chemin (path), determine si la condition d'etre un EN est respectue sur le chemin "path" pour
        un certain joueur "player".
        Coalition contient, si elles ont deja ete calculees, les valeurs des noeuds pour les jeux de coalition

        """

        path_cost = self.cost_for_one_player(path, player)  # calcule les couts de tous les joueurs
        if coalitions is None:
            coalitions = {}


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

        """
        A partir d un chemin calcule le cout de chaque joueur ainsi que les joueurs ayant atteint leur objectif sur ce
        chemin
        :param path: un chemin
        :return: tuple cost,player ou cost est le cout pour chaque joueur et player les joueurs ayant atteint leur
         objectif
        """


        player_goal = []

        cost = self.cost_for_all_players(path)

        for i in cost.keys():

            if cost[i] != float("infinity"):
                player_goal.append(i)

        return (cost, player_goal)


    def filter_best_result(self, result):

        """
        A partir d un ensemble de chemin, retient uniquement le meilleur (celui avec le maximum de joueur ayant atteint
        leur object et dont la somme des couts de ces joueurs est minimale
        :param result: ensemble de chemins
        :return: meilleur chemin parmi l ensemble de chemins
        """
        best_cost = None
        value_best_cost = float("infinity")
        best_reached = None
        value_best_reached = 0
        partial_value_best_cost = float("infinity")

        for res in result:
            (cost, player_goal) = self.get_info_path(res)

            sum_cost = 0

            for c in cost.values():
                sum_cost += c

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
        """
        A partir des couts de tous les joueurs, calcule la somme des couts pour un certain sous ensemble de joueur
        :param player_reached: sous ensemble de joueurs
        :param all_cost: couts pour tous les joueurs
        :return: somme des couts pour les joueurs de l'ensemble player_reached
        """

        cost = 0
        for p in player_reached:
            cost += all_cost[p]
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

        print "Matrice d'adjacence"
        mat = Graph.list_succ_to_mat()
        string = Graph.mat_to_string(mat)
        print string














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



def test_real_dijkstra():
    v0 = Vertex(0, 1)
    v1 = Vertex(1, 2)
    v2 = Vertex(2, 1)
    v3 = Vertex(3, 1)
    v4 = Vertex(4, 2)
    v5 = Vertex(5, 2)
    v6 = Vertex(6, 1)

    vertices = [v0, v1, v2, v3, v4, v5, v6]

    pred0 = [(1, 1), (2, 9), (0, 1)]
    pred1 = [(3, 1)]
    pred2 = [(5, 1)]
    pred3 = [(5, 3), (2, 1)]
    pred4 = [(6, 1)]
    pred5 = [(4, 4)]
    pred6 = [(5, 2)]

    list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6]
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    graph = Graph(vertices, None, list_pred, list_succ)
    graph_real_dijk = ReachabilityGame.graph_transformer_real_dijkstra(graph)
    goal = set([0])

    T = dijkstraMinMax(graph_real_dijk, goal)

    print_result(T, goal, list_succ)

def test_succ_to_mat():
    pred0 = [(1, 1), (2, 9), (0, 1)]
    pred1 = [(3, 1)]
    pred2 = [(5, 1)]
    pred3 = [(5, 3), (2, 1)]
    pred4 = [(6, 1)]
    pred5 = [(4, 4)]
    pred6 = [(5, 2)]

    list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6]
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    mat = Graph.list_succ_to_mat(list_succ)
    string = Graph.mat_to_string(mat)

    print string







def generate_vertex_uniform():

    vertex,partition = ReachabilityGame.generate_vertex_player_uniform(4, 5)
    for v in vertex:
        print "sommet v"+str(v.id)+" joueur :" + str(v.player)
    print "vertex ",vertex
    print "partition ", partition


def test():
    v0 = Vertex(0,2)
    v1 = Vertex(1,2)
    v2 = Vertex(2,1)
    v3 = Vertex(3,1)
    v4 = Vertex(4,3)
    v5 = Vertex(5,1)
    v6 = Vertex(6,1)
    v7= Vertex(7,2)
    v8 = Vertex(8,2)
    v9 = Vertex(9,1)

    vertex = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]

    pred0 = [(1,20)]
    pred1 = []
    pred2 =[]
    pred3 = []
    pred4 = []
    pred5 = []
    pred6 = []
    pred7 =[]
    pred8 = []
    pred9 = []

    list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9]
    graph = Graph(vertex, None, list_pred, None, 20)


    goal = [{0},{1}]

    game = ReachabilityGame(2, graph, v1, goal, None)

    path = [v1, v0]

    cost = game.cost_for_all_players(path)
    print cost

def find_loop_test():

    path = [0,1,2,1,4,5,6,7,2,7,8,0,9]
    print ReachabilityGame.find_loop(path,2)

if __name__ == '__main__':

    #test_generate_game()

    #test_path_cost()

    #test_nash_equilibrium()

    #test_real_dijkstra()

    #a_star_test1()
    #a_star_test2()

    #generate_vertex_uniform()

    #test()

    find_loop_test()
