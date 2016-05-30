
import random
import heapq
import copy
import time


from DijkstraMinMax import dijkstraMinMax
from DijkstraMinMax import VertexDijkPlayerMax
from DijkstraMinMax import VertexDijkPlayerMin
from DijkstraMinMax import convertPred2NbrSucc
from DijkstraMinMax import get_all_values
from DijkstraMinMax import print_result
from DijkstraMinMax import get_succ_in_opti_strat
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
                    #weight_decimal = random.random()
                    weight_decimal = 0

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
                    #weight_decimal = random.random()
                    weight_decimal = 0


                    if i != j:
                        new_succ.append((j, weight + weight_decimal))

                res.append(new_succ)
            return res

        if type == "pred":
            for i in range(0, nbr_vertex):
                new_pred = []
                for j in range(0, nbr_vertex):
                    weight = random.randint(a, b)
                    #weight_decimal = random.random()
                    weight_decimal = 0


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


class Node_restart_search(Node):

    def __init__(self, current, parent, eps, cost, value = float("infinity"), blocked = set([])):

        Node.__init__(self, current, parent, eps, cost, value)
        self.blocked = blocked



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

        return (self.player + 1)* (len(self.graph.vertex) + sum)

    @staticmethod
    def path_vertex_to_path_index(path):
        res = []
        for i in path:
            res.append(i.id)
        return res

    @staticmethod
    def generate_vertex_player_uniform(nbr_player, nbr_vertex):

        #todo: id_to_player inutile

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

        """
        init est l'ID du sommet initial
        """

        #todo: dic inutile

        pred = Graph.generate_complete_graph(nbr_vertex,"pred", a, b)
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
    def graph_transformer_real_dijkstra(graph):
        """
         A partir d'un graphe, modelise le graphe du jeu min-max tel que le joueur voulant minimiser est min_player

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

    def parcours_d_arbre(self, deep):

        #todo: ! le premier vertex peut etre un etat objectif

        path_init = [self.init]
        deep = deep - 1
        coalitions = {}
        reach_player = []
        result = []

        return self.parcours_d_arbre_aux(path_init, deep, coalitions, reach_player, result)

    def parcours_d_arbre_aux(self, path, deep, coalitions, reach_player, result):

        """
        Fonction auxiliaire permettant de parcourir un arbre sur une hauteur de "deep"

        :param path: le chemin courant
        :param deep: la profondeur maximale de parcourt de l'arbre
        :param coalitions: dictionnaire (joueur -> tableau de valeurs)
        :param cost : pour chaque joueur le cout du chemin
        :param eps: cout du prefixe du chemin
        :param reach_player: ensemble des joueur ayant atteint leur objectif
        :param result: ensemble des equilibres de Nash retournes

        """

        doc = open("parcours_arbre.txt", "a")
        result_doc = open("result.txt", "a")
        if deep == 0:
            (nash, coalitions) = self. is_a_Nash_equilibrium(path,coalitions)
            if nash:
                result_doc.write(str(ReachabilityGame.path_vertex_to_path_index(path)))
                result_doc.write("\n")
                result.append(path)
            #return result


        else:
            last_vertex = path[len(path) - 1]
            succ_current = self.graph.succ[last_vertex.id]
            for i in range(0, len(succ_current)):

                (succ, w) = succ_current[i]

                succ_vertex = self.graph.vertex[succ]
                new_path = []
                new_path[0:len(path)] = path
                new_path.append(succ_vertex)

                doc.write(str(ReachabilityGame.path_vertex_to_path_index(new_path)))
                doc.write("\n")

                (goal, player_new_goal) = self.is_a_goal(succ_vertex)
                if goal and player_new_goal not in reach_player:
                    _reach_player = []
                    _reach_player[0:len(reach_player)] = reach_player
                    _reach_player.append(player_new_goal)
                    (nash, coalitions) = self.is_a_Nash_equilibrium_one_player(new_path, player_new_goal, coalitions)
                    if nash:
                        if len(_reach_player) == self.player:
                            result_doc.write(str(ReachabilityGame.path_vertex_to_path_index(new_path)))
                            result_doc.write("\n")
                            result.append(new_path)
                        else:
                            self.parcours_d_arbre_aux(new_path, deep - 1, coalitions, _reach_player, result)

                else:
                    self.parcours_d_arbre_aux(new_path, deep - 1 , coalitions, reach_player, result)
        doc.close()
        result_doc.close()
        return result



    def compute_all_dijkstra(self):
        """
        Calcule la longueur du plus court chemin pour aller d un noeud a un etat objectif et ce pour tout objectif
        """
        result ={}
        union_goal = set([])
        for i in range(0, len(self.goal)):
            union_goal = union_goal.union(self.goal[i])

        for g in union_goal:

            graph_dijk = self.graph_transformer_real_dijkstra(self.graph)
            T = dijkstraMinMax(graph_dijk,{g})
            res = get_all_values(T)
            result[g] = res


        return result



    #test d'un parcours d'arbre un peu plus intelligent


    def best_first_search(self, heuristic, frontier = None, allowed_time = float("infinity")):

        if heuristic is ReachabilityGame.a_star:
            all_dijk = self.compute_all_dijkstra()
        else:
            all_dijk = None

        mon_fichier = open("test.txt", "w")
        mon_fichier.write("************************** \n")
        parcours = open("parcours_best.txt", "w")
        result = open("result_best_first_search.txt", "w")
        result.write("info sur le jeu : \n")
        result.write(Graph.mat_to_string(Graph.list_succ_to_mat(self.graph.succ)))
        result.write("\n")
        result.write("nombres de noeuds :" +repr(len(self.graph.vertex)))
        result.write("\n")
        result.write("noeud initial : v" + repr(self.init.id))
        result.write("\n")
        result.write("objectifs :" + repr(self.goal))
        result.write("\n")
        result.write("partitions :" + repr(self.part))
        result.write("\n")
        result.write("temps permis: " +repr(allowed_time))

        result.write("\n")
        result.write("\n")


        start = time.time()



        max_length = self.compute_max_length()


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
                mon_fichier.close()
                raise BestFirstSearchError(" Plus d'elements dans la frontiere")

            candidate_node = heapq.heappop(frontier)
            candidate_path = candidate_node.current

            mon_fichier.write("pop : ")
            mon_fichier.write(str(ReachabilityGame.path_vertex_to_path_index(candidate_path)))
            mon_fichier.write("\n")

            parcours.write(str(ReachabilityGame.path_vertex_to_path_index(candidate_path)))
            parcours.write("------->")
            parcours.write(str(candidate_node.value))
            parcours.write("\n")

            if len(candidate_node.cost) == self.player:
                parcours.write("j'ai trouve un EN : ")
                parcours.write(str(candidate_path))
                parcours.write("\n")

                result.write("j'ai trouve un EN : ")
                result.write(str(candidate_path))
                result.write("\n")

                parcours.close()
                result.close()
                mon_fichier.close()
                #Alors on sait qu'il s'agit d'un equilibre de Nash car tous les cost ont ete initialises
                return candidate_path

            if len(candidate_path) == max_length :
                #Il se peut que ce soit un equilibre de Nash tel que tous les joueurs n'ont pas vu leur objectif mais
                #tel que la longueur max est atteinte: il faut donc tester pour ceux qui n'ont pas encore atteint leur obj

                (nash,coalition) = self.is_a_Nash_equilibrium(candidate_path, candidate_node.cost.keys())
                if nash:
                    parcours.write("j'ai trouve un EN : ")
                    parcours.write(str(candidate_path))
                    parcours.write("\n")

                    result.write("j'ai trouve un EN : ")
                    result.write(str(candidate_path))
                    result.write("\n")

                    parcours.close()
                    result.close()
                    mon_fichier.close()
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
                    mon_fichier.write(str(ReachabilityGame.path_vertex_to_path_index(new_path)))

                    (goal, player) = self.is_a_goal(succ_vertex)

                    if goal and player not in candidate_node.cost:
                        mon_fichier.write("nouvel objectif atteint pour "+str(player))
                        mon_fichier.write("\n")

                        (nash, coalitions) = self.is_a_Nash_equilibrium_one_player(new_path, player)


                        mon_fichier.write("pour le moment je respecte les conditions d' etre un EN :")

                        mon_fichier.write(str(nash))
                        mon_fichier.write("\n")

                        if nash:

                            new_cost = copy.deepcopy(candidate_node.cost)
                            new_cost[player] = epsilon
                            value = heuristic(self, new_cost,epsilon, len(new_path), succ_vertex, all_dijk)

                            mon_fichier.write("valeur du nouveau chemin :")
                            mon_fichier.write(str(value))
                            mon_fichier.write("\n")


                            new_node = Node(new_path, candidate_node, epsilon, new_cost, value)

                            heapq.heappush(frontier, new_node)
                            mon_fichier.write("frontiere :")
                            mon_fichier.write(str(frontier))
                            mon_fichier.write("\n")

                    else:
                        value = heuristic(self, candidate_node.cost,epsilon, len(new_path), succ_vertex, all_dijk)
                        mon_fichier.write("valeur du chemin : ")
                        mon_fichier.write(str(value))
                        mon_fichier.write("\n")

                        new_node = Node(new_path, candidate_node, epsilon, candidate_node.cost, value)
                        heapq.heappush(frontier, new_node)
                        mon_fichier.write("border :")
                        mon_fichier.write(str(frontier))
                        mon_fichier.write("\n")

        mon_fichier.close()
        parcours.close()
        result.write("Calcul stoppe")
        result.write("\n")
        result.write("**********************")
        result.close()
        return

    def restart_best_first_search_aux(self, heuristic, border=None, allowed_time=float("infinity")):


        pop = open("pop_restart_best_first_search.txt", "a")
        pop.write("**************** \n")
        pop.write("nouvelle tentative \n")

        only_pop = open("only_pop.txt", "a")
        only_pop.write("**************** \n")
        only_pop.write("nouvelle tentative \n")

        result = open("result_restart.txt", "w")
        result.write("info sur le jeu : \n")
        result.write(Graph.mat_to_string(Graph.list_succ_to_mat(self.graph.succ)))
        result.write("\n")
        result.write("nombres de noeuds :" + repr(len(self.graph.vertex)))
        result.write("\n")
        result.write("noeud initial : v" + repr(self.init.id))
        result.write("\n")
        result.write("objectifs :" + repr(self.goal))
        result.write("\n")
        result.write("partitions :" + repr(self.part))
        result.write("\n")
        result.write("temps permis: " + repr(allowed_time))

        result.write("\n")
        result.write("\n")

        start = time.time()


        max_length = (self.player + 1) * self.graph.max_weight * len(self.graph.vertex)

        if border is None :
            cost = {}
            (goal, player) = self.is_a_goal(self.init)
            if goal:
                cost[player] = 0

            value = heuristic(self, cost,0, 1, self.init)
            initial_node = Node_restart_search([self.init], Node([],None, 0, {}, float("infinity")), 0, cost, value, set([]))
            border = [initial_node]

        returned_border = border
        returned_candidate = border[0]
        returned_succ = None

        returned = False


        while(len(border) != 0 and time.time() - start <= allowed_time):

            candidate = heapq.heappop(border)

            pop.write( "POP :")
            pop.write(str(candidate.current))
            pop.write("\n")

            only_pop.write("POP :")
            only_pop.write(str(candidate.current))
            only_pop.write("\n")


            last_vertex = candidate.current[-1]
            list_succ = self.graph.succ[last_vertex.id]

            random.shuffle(list_succ)

            actual_border = copy.copy(border)
            actual_candidate = candidate
            if len(candidate.current) < max_length:
                for i in range(0, len(list_succ)):

                    (succ, w) = list_succ[i]
                    succ_vertex = self.graph.vertex[succ]

                    pop.write("blocked : ")
                    pop.write(str(candidate.blocked))
                    pop.write("\n")

                    if succ not in candidate.blocked:

                        new_path = []
                        new_path[0 : len(candidate.current)] = candidate.current
                        new_path.append(succ_vertex)

                        pop.write("on rajoute : ")
                        pop.write(str(new_path))
                        pop.write("\n")

                        epsilon = candidate.eps + w

                        (goal, player) = self.is_a_goal(succ_vertex)

                        if goal and player not in candidate.cost:

                            pop.write(" un nouvel objectif est atteint :" + str(goal) + "pour le joueur :" + str(player))
                            pop.write("\n")

                            if not returned:
                                returned_border = copy.copy(actual_border)
                                returned_candidate = actual_candidate
                                returned_succ = succ
                                returned = True

                            new_cost = copy.copy(candidate.cost)
                            new_cost[player] = epsilon

                            (nash, coalition) = self.is_a_Nash_equilibrium_one_player(new_path, player)
                            if nash:
                                pop.write("on respecte les conditions d'en :")
                                pop.write(str(nash))
                                pop.write("\n")

                                if len(new_cost) == self.player or len(new_path) == max_length:

                                    pop.write("j ai trouve un EN :")
                                    pop.write(str(new_path))
                                    pop.write("\n")
                                    pop.close()
                                    only_pop.close()
                                    return False, new_path, None, None

                                value = heuristic(self, new_cost,epsilon, len(new_path), succ_vertex)
                                new_node = Node_restart_search(new_path,candidate, epsilon, new_cost, value, set([]))
                                heapq.heappush(border, new_node)

                                pop.write("frontiere :")
                                pop.write(str(border))
                                pop.write("\n")


                        else:
                            value = heuristic(self, candidate.cost, epsilon, len(new_path), succ_vertex)
                            new_node = Node_restart_search(new_path, candidate, epsilon, candidate.cost, value, set([]))
                            heapq.heappush(border, new_node)

                            pop.write("frontiere :")
                            pop.write(str(border))
                            pop.write("\n")

        pop.close()
        only_pop.close()
        return True, returned_candidate, returned_border, returned_succ


    def restart_best_first_search(self, heuristic, allowed_time = float("infinity")):

        (fail, candidate, border, succ) = self.restart_best_first_search_aux(heuristic, None, allowed_time)
        nash_found = not fail
        attempt = 5

        if not fail:
            return candidate
        else:
            while not nash_found and attempt >= 0:
                print "je repars de:", str(candidate.current)


                if succ is not None:
                    candidate.blocked.add(succ)
                print "et je ne plus aller vers", str(candidate.blocked)
                candidate.value = - float("infinity")
                heapq.heappush(border, candidate)

                (fail, candidate, border, succ) = self.restart_best_first_search_aux(heuristic, border, allowed_time)

                if not fail:
                    return candidate
                attempt -= 1




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

        #todo: on suppose un seul objectif par joueur

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
        nb_vertex = len(game.graph.vertex)
        nb_reached = len(cost)
        max_length_path = game.compute_max_length()

        value = 0

        if nb_reached == 0:
            return float("infinity")
        else:
            keys = cost.keys()
            for p in keys:
                value += cost[p]
            penality = (nb_player - nb_reached)* max_weight* (max_length_path - length_path) + length_path + (epsilon * (nb_player - nb_reached))
            return value + penality

    @staticmethod
    def heuristic_short_path(game, cost, epsilon, length_path, last_vertex = None, all_dijk = None):



        nb_player = game.player
        max_weight = game.graph.max_weight
        nb_vertex = len(game.graph.vertex)
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
            return max_length_path * nb_player* max_weight+ tab_result[last_vertex.id] + epsilon*nb_player
        else:
            keys = cost.keys()
            for p in keys:
                value += cost[p]
            penality = (nb_player - nb_reached) * max_weight * (max_length_path - length_path) + length_path + (epsilon * (nb_player - nb_reached))

            return value + penality + tab_result[last_vertex.id]

    @staticmethod
    def profondeur(game, cost, epsilon, length_path, last_vertex = None):
        return 0




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

        graph_dijk = ReachabilityGame.graph_transformer_real_dijkstra(self.graph)


        T = dijkstraMinMax(graph_dijk, set([goal]))
        successor = get_succ_in_opti_strat(T, set([goal]), self.graph.succ)
        goal_is_reached = False
        path = [self.init]

        if self.init.id == goal:
            return path
        else:

            while not goal_is_reached:
                succ = self.graph.vertex[successor[path[-1].id]]
                path.append(succ)



                if succ.id == goal:
                    goal_is_reached = True

            return path

    def best_first_search_with_init_path(self, heuristic, allowed_time = float("infinity")):

        (path, player) = self.init_search()

        (nash, coalition) = self.is_a_Nash_equilibrium_one_player(path, player)

        if nash:
            cost = self.cost_for_all_players(path,True)
            epsilon = self.compute_epsilon(path)
            value = heuristic(self, cost, epsilon, len(path), path[-1])
            eps = self.compute_epsilon(path)
            node = Node(path, Node([], None, 0, cost, value), eps, cost, value)
            border = [node]

            res = self.best_first_search(heuristic, border, allowed_time)
            return res

        else:
            return None

    def best_first_search_with_init_path_both_two(self, heuristic, allowed_time=float("infinity")):

        for p in range(1, self.player +1):

            res = self.best_first_search_with_init_path_both_two_aux(heuristic, p, allowed_time)

            if res is not None:
                return res

        return None

    def best_first_search_with_init_path_both_two_aux(self, heuristic, player, allowed_time=float("infinity")):

        if heuristic is ReachabilityGame.a_star:
            all_dijk = self.compute_all_dijkstra()
        else:
            all_dijk = None
        # on essaie avec l objectif du joueur player
        goal = self.goal[player-1].pop()
        self.goal[player-1].add(goal)
        path = self.init_search_goal(goal)

        (nash, coalition) = self.is_a_Nash_equilibrium_one_player(path, player)

        if nash:
            cost = self.cost_for_all_players(path, True)
            epsilon = self.compute_epsilon(path)
            value = heuristic(self, cost, epsilon, len(path), path[-1], all_dijk)
            eps = self.compute_epsilon(path)
            node = Node(path, Node([], None, 0, cost, value), eps, cost, value)
            border = [node]

            res = self.best_first_search(heuristic, border, allowed_time)
            return res

        else:
            return None


    def compute_epsilon(self, path):

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
                    cost[player] = weight
                    reach_goals_player.add(player)
                    reach_goals.add(v.id)

        if (not only_reached):
            for p in range(1, self.player + 1):
                if p not in cost:
                    cost[p] = float("infinity")

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


    def is_a_Nash_equilibrium(self, path,already_test = None, coalitions = None):

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
        # dans un premier temps supposons que chaque element du path est un noeud (id,player)
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

        for i in cost.keys():

            if cost[i] != float("infinity"):
                player_goal.append(i)

        return (cost, player_goal)


    def filter_best_result(self, result):

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
        print mat














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

def test_restart():
    mat = [[0, 6, 4, 4, 5],[7, 0, 4, 1, 10],[5, 1, 0, 3, 10],[7, 1, 1, 0, 8],[6, 3, 1, 4, 0]]

    list_pred = Graph.matrix_to_list_pred(mat)
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    v0 = Vertex(0, 2)
    v1 = Vertex(1, 2)
    v2 = Vertex(2, 1)
    v3 = Vertex(3, 1)
    v4 = Vertex(4, 1)

    vertex = [v0, v1, v2, v3, v4]

    graph = Graph(vertex, mat, list_pred, list_succ, 10)

    goals = [set([4]), set([0])]

    game = ReachabilityGame(2, graph, v3, goals, None, {0:2, 1:2, 2:1, 3:1, 4:1})

    res = game.restart_best_first_search(ReachabilityGame.heuristic, 5)

    res2 = game.best_first_search_with_init_path(ReachabilityGame.heuristic, 5)

    result = game.test_random_path(100, (game.player + 1) * game.graph.max_weight * len(game.graph.vertex))

    print  str(res)
    print "  "
    print str(res2)
    print "  "


    for r in result:
        print r

def test_restart_2():
    mat = [ [0, 2, 3, 5, 5],[2, 0, 7, 7, 2],[10, 6, 0, 1, 9],[9, 3, 2, 0, 7], [10, 6, 8, 5, 0]]
    list_pred = Graph.matrix_to_list_pred(mat)
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    v0 = Vertex(0, 1)
    v1 = Vertex(1, 1)
    v2 = Vertex(2, 1)
    v3 = Vertex(3, 1)
    v4 = Vertex(4, 1)

    vertex = [v0, v1, v2, v3, v4]

    graph = Graph(vertex, mat, list_pred, list_succ, 10)

    goals = [set([0]), set([4])]

    game = ReachabilityGame(2, graph, v3, goals, None, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})

    res = game.restart_best_first_search(ReachabilityGame.heuristic, 5)

    #res2 = game.best_first_search_with_init_path(ReachabilityGame.heuristic, 5)

    #result = game.test_random_path(100, (game.player + 1) * game.graph.max_weight * len(game.graph.vertex))



    #print game.is_a_Nash_equilibrium(path)

    print  str(res)
    print "  "
    #print str(res2)
    print "  "



def a_star_test1():
    mat = [[0, 7, 9, 5, 4],[5, 0, 2, 9, 3],[6, 4, 0, 9, 3],[2, 7, 2, 0, 10],[8, 9, 7, 3, 0]]

    list_pred = Graph.matrix_to_list_pred(mat)
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    v0 = Vertex(0, 1)
    v1 = Vertex(1, 2)
    v2 = Vertex(2, 2)
    v3 = Vertex(3, 2)
    v4 = Vertex(4, 2)

    vertex = [v0, v1, v2, v3, v4]

    graph = Graph(vertex, mat, list_pred, list_succ, 10)

    goals = [{0}, {4}]

    game = ReachabilityGame(2, graph, v3, goals, None, {0: 1, 1: 2, 2: 2, 3: 2, 4: 2})

    res = game.best_first_search(ReachabilityGame.a_star,None, 5)

    print "a_star", str(res)
    print "EN ? ", game.is_a_Nash_equilibrium(res)

    res2 = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star, 5 )

    print "both-two ", str(res2)
    print "En? ", game.is_a_Nash_equilibrium(res2)





def a_star_test2():
    mat = [[0, 9, 3, 3, 5],[3, 0, 3, 9, 1],[9, 8, 0, 10, 1],[8, 6, 3, 0, 8],[4, 4, 9, 4, 0]]

    list_pred = Graph.matrix_to_list_pred(mat)
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    v0 = Vertex(0, 1)
    v1 = Vertex(1, 2)
    v2 = Vertex(2, 2)
    v3 = Vertex(3, 2)
    v4 = Vertex(4, 1)

    vertex = [v0, v1, v2, v3, v4]

    graph = Graph(vertex, mat, list_pred, list_succ, 10)

    goals = [{1}, {0}]

    game = ReachabilityGame(2, graph, v3, goals, None, {0: 1, 1: 2, 2: 2, 3: 2, 4: 2})

    res = game.best_first_search(ReachabilityGame.a_star,None, 5)

    print "a_star", str(res)
    print "EN ? ", game.is_a_Nash_equilibrium(res)

    res2 = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star, 5 )

    print "both-two ", str(res2)
    print "En? ", game.is_a_Nash_equilibrium(res2)



if __name__ == '__main__':

    #test_generate_game()

    #test_path_cost()

    #test_nash_equilibrium()

    #test_real_dijkstra()

    #test_restart_2()
    a_star_test2()