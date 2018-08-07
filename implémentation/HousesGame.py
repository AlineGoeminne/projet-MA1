"""
Ce module permet de representer le "jeu des maisons" inspire du papier "efficient energy distribution in a smart grid
using multi-player games"
"""


from GraphGame import ReachabilityGame
from GraphGame import Vertex
from GraphGame import Graph
from GraphToDotConverter import graph_house_to_dot
from GraphToDotConverter import backward_house_to_dot
from GraphToDotConverter import minMax_graph_to_dot
from Value import *

import math
import random
import copy



class Task(object):

    def __init__(self, id, energy_consumption):

        self.id = id
        self.eC = energy_consumption


    def __str__(self):
        return repr(self.id)

class TaskFactory(object):
    number_tasks = 0

    def create_task(self, energy_consumption):
        TaskFactory.number_tasks += 1
        task = Task(TaskFactory.number_tasks, energy_consumption)
        return task

class Pref_task(object):

    def __init__(self,task,pref_time_set):
        self.task = task
        self.pref_set = pref_time_set #ensemble periodes de tps pdt lesquelles les taches peuvent s executer

    def __str__(self):
        return "<" + str(self.task) + "," + str(self.pref_set) + ">"

    def __repr__(self):
        nb_elem= len(self.pref_set)
        j = 0
        string = "{"
        for i in self.pref_set:
            j += 1
            if j != nb_elem:
                string += str(i)
                string +=","
            else:
                string += str(i)

        string += "}"
        return "<" + str(self.task) + "," + string + ">"

    def __eq__(self, other):
        return self.task.id == other.task.id and self.pref_set == other.pref_set


class HousesVertex(Vertex):

    def __init__(self,index,player,time, list_pref_tasks_for_all):
        Vertex.__init__(self,index,player)
        self.time = time # intervalle de temps a laquelle on se trouve
        self.tasks = list_pref_tasks_for_all # liste des listes des taches avec leurs pref pour chaque joueur

    def __str__(self):
        return "(id: "+ str(self.id) +",t: "+ str(self.time)+ ",p :" + str(self.player) +",tasks: "+str(self.tasks)+")"

    def __eq__(self, other):
        return self.id == other.id



class MemoHousesVertex(object):

    def __init__(self):

        self.all_vertices = {}

    def get_memo(self, time, turn_player, list_pref_tasks_for_all):
        return self.all_vertices.get((time, tuple(turn_player), tuple(map(tuple,list_pref_tasks_for_all))))

    def put_memo(self, time, turn_player,list_pref_tasks_for_all, vertex):

        self.all_vertices[(time, tuple(turn_player), tuple(map(tuple,list_pref_tasks_for_all)))] = vertex



class HousesVertexFactory(object):

    #number_houses_vertex = 0

    def __init__(self, nb_init = 0):
        self.number_houses_vertex = nb_init

    def create_houses_vertex(self, player,time,list_pref_tasks_for_all):
        vertex = HousesVertex(self.number_houses_vertex,player, time, list_pref_tasks_for_all)
        self.number_houses_vertex += 1
        return vertex



class WrappingConstHousesGame(object):
    def __init__(self, nb_player,nb_interval,eP,pIn,pOut):

        self.player = nb_player
        self.nb_interval = nb_interval
        self.eP = eP
        self.pIn = pIn
        self.pOut = pOut



class HousesGame(ReachabilityGame):

    def __init__(self, player, nb_interval, energy_production, list_pref_tasks_for_all, pIn, pOut, test=False):

        self.pIn = pIn
        self.pOut = pOut
        self.eP = energy_production
        self.player = player
        self.nb_interval = nb_interval
        wrap = WrappingConstHousesGame(player, nb_interval, energy_production, pIn, pOut)
        (graph, init, goal, partition) = HousesGame.generate_houses_game_tree(wrap, list_pref_tasks_for_all,test)
        ReachabilityGame.__init__(self, player, graph, init, goal, partition)

    @staticmethod
    def is_an_objective(vertex, player):
        return len(vertex.tasks[player - 1]) == 0

    @staticmethod
    def is_an_objective_for(vertex, nb_player):

        obj_for = set()

        for p in xrange(1, nb_player+1):
            if HousesGame.is_an_objective(vertex, p):
                obj_for.add(p)
        return obj_for


    @staticmethod
    def set_to_objective(vertex, goal, nb_player, all = False):

        """
         Ajoute le noeud vertex comme objectif aux joueurs adequats.

         :param vertex : noeud teste
         :param goal : liste des ensembles objectifs actuels
         :param nb_player : nb de joueurs
         :param all : si all est False, si le joueur i a fini toutes ses taches alors vertex est un noeud objectif
         du joueur i, si all est True, il faut que TOUS les joueurs aient fini toutes leurs taches pour etre un noeud
         objectif (pour tous les joueurs)

        """

        obj_for = HousesGame.is_an_objective_for(vertex,nb_player)

        if all and len(obj_for) == nb_player:
            for p in obj_for:
                goal[p-1].add(vertex.id)
        if not all:
            for p in obj_for:
                goal[p-1].add(vertex.id)



    @staticmethod

    def keep_max(max_values, new_values):

        """
         Soient t1 et t2 deux uplets, on garde composante par composante, dans t1, celle qui a la plus grande valeur en
         valeur absolue.
        """

        n = len(max_values)
        for i in xrange(n):
            abs_new = abs(new_values[i])
            if abs_new > max_values[i]:
                max_values[i] = abs_new

    @staticmethod
    def all_compleated(list_pref_task_for_all):


        for x in list_pref_task_for_all:
            if len(x) !=0:
                return False
        return True

    @staticmethod
    def aux_generate_houses_game_tree( wrap,
                                      list_pref_tasks,
                                      time,
                                      all_vertices,
                                      list_succ,
                                      goal,
                                      factory,
                                      turn_player,
                                      partial_weight,
                                      max_values,
                                      fixed_order = False):

        """
          Fonction auxiliaire pour la generation de l arbre du jeu.  (Voir generate_houses_game_tree)

        """


        player = turn_player[-1]

        vertex = factory.create_houses_vertex(player, time, list_pref_tasks)
        all_vertices.append(vertex)
        list_succ.append([])
        if len(turn_player) == wrap.player:
            #for p in xrange(1, nb_player + 1):

                #if HousesGame.is_an_objective(vertex, p):
                    #goal[p - 1].add(vertex.id)
            HousesGame.set_to_objective(vertex,goal,wrap.player, True)



        if time == wrap.nb_interval+1 : #temps ecoule

            list_succ[vertex.id].append((vertex.id,(0,) * wrap.player))
            return vertex

        else:
            new_turn_player = None
            time_changed = len(turn_player) == 1
            if time_changed: #il reste un unique joueur qui doit jouer
                new_turn_player = range(1,wrap.player + 1)
                if not fixed_order:
                    random.shuffle(new_turn_player)

            for a in list_pref_tasks[player - 1]:
                if time in a.pref_set:
                    new_list = copy.deepcopy(list_pref_tasks)
                    new_list[player - 1].remove(a)
                    up_partial_weight = HousesGame.update_partial_weight(partial_weight, a.task.eC, player)

                    if not (time_changed):
                        turn = copy.copy(turn_player)
                        turn.pop()


                        succ = HousesGame.aux_generate_houses_game_tree(wrap,
                                                                      new_list,
                                                                      time,
                                                                      all_vertices,
                                                                      list_succ,
                                                                      goal,
                                                                      factory,
                                                                      turn,
                                                                      up_partial_weight,
                                                                      max_values,
                                                                      fixed_order)
                        list_succ[vertex.id].append((succ.id, (0,) * wrap.player))

                    else:
                        turn = copy.copy(new_turn_player)
                        reinitialize_partial_weight = [0] * wrap.player
                        succ = HousesGame.aux_generate_houses_game_tree(wrap,
                                                                      new_list,
                                                                      time + 1,
                                                                      all_vertices,
                                                                      list_succ,
                                                                      goal,
                                                                      factory,
                                                                      turn,
                                                                      reinitialize_partial_weight,
                                                                      max_values,
                                                                      fixed_order)

                        weight = HousesGame.compute_real_weight(wrap,up_partial_weight)
                        HousesGame.keep_max(max_values, weight)
                        list_succ[vertex.id].append((succ.id,weight))

            new_list = copy.deepcopy(list_pref_tasks)
            up_partial_weight = copy.copy(partial_weight)

            if not (time_changed):
                turn = copy.copy(turn_player)
                turn.pop()

                succ = HousesGame.aux_generate_houses_game_tree(wrap,
                                                          new_list,
                                                          time,
                                                          all_vertices,
                                                          list_succ,
                                                          goal,
                                                          factory,
                                                          turn,
                                                          up_partial_weight,
                                                          max_values,
                                                          fixed_order)
                list_succ[vertex.id].append((succ.id, (0,) * wrap.player))

            else:
                turn = copy.copy(new_turn_player)
                new_partial_weight = [0] * wrap.player
                succ = HousesGame.aux_generate_houses_game_tree(wrap,
                                                          new_list,
                                                          time + 1,
                                                          all_vertices,
                                                          list_succ,
                                                          goal,
                                                          factory,
                                                          turn,
                                                          new_partial_weight,
                                                          max_values,
                                                          fixed_order)

                list_succ[vertex.id].append((succ.id, HousesGame.compute_real_weight(wrap,up_partial_weight)))

            return vertex

    @staticmethod
    def generate_houses_game_tree(wrap,
                                  list_pref_tasks,
                                  fixed_order = False):


        """
        :param wrap: contient les donnees : nombres de joueurs, nombres d'intervalle de temps, energie produite par
         unite de temps, prix de l unite d energie achetee a l interieur du reseau, prix de l unite d energie achetee en
         dehors du reseau

         :param list_pref_tasks : liste maison par maison, de la listes des taches a accomplir (munie des contraintes de temps)
         :type list_pref_tasks: tableau de tableaux de Pref_task

        :param fixed_ordre : True si l'ordre dans lequel les maisons choisissent leur action est fixe, False s il est de
         termine de maniere random
        """

        turn = range(1, wrap.player + 1)
        if not fixed_order:
            random.shuffle(turn)
        goal = []
        for x in xrange(wrap.player):
            goal.append(set())
        time = 1
        all_vertices = []
        list_succ = []
        partial_weight = [0] * wrap.player
        factory = HousesVertexFactory()
        max_values = [-float("infinity")] * wrap.player
        HousesGame.aux_generate_houses_game_tree(wrap,list_pref_tasks,time,all_vertices,list_succ,goal,factory,turn,partial_weight, max_values,fixed_order)
        graph = Graph(all_vertices, None, None, list_succ, tuple(max_values))
        return graph, all_vertices[0], goal, None



    @staticmethod
    def update_partial_weight(partial_weight,weight,player):
        new_partial_weight = copy.copy(partial_weight)
        new_partial_weight[player-1] += weight
        return new_partial_weight

    @staticmethod
    def compute_real_weight(wrap, partial_weight):

        """
            Calcule ce que chaque maison doit payer apres  l ecoulement d un intervalle de temps.
        """

        res = [0] * wrap.player
        G_pos = set() #ensemble des maisons ayant un gain positif weight- prod > 0
        G_neg = set() # ensemble des maisons ayant un gain negatif weight- prod < 0
        for p in xrange(1, wrap.player+1):
            if wrap.eP - partial_weight[p-1] > 0:
                G_pos.add(p)
            if wrap.eP - partial_weight[p-1] < 0:
                G_neg.add(p)

        benef = 0 #consommation superflue d energie pour toutes les maisons

        for p in G_pos:
            benef += wrap.eP - partial_weight[p-1]

        defi = 0 # deficit d energie pour toutes les maisons

        for p in G_neg:
            defi += partial_weight[p-1] - wrap.eP

        if benef - defi >= 0:

            for p in G_neg:

                res[p-1] = (wrap.eP - partial_weight[p-1])* wrap.pIn

            for p in G_pos:
                temp = wrap.eP - partial_weight[p-1]
                res[p-1] = int(math.floor((temp/benef)*defi * wrap.pIn))


        else : #benef - defi < 0

            for p in G_pos:
                res[p-1] = (wrap.eP - partial_weight[p-1]) * wrap.pIn
            for p in G_neg:
                temp = ((wrap.eP - partial_weight[p-1])/defi)*benef
                res[p-1] = int(math.floor(temp * wrap.pIn + (wrap.eP - partial_weight[p-1]- temp) * wrap.pOut))



        res = map(lambda x: -x, res )


        return tuple(res)



    #/!\ A ne pas utiliser
    #TODO: A revoir pour que ca cree le meme jeu que la generation de l arbre! (ex: ensembles objectifs, fonction de poids ,...
    def aux_generate_houses_game_dag(self, nb_player,
                                     nbr_interval,
                                     list_pref_tasks_for_all,
                                     time,
                                     all_vertices,
                                     list_succ,
                                     goal,
                                     memo,
                                     factory,
                                     turn_player):
        time_changed = False
        new_vertex = False

        player = turn_player[-1]
        vertex = memo.get_memo(time, turn_player, list_pref_tasks_for_all)

        if vertex is None:
            new_vertex = True
            vertex = factory.create_houses_vertex(player, time, list_pref_tasks_for_all)
            all_vertices.append(vertex)
            list_succ.append([])
            memo.put_memo(time, turn_player, list_pref_tasks_for_all, vertex)


            for p in xrange(1, nb_player+1):

                if HousesGame.is_an_objective(vertex, p):
                    goal[p - 1].add(vertex.id)

        turn_player.pop()

        if (time == nbr_interval+1):
            if new_vertex:
                list_succ[vertex.id].append((vertex.id,self.compute_weight_old(vertex, vertex)))

            return vertex

        else:

            if new_vertex:

                old_time = time
                if len(turn_player) == 0:  # alors on a fait joue tous les joueurs pour cet intervalle de temps
                    turn_player_bis = range(1, nb_player + 1)
                    random.shuffle(turn_player_bis)  # on redefinit l ordre des joueurs pour le prochain tour
                    turn_player = turn_player_bis
                    time_changed = True

                actions = list_pref_tasks_for_all[player - 1]  # on recupere les actions possibles pour le joueur considere
                for a in actions:

                    if old_time in a.pref_set:  # on verifie qu on peut exectuer cette action a ce moment la
                        new_list = []
                        for l in list_pref_tasks_for_all:
                            new_list.append(copy.copy(l))
                        new_turn = copy.copy(turn_player)
                        new_time = time
                        time = new_time

                        new_list[player - 1].remove(a)
                        if time_changed:
                            succ = self.aux_generate_houses_game_dag(nb_player, nbr_interval, new_list, time + 1,
                                                                     all_vertices, list_succ, goal, memo, factory,
                                                                     new_turn)
                        else:
                            succ = self.aux_generate_houses_game_dag(nb_player, nbr_interval, new_list, time,
                                                                     all_vertices, list_succ, goal, memo, factory,
                                                                     new_turn)

                        list_succ[vertex.id].append((succ.id, self.compute_weight_old(vertex, succ)))

                # on considere maintenant ne rien faire comme une action
                new_list = []
                for l in list_pref_tasks_for_all:
                    new_list.append(copy.copy(l))
                new_turn = copy.copy(turn_player)
                new_time = time
                time = new_time

                if time_changed:
                    succ =self.aux_generate_houses_game_dag(nb_player, nbr_interval, new_list, time + 1, all_vertices,
                                                            list_succ, goal, memo, factory, new_turn)
                else:
                    succ = self.aux_generate_houses_game_dag(nb_player, nbr_interval, new_list, time, all_vertices,
                                                             list_succ, goal, memo, factory, new_turn)

                list_succ[vertex.id].append((succ.id, self.compute_weight_old(vertex, succ)))
            return vertex

    #/!\ A ne pas utiliser
    def generate_houses_game_dag(self, nb_player, nbr_interval, list_pref_tasts_for_all):
        turn = range(1, nb_player + 1)
        random.shuffle(turn)
        goal = []
        for x in range(nb_player):
            goal.append(set())
        time = 1
        all_vertices = []
        list_succ = []
        memo = MemoHousesVertex()
        factory = HousesVertexFactory()
        self.aux_generate_houses_game_dag(nb_player, nbr_interval, list_pref_tasts_for_all, time, all_vertices, list_succ, goal, memo, factory, turn)
        graph = Graph(all_vertices,None,None,list_succ)
        return graph,all_vertices[0],goal,None


    def compute_weight_old(self, first_vertex, second_vertex):

        " Version naive pour calculer les poids entre deux noeuds"

        p = first_vertex.player
        nb_p = len(first_vertex.tasks)

        res1 = map(lambda x: x.task, first_vertex.tasks[p - 1])
        res2 = map(lambda x: x.task, second_vertex.tasks[p - 1])

        diff = set(res1) - set(res2)

        consumption = 0
        for t in diff:
            consumption += t.eC

        consumption -= self.eP
        res = [0] * nb_p
        res[p - 1] = consumption

        return tuple(res)





class HousesGameTest(HousesGame):

    """
        Cette classe est uniquement utilisee pour les tests, afin d obtenir tout le temps le meme jeu en fixant
        l ordre des joueurs. Si n est le nombre de joueur alors l'ordre est : n , n-1, n-2, ..., 1.
    """

    def __init__(self,player,nbr_interval, energy_production, list_pref_tasks_for_all, pIn, pOut):

        HousesGame.__init__(self, player, nbr_interval, energy_production, list_pref_tasks_for_all, pIn, pOut,True)









def parser_houses(file_name):

    """

    Permet de recuperer les donnees du jeu ( nombre d'intervalles, nombre de maison, energie produite par unite de temps,
    cout d achet d energie dans le reseau, cout d achat d energie en dehors du reseau, tache a accomplir maison par maison
    ainsi que les contraintes de temps de celles-ci)

    """



    nb_intervalle = 0
    nb_houses = 0
    energy_production = 0
    p_in = 0
    p_out = 0
    task_factory = TaskFactory()

    pref_tasks_list = []

    task_begin = False


    with open(file_name,"r") as file:
        lines = file.readlines()

        for l in lines:
            res = l.split("\n")[0]


            if res == "&":
                task_begin = True
                pref_tasks_list.append([])
                nb_houses += 1

            else:
                if task_begin:
                    pref = map(int, res.split(" "))
                    pref_set = set()
                    for i in range(1,len(pref)):
                        pref_set.add(pref[i])
                    task = task_factory.create_task(pref[0])
                    pref_task = Pref_task(task, pref_set)
                    pref_tasks_list[-1].append(pref_task)

            if (not task_begin):
                res = res.split(" ")
                nb_intervalle = int(res[0])
                energy_production = int(res[1])
                p_out = int(res[2])
                p_in = int(res[3])



    file.close()

    return nb_houses, energy_production, nb_intervalle, pref_tasks_list, p_out, p_in


def run_generate_graph_houses(inpout,outpout):

    " Genere le DOT du graphe du jeu donne en input "

    nb_houses, energy_production, nb_intervalle, pref_tasks_list, p_out, p_in = parser_houses(inpout)
    game = HousesGame(nb_houses, nb_intervalle, energy_production, pref_tasks_list,p_in,p_out)
    graph_house_to_dot(game,outpout)



#Quelques tests


def test_backward(inpout):
    nb_houses, energy_production, nb_intervalle, pref_tasks_list, p_out, p_in = parser_houses(inpout)
    game = HousesGame(nb_houses, nb_intervalle, energy_production, pref_tasks_list, p_in, p_out)
    strategies = game.backward()
    backward_house_to_dot(game,strategies, "backward.dot")




    mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
    game.graph.mat = mat

    list_pred = Graph.matrix_to_list_pred(mat)
    game.graph.pred = list_pred

    strat_ = copy.deepcopy(strategies)
    SPE = game.get_SPE_until_last_reach(strat_, nb_intervalle, nb_houses)
    print "SPE", SPE, "INFO", game.get_info_path(SPE)
    a_star = game.best_first_search(ReachabilityGame.a_star_negative,None, float("infinity"), True, True)
    strat_ = copy.deepcopy(strategies)

    SPE = game.get_SPE_until_last_reach(strat_,nb_intervalle,nb_houses)

    print "SPE ", SPE, " info ", game.get_info_path(list(SPE))
    print "---------------"
    print "A_STAR", a_star,"info", game.get_info_path(a_star)
    graph_min_max = ReachabilityGame.graph_transformer(game.graph, game.init.player)
    values_player = compute_value_with_negative_weight(graph_min_max, game.goal[game.init.player - 1], True, game.init.player - 1)[0]
    print "---------------"
    print "VALUES", values_player
    minMax_graph_to_dot(game, values_player,"min_max_back.dot")


    print game.is_a_Nash_equilibrium(a_star,None,None,True,True)







def first_simple_test():

    task_fact = TaskFactory()

    t1 = task_fact.create_task()
    t2 = task_fact.create_task()
    t3 = task_fact.create_task()
    t4 = task_fact.create_task()

    p1 = Pref_task(t1, {1,3})
    p2 = Pref_task(t2, {1,2})
    p3 = Pref_task(t3, {1,2})
    p4 = Pref_task(t4, {2})

    game = HousesGame(2,3,[[p1,p2],[p3,p4]])
    print game.goal
    graph_house_to_dot(game,"houses_test.dot")


def parser_test():

    print parser_houses("file_houses.txt")

def run_dot_parser_test():

    run_generate_graph_houses("file_houses.txt", "houses_test.dot")




if __name__ == '__main__':



    #first_simple_test()
    #parser_test()
    #run_dot_parser_test()
    #memo_test()
    #test_dag_to_tree("file_houses.txt", "graph_houses.dot","tree_houses.dot")
    test_backward("Test/Maisons/file_houses.txt")

