from GraphGame import ReachabilityGame
from GraphGame import Vertex
from GraphGame import Graph
from GraphToDotConverter import graph_house_to_dot
from GraphToDotConverter import tree_house_to_dot
from GraphToDotConverter import backward_house_to_dot
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

class Wraping2GenerateHousesGame(object):

    def __init__(self,time, all_vertices_list,list_succ, goal, memo, houses_factory, turn_player_list):
        self.time = time
        self.all_vertices = all_vertices_list
        self.succ = list_succ
        self.goal = goal
        self.memo = memo
        self.factory = houses_factory
        self.turn_player = turn_player_list




class HousesGame(ReachabilityGame):

    @staticmethod
    def is_an_objective(vertex, player):

        return len(vertex.tasks[player - 1]) == 0


    @staticmethod

    def keep_max(max_values, new_values):

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

    def aux_generate_houses_game_tree(self,nb_player,
                                      nb_interval,
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
        player = turn_player[-1]

        vertex = factory.create_houses_vertex(player, time, list_pref_tasks)
        all_vertices.append(vertex)
        list_succ.append([])
        if len(turn_player) == self.player:
            for p in xrange(1, nb_player + 1):

                if HousesGame.is_an_objective(vertex, p):
                    goal[p - 1].add(vertex.id)

        if time == nb_interval+1 : #temps ecoule

            list_succ[vertex.id].append((vertex.id,(0,)*nb_player))
            return vertex

        else:
            new_turn_player = None
            time_changed = len(turn_player) == 1
            if time_changed: #il reste un unique joueur qui doit jouer
                new_turn_player = range(1,nb_player+1)
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


                        succ = self.aux_generate_houses_game_tree(nb_player,
                                                                      nb_interval,
                                                                      new_list,
                                                                      time,
                                                                      all_vertices,
                                                                      list_succ,
                                                                      goal,
                                                                      factory,
                                                                      turn,
                                                                      up_partial_weight,
                                                                      max_values)
                        list_succ[vertex.id].append((succ.id, (0,) * nb_player))

                    else:
                        turn = copy.copy(new_turn_player)
                        reinitialize_partial_weight = [0] * nb_player
                        succ = self.aux_generate_houses_game_tree(nb_player,
                                                                      nb_interval,
                                                                      new_list,
                                                                      time + 1,
                                                                      all_vertices,
                                                                      list_succ,
                                                                      goal,
                                                                      factory,
                                                                      turn,
                                                                      reinitialize_partial_weight,
                                                                      max_values)

                        weight = self.compute_real_weight(up_partial_weight)
                        HousesGame.keep_max(max_values, weight)
                        list_succ[vertex.id].append((succ.id,weight))

            new_list = copy.deepcopy(list_pref_tasks)
            up_partial_weight = copy.copy(partial_weight)

            if not (time_changed):
                turn = copy.copy(turn_player)
                turn.pop()

                succ = self.aux_generate_houses_game_tree(nb_player,
                                                          nb_interval,
                                                          new_list,
                                                          time,
                                                          all_vertices,
                                                          list_succ,
                                                          goal,
                                                          factory,
                                                          turn,
                                                          up_partial_weight,
                                                          max_values)
                list_succ[vertex.id].append((succ.id, (0,) * nb_player))

            else:
                turn = copy.copy(new_turn_player)
                new_partial_weight = [0]*nb_player
                succ = self.aux_generate_houses_game_tree(nb_player,
                                                          nb_interval,
                                                          new_list,
                                                          time + 1,
                                                          all_vertices,
                                                          list_succ,
                                                          goal,
                                                          factory,
                                                          turn,
                                                          new_partial_weight,
                                                          max_values)

                list_succ[vertex.id].append((succ.id, self.compute_real_weight(up_partial_weight)))

            return vertex


    def generate_houses_game_tree(self, nb_player,
                                  nb_interval,
                                  list_pref_tasks,
                                  fixed_order = False):

        turn = range(1, nb_player + 1)
        if not fixed_order:
            random.shuffle(turn)
        goal = []
        for x in xrange(nb_player):
            goal.append(set())
        time = 1
        all_vertices = []
        list_succ = []
        partial_weight = [0]*self.player
        factory = HousesVertexFactory()
        max_values = [-float("infinity")]*nb_player
        self.aux_generate_houses_game_tree(nb_player,nb_interval,list_pref_tasks,time,all_vertices,list_succ,goal,factory,turn,partial_weight, max_values)
        graph = Graph(all_vertices, None, None, list_succ, tuple(max_values))
        return graph, all_vertices[0], goal, None



    @staticmethod
    def update_partial_weight(partial_weight,weight,player):
        new_partial_weight = copy.copy(partial_weight)
        new_partial_weight[player-1] += weight
        return new_partial_weight

    def compute_real_weight(self, partial_weight):

        res = [0] * self.player

        totC = 0
        totO = 0

        for i in xrange(0,self.player):

            totC += max(0, partial_weight[i] - self.eP)
            totO += partial_weight[i]
        totO -= self.player* self.eP
        bTot = (totC - totO) * self.cIn + totO* self.cOut

        if totC != 0:
            ratio = bTot / totC

            for i in xrange(0,self.player):

                res[i] = ratio * (partial_weight[i] - self.eP)

        #todo trouver une solution acceptable
        else:
            for i in xrange(0,self.player):
                res[i] = partial_weight[i] - self.eP
                #res[i] = 42

        return tuple(res)




    #TODO:retirer le max_iter
    def aux_generate_houses_game_dag(self, nb_player,
                                     nbr_interval,
                                     list_pref_tasks_for_all,
                                     time,
                                     all_vertices,
                                     list_succ,
                                     goal,
                                     memo,
                                     factory,
                                     turn_player,
                                     max_iter = 10000000):
        max_iter -=1
        time_changed = False
        new_vertex = False

        player = turn_player[-1]
        vertex = memo.get_memo(time, turn_player, list_pref_tasks_for_all)
        #vertex = memo.all_vertices.get((time, tuple(turn_player), tuple(map(tuple,list_pref_tasks_for_all))))

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

        if (time == nbr_interval and len(turn_player) == 0):
                #or HousesGame.all_compleated(list_pref_tasks_for_all)\
                #or max_iter ==0:
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
                                                                     new_turn, max_iter)
                        else:
                            succ = self.aux_generate_houses_game_dag(nb_player, nbr_interval, new_list, time,
                                                                     all_vertices, list_succ, goal, memo, factory,
                                                                     new_turn, max_iter)

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
                                                            list_succ, goal, memo, factory, new_turn, max_iter)
                else:
                    succ = self.aux_generate_houses_game_dag(nb_player, nbr_interval, new_list, time, all_vertices,
                                                             list_succ, goal, memo, factory, new_turn, max_iter)

                list_succ[vertex.id].append((succ.id, self.compute_weight_old(vertex, succ)))
            return vertex

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
        #wrap = Wraping2GenerateHousesGame(1, [], [], goal, MemoHousesVertex(), HousesVertexFactory(), turn)
        self.aux_generate_houses_game_dag(nb_player, nbr_interval, list_pref_tasts_for_all, 1, all_vertices, list_succ, goal, memo, factory, turn)
        graph = Graph(all_vertices,None,None,list_succ)
        return graph,all_vertices[0],goal,None


    def compute_weight_old(self, first_vertex, second_vertex):
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

    def __init__(self, player, nbr_interval, energy_production, list_pref_tasks_for_all, cIn = None, cOut = None):

        self.cIn = cIn
        self.cOut = cOut
        self.eP = energy_production
        self.player = player
        (graph, init, goal, partition) = self.generate_houses_game_tree(player, nbr_interval, list_pref_tasks_for_all)
        ReachabilityGame.__init__(self,player,graph,init,goal,partition)


    @staticmethod
    def choice_action(cost, player):

        return HousesGame.all_the_best_actions(cost, player)


    @staticmethod
    def one_of_the_best_action(cost, player):
        pass

    @staticmethod
    def all_the_best_actions(actions, player):
        (best,cost) = actions.popitem()


        res = {best}
        for a in iter(actions):
            new_cost = actions[a][player-1]
            if new_cost < cost[player-1]:
                res = set()
                res.add(a)
                cost = actions[a]
            if new_cost == cost[player-1]:
                res.add(a)

        return res, cost
    @staticmethod
    def minimize_the_sum_action(cost):

        pass

    def aux_backward(self, vertex,  strategies):

        vertex_id = vertex.id

        if self.graph.succ[vertex_id][0][0] == vertex_id: # boucle -> etat terminal
            res = [0]*self.player
            for p in xrange(1, self.player+1):
                if vertex.id not in self.goal[p-1]:
                    res[p-1] = float("infinity")

            strategies[vertex_id] = ([vertex_id],res)
            return res

        else:
            all_succ = self.graph.succ[vertex_id]
            all_possibilities = {}
            goal, players = self.is_a_goal(vertex)

            for tuple_s in all_succ:
                s = self.graph.vertex[tuple_s[0]]
                sub_values = self.aux_backward(s, strategies)
                values = ReachabilityGame.sum_two_vector_of_weight(sub_values, tuple_s[1],players)
                for p in players:
                    values[p-1] = 0
                all_possibilities[s.id] = values
            (choices, cost) = HousesGame.choice_action(all_possibilities, vertex.player)
            strategies[vertex_id] = (choices,cost)
            return cost






    def backward(self):

        init = self.graph.vertex[0]
        strategies = {}
        self.aux_backward(init,strategies)
        return strategies

    @staticmethod

    def get_SPE(strategies, nb_interval, nb_player):

        length = nb_interval * nb_player + 1 # puisque c'est un  arbre on arrete a la profondeur

        res = [0]
        v_current_id = 0
        while len(res) < length:
            v_current_id = strategies[v_current_id][0].pop()
            res.append(v_current_id)

        return res
















def parser_houses(file_name):

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
                    #print res.split(" ")
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

    nb_houses, energy_production, nb_intervalle, pref_tasks_list, p_out, p_in = parser_houses(inpout)
    game = HousesGame(nb_houses, nb_intervalle, energy_production, pref_tasks_list,p_in,p_out)
    graph_house_to_dot(game,outpout)


def test_backward(inpout):
    nb_houses, energy_production, nb_intervalle, pref_tasks_list, p_out, p_in = parser_houses(inpout)
    game = HousesGame(nb_houses, nb_intervalle, energy_production, pref_tasks_list, p_in, p_out)
    strategies = game.backward()
    backward_house_to_dot(game,strategies, "backward.dot")


    mat = game.graph.list_succ_to_mat(game.graph.succ, True, game.player)
    game.graph.mat = mat

    list_pred = Graph.matrix_to_list_pred(mat)
    game.graph.pred = list_pred
    print game.graph.max_weight
    a_star = game.best_first_search(ReachabilityGame.a_star_negative,None, float("infinity"), True, True)
    print "SOL", a_star




def test_dag_to_tree(inpout, output_dag, output_tree):

    nb_houses, energy_production, nb_intervalle, pref_tasks_list, p_out, p_in = parser_houses(inpout)
    game = HousesGame(nb_houses,nb_intervalle, energy_production, pref_tasks_list)
    new_succ, new_vertices = game.dag_to_tree()
    tree_house_to_dot(new_succ,game.goal,[],game.graph.vertex,new_vertices,output_tree)
    graph_house_to_dot(game, output_dag)





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

def memo_test():

    t1 = Task(1)
    t2 = Task(2)
    t3 = Task(1)
    t4 = Task(2)

    p1 = Pref_task(t1,{1,3})
    p2 = Pref_task(t2, {2})

    p3 = Pref_task(t3,{1,3})
    p4 = Pref_task(t4, {2})

    list1 = [p1,p2]
    #list2 = [p3,p4]
    list2 = copy.copy(list1)

    print p1 == p3
    print p2 == p4

    print p1.pref_set == p3.pref_set
    print tuple([tuple(list1),tuple([])]) == tuple([tuple(list2),tuple([])])

    print hash((1, (1,2), tuple([tuple(list1),tuple([])]))) == hash((1, (1,2), tuple([tuple(list2),tuple([])])))
    print (1, (1,2), tuple([tuple(list1),tuple([])]))

    memo = {}
    vertex = HousesVertex(1,1,2,[list1,[]])
    memo[(1,(1,2),tuple([tuple(list1),tuple([])]))] = vertex
    print "memo", memo
    print memo.get((1,(1,2),tuple([tuple(list2),tuple([])])))

def test():

    v1 = HousesVertex(1,2,3,[])
    v2 = HousesVertex(v1.id,v1.player, v1.time, v1.tasks)

    v2.time = 42
    v2.tasks.append(3)

    print v1.time, v2.time
    print v1.tasks



if __name__ == '__main__':



    #first_simple_test()
    #parser_test()
    #run_dot_parser_test()
    #memo_test()
    #test_dag_to_tree("file_houses.txt", "graph_houses.dot","tree_houses.dot")
    test_backward("file_houses.txt")

