from GraphGame import ReachabilityGame
from GraphGame import Vertex
from GraphGame import Graph
from GraphToDotConverter import graph_house_to_dot
import random
import copy

class Task(object):


    def __init__(self, id):

        self.id = id

    def __str__(self):
        return repr(self.id)

class TaskFactory(object):
    number_tasks = 0

    def create_task(self):
        TaskFactory.number_tasks += 1
        task = Task(TaskFactory.number_tasks)
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

    def compute_weight(self,vertex2):
        return 1

class MemoHousesVertex(object):

    def __init__(self):

        self.all_vertices = {}

    def get_memo(self, time, player, list_pref_tasks_for_all):
        return self.all_vertices.get((time, player, tuple(map(tuple,list_pref_tasks_for_all))))

    def put_memo(self, time, player,list_pref_tasks_for_all, vertex):

        self.all_vertices[(time, player, tuple(map(tuple,list_pref_tasks_for_all)))] = vertex


class HousesVertexFactory(object):

    number_houses_vertex = 0

    def create_houses_vertex(self, player,time,list_pref_tasks_for_all):
        vertex = HousesVertex(HousesVertexFactory.number_houses_vertex,player, time, list_pref_tasks_for_all)
        HousesVertexFactory.number_houses_vertex += 1
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
    def all_compleated(list_pref_task_for_all):


        for x in list_pref_task_for_all:
            if len(x) !=0:
                return False
        return True

    @staticmethod
    def aux_generate_houses_game(nb_player,
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
        vertex = None
        #vertex = memo.get_memo(time, player, list_pref_tasks_for_all)
        #print "memo", str(memo.all_vertices)

        if vertex is None:
            new_vertex = True
            vertex = factory.create_houses_vertex(player, time, list_pref_tasks_for_all)
            all_vertices.append(vertex)
            list_succ.append([])
            #memo.put_memo(time, player, list_pref_tasks_for_all, vertex)
            for p in xrange(1, nb_player+1):

                if HousesGame.is_an_objective(vertex, p):
                    goal[p - 1].add(vertex.id)

        turn_player.pop()

        if (time == nbr_interval and len(turn_player) == 0)\
                or HousesGame.all_compleated(list_pref_tasks_for_all)\
                or max_iter ==0:
            if new_vertex:
                list_succ[vertex.id].append((vertex.id,vertex.compute_weight(vertex)))

            return vertex

        else:



            old_time = time
            if len(turn_player) == 0:  # alors on a fait joue tous les joueurs pour cet intervalle de temps
                turn_player_bis = range(1, nb_player + 1)
                #random.shuffle(turn_player_bis)  # on redefinit l ordre des joueurs pour le prochain tour
                turn_player = turn_player_bis
                time_changed  = True

            actions = list_pref_tasks_for_all[player-1]  # on recupere les actions possibles pour le joueur considere
            for a in actions:

                if old_time in a.pref_set: #on verifie qu on peut exectuer cette action a ce moment la
                    new_list = copy.deepcopy(list_pref_tasks_for_all)
                    new_turn = copy.copy(turn_player)
                    new_time = time
                    time = new_time

                    new_list[player - 1].remove(a)
                    if time_changed:
                        succ = HousesGame.aux_generate_houses_game(nb_player, nbr_interval, new_list, time+1,all_vertices,list_succ, goal, memo,factory,new_turn , max_iter)
                    else:
                        succ = HousesGame.aux_generate_houses_game(nb_player, nbr_interval, new_list, time,all_vertices,list_succ, goal, memo,factory,new_turn , max_iter)

                    list_succ[vertex.id].append((succ.id, vertex.compute_weight(succ)))

            # on considere maintenant ne rien faire comme une action
            new_list = copy.deepcopy(list_pref_tasks_for_all)
            new_turn = copy.copy(turn_player)
            new_time = time
            time = new_time

            if time_changed:
                succ = HousesGame.aux_generate_houses_game(nb_player, nbr_interval, new_list, time+1, all_vertices, list_succ,goal, memo,factory, new_turn, max_iter)
            else:
                succ = HousesGame.aux_generate_houses_game(nb_player, nbr_interval, new_list, time, all_vertices, list_succ,goal, memo,factory, new_turn, max_iter)

            list_succ[vertex.id].append((succ.id, vertex.compute_weight(succ)))
            return vertex

    @staticmethod
    def generate_houses_game(nb_player, nbr_interval, list_pref_tasts_for_all):
        turn = range(1, nb_player + 1)
        #random.shuffle(turn)
        goal = []
        for x in range(nb_player):
            goal.append(set())
        time = 1
        all_vertices = []
        list_succ = []
        memo = MemoHousesVertex()
        factory = HousesVertexFactory()
        #wrap = Wraping2GenerateHousesGame(1, [], [], goal, MemoHousesVertex(), HousesVertexFactory(), turn)
        HousesGame.aux_generate_houses_game(nb_player, nbr_interval, list_pref_tasts_for_all, 1, all_vertices, list_succ, goal, memo, factory, turn)
        graph = Graph(all_vertices,None,None,list_succ)
        print factory.number_houses_vertex
        return graph,all_vertices[0],goal,None

    def __init__(self, player, nbr_interval, list_pref_tasks_for_all):

        (graph, init, goal, partition) = HousesGame.generate_houses_game(player,nbr_interval,list_pref_tasks_for_all)
        ReachabilityGame.__init__(self,player,graph,init,goal,partition)



def parser_houses(file_name):

    nb_intervalle = 0
    nb_houses = 0
    task_factory = TaskFactory()

    pref_tasks_list = []

    task_begin = False


    with open(file_name,"r") as file:
        lines = file.readlines()

        for l in lines:
            print l
            res = l.split("\n")[0]
            print res


            if res == "&":
                task_begin = True
                pref_tasks_list.append([])
                nb_houses += 1

            else:
                if task_begin:
                    pref = map(int, res.split(" "))
                    pref_set = set()
                    for x in pref:
                        pref_set.add(x)
                    task = task_factory.create_task()
                    pref_task = Pref_task(task, pref_set)
                    pref_tasks_list[-1].append(pref_task)

            if (not task_begin):
                nb_intervalle = int(res)


    file.close()

    return nb_houses, nb_intervalle, pref_tasks_list


def run_generate_graph_houses(inpout,outpout):

    nb_houses, nb_intervalle, pref_tasks_list = parser_houses(inpout)
    game = HousesGame(nb_houses,nb_intervalle,pref_tasks_list)
    graph_house_to_dot(game,outpout)









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

    first_simple_test()
    #parser_test()
    #run_dot_parser_test()

