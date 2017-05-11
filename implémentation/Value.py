from GraphGame import Vertex
from GraphGame import Graph
from GraphGame import ReachabilityGame
import GraphToDotConverter
import copy
import numpy as np
import scipy.sparse.csgraph

def compute_value_with_negative_weight(graph, goal):

    V = len(graph.vertex)
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
        print tab_value
        old_tab_value = copy.deepcopy(tab_value)
        iter +=1

        for v in graph.vertex:

            if not (v.id in goal):



                if v.player == 1: # on cherche a minimiser
                    compute_min = min_succ_value(graph.succ[v.id], old_tab_value)
                    tab_value[v.id] = compute_min[0]
                    if tab_value[v.id] != old_tab_value[v.id]:
                        min_1[v.id] = compute_min[1]
                        if old_tab_value[v.id] == float("infinity"):
                            min_2[v.id] = compute_min[1]



                else: # on cherche a maximiser
                    compute_max = max_succ_value(graph.succ[v.id], old_tab_value)
                    tab_value[v.id] = compute_max[0]
                    max[v.id] = compute_max[1]

                if tab_value[v.id] < -(V - 1) * W:
                    tab_value[v.id] = -float("infinity")



    return (tab_value, min_1, min_2, max)







def min_succ_value(succ, tab_value):

    min = float("infinity")
    arg_min = None

    for p in succ:

        temp = p[1] + tab_value[p[0]]
        if temp < min:
            min = temp
            arg_min = p[0]
    return (min, arg_min)

def max_succ_value(succ, tab_value):


    max = - float("infinity")
    arg_max = None

    for p in succ:

        temp = p[1] + tab_value[p[0]]
        if temp > max:
            max = temp
            arg_max = p[0]
    return (max , arg_max)





def example_from_the_article():
    W = 10

    v0 = Vertex(0, 2)
    v1 = Vertex(1, 1)
    v2 = Vertex(2, 2)

    vertex = [v0, v1, v2]

    succ0 = [(1, -1), (2, -W)]
    succ1 = [(0, 0), (2, 0)]
    succ2 = [(2, 0)]

    succ = [succ0, succ1, succ2]

    graph = Graph(vertex, None, None, succ, W)

    target = {2}

    res = compute_value_with_negative_weight(graph, target)

    game = ReachabilityGame(2, graph, None, target, None)

    GraphToDotConverter.minMax_graph_to_dot(game, res[0], "test.dot")

    print res

def other_example():

    v0 = Vertex(0, 1)
    v1 = Vertex(1, 1)
    v2 = Vertex(2, 1)
    v3 = Vertex(3, 2)

    vertex = [v0, v1, v2, v3]

    succ1 = [(1, 0), (3, 0)]
    succ2 = [(0, -1), (2, 10)]
    succ3 = [(2, 0)]
    succ4 = [(3, 0)]

    succ = [succ1, succ2, succ3, succ4]

    graph = Graph(vertex, None, None, succ, 10)

    target = {2}

    res = compute_value_with_negative_weight(graph, target)

    game = ReachabilityGame(2,graph,None,target,None)

    GraphToDotConverter.minMax_graph_to_dot(game,res[0], "test2.dot")

    print res


def example_from_master_thesis():

    v0 = Vertex(0,1)
    v1 = Vertex(1,2)
    v2 = Vertex(2,1)
    v3 = Vertex(3,1)
    v4 = Vertex(4,1)
    v5 = Vertex(5,2)
    v6 = Vertex(6,1)
    v7 = Vertex(7,1)

    vertex = [v0, v1, v2, v3, v4, v5, v6, v7]

    succ0 = [(1,1)]
    succ1 = [(4, 1), (2, 1)]
    succ2 = [(2, 0)]
    succ3 = [(5, -1)]
    succ4 = [(6, -1)]
    succ5 = [(6,1),(7,-1)]
    succ6 = [(7,1),(4,0)]
    succ7 = [(7,0)]

    succ = [succ0,succ1, succ2, succ3, succ4, succ5, succ6, succ7]

    graph = Graph(vertex, None, None, succ, 1)

    target = {7}

    res = compute_value_with_negative_weight(graph, target)

    game = ReachabilityGame(2, graph, None, target, None)

    GraphToDotConverter.minMax_graph_to_dot(game, res[0], "from_master_thesis_example.dot")

    print res


def from_simon_example():
    v0 = Vertex(0, 2)
    v1 = Vertex(1, 1)
    v2 = Vertex(2, 1)

    vertex = [v0, v1, v2]

    succ1 = [(1, -1), (2, -25)]
    succ2 = [(0, 0), (2, 0)]
    succ3 = []

    succ = [succ1, succ2, succ3]

    graph = Graph(vertex, None, None, succ, 25)

    target = {2}

    res = compute_value_with_negative_weight(graph, target)

    game = ReachabilityGame(2, graph, None, target, None)

    GraphToDotConverter.minMax_graph_to_dot(game, res[0], "test3.dot")

    print res

def is_there_infinite_value_test():

    v0 = Vertex(0,1)
    v1 = Vertex(1,1)
    v2 = Vertex(2,1)
    v3 = Vertex(3,2)
    v4 = Vertex(4,1)
    v5 = Vertex(5,1)

    vertex = [v0, v1, v2, v3, v4, v5]

    succ0 = [(1,1),(2,1)]
    succ1 = [(1,0)]
    succ2 = [(3,-5)]
    succ3 = [(5,50),(4,-1)]
    succ4 = [(2,-1),(5,1)]
    succ5 = [(5,0)]

    succ = [succ0, succ1, succ2, succ3, succ4, succ5]

    graph = Graph(vertex, None, None, succ, 50)

    target = {5}

    res = compute_value_with_negative_weight(graph, target)


    game = ReachabilityGame(2, graph, None, target, None)

    GraphToDotConverter.minMax_graph_to_dot(game, res[0], "no_inifinity.dot")

    print res





def test():
    g = np.array([[0,0,-2,0],[4,0,3,0],[0,0,0,2],[0,-1,0,0]])
    a = scipy.sparse.csgraph.floyd_warshall(g)
    print a
    print np.amin(a)




if __name__ == '__main__':
    #is_there_infinite_value_test()
    #other_example()
    example_from_master_thesis()













