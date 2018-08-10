import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from GraphGame import *
import numpy as np



def a_star_test1():
    mat = [[np.inf, 7, 9, 5, 4],[5, np.inf, 2, 9, 3],[6, 4, np.inf, 9, 3],[2, 7, 2, np.inf, 10],[8, 9, 7, 3, np.inf]]

    #mat_np = np.matrix(mat)
    #G2_sparse = csgraph_from_dense(mat_np, null_value=np.inf)

    #res = sparse.csgraph.johnson(G2_sparse)
    #print "PLUS COURTS CHEMINS ", res



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

    game = ReachabilityGame(2, graph, v3, goals, None)
    a_star = game.best_first_search(ReachabilityGame.a_star_positive, None, 5)

    init = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star_positive, 30)

    candidate = game.best_first_search(ReachabilityGame.short_path_evaluation, None, 5)
    first = game.breadth_first_search()
    breadth = game.breadth_first_search(False, 1)
    breadth_result = game.filter_best_result(breadth)

    random = game.test_random_path(100, game.compute_max_length())
    random_result = game.filter_best_result(random)


    if a_star is not None:

        print "A_star :", str(a_star)
        (nash1, coal) = game.is_a_Nash_equilibrium(a_star)
        print "Est un EN? ", nash1
        (cout, atteint) = game.get_info_path(a_star)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"
    else:
        print "A_star a echoue"

    if random_result is not None:
        print "Random : ", str(random_result)
        (nash2, coal) = game.is_a_Nash_equilibrium(random_result)
        print "Est un EN? ", nash2
        (cout, atteint) = game.get_info_path(random_result)
        print "Information sur l'outcome :\nCout pour chaque joueur: ", cout, " joueur ayant atteint leur objectif ", atteint
        print"\n"
    else :
        print "Random result a echoue"


    if candidate is not None:
        print "Best-first search shortest_path ", str(candidate)
        (nash3, coal) = game.is_a_Nash_equilibrium(candidate)
        print "Est un EN? ", nash3
        (cout, atteint) = game.get_info_path(candidate)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"
    else:
        print "Best first search shortest path a echoue"

    if init is not None:
        print "Init best-first search ", str(init)
        (nash4, coal) = game.is_a_Nash_equilibrium(init)
        print "Est un EN? ", nash4
        (cout, atteint) = game.get_info_path(init)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"
    else :
        print "Init best-first search a echoue"

    if first is not None:
        print "Breadth-first search first", str(first)
        (nash5, coal) = game.is_a_Nash_equilibrium(first)
        print "Est un EN? ", nash5
        (cout, atteint) = game.get_info_path(first)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"
    else:
        print "Breadth-girst search first a echoue"

    if breadth_result is not None:

        print "Breadth-first search multi", str(breadth_result)
        (nash6, coal) = game.is_a_Nash_equilibrium(breadth_result)
        print "Est un EN? ", nash6
        (cout, atteint) = game.get_info_path(breadth_result)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"
    else:
        print "Breadth first search multi a echoue"




def a_star_test2():
    mat = [[np.inf, 9, 3, 3, 5],[3, np.inf, 3, 9, 1],[9, 8, np.inf, 10, 1],[8, 6, 3, np.inf, 8],[4, 4, 9, 4, np.inf]]

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

    game = ReachabilityGame(2, graph, v3, goals, None)

    res = game.best_first_search(ReachabilityGame.a_star_positive, None, 5)

    print "a_star", str(res)
    print "EN ? ", game.is_a_Nash_equilibrium(res)

    res2 = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star_positive, 5)

    print "Init ", str(res2)
    if not res2 is None:
        print "En? ", game.is_a_Nash_equilibrium(res2)

def test_best_first_search():

        #sur l'exemple "etrange"

        v0 = Vertex(0, 1)
        v1 = Vertex(1, 2)
        v2 = Vertex(2, 1)
        v3 = Vertex(3, 1)

        vertex = [v0, v1, v2, v3]

        pred0 = [(1, 1), (2, 1)]
        pred1 = [(0, 1), (3, 1)]
        pred2 = [(0, 1)]
        pred3 = [(1, 1)]

        pred = [pred0, pred1, pred2, pred3]
        succ = Graph.list_pred_to_list_succ(pred)

        graph = Graph(vertex, None, pred, succ, 1)

        goal = [{3}, {2}]

        game = ReachabilityGame(2, graph, v0, goal, None)

        candidate = game.best_first_search(ReachabilityGame.short_path_evaluation, None, 5)
        candidate_a_star = game.best_first_search(ReachabilityGame.a_star_positive, None, 5)

        random = game.test_random_path(100, game.compute_max_length())
        random_result = game.filter_best_result(random)

        init = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star_positive, 5)

        print "A_star :", str(candidate_a_star)
        (nash1, coal) = game.is_a_Nash_equilibrium(candidate_a_star)
        print "Est un EN? ", nash1
        (cout,atteint) = game.get_info_path(candidate_a_star)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"

        print "Best-first search shortest_path ",str(candidate)
        (nash2, coal) = game.is_a_Nash_equilibrium(candidate)
        print "Est un EN? ", nash2
        (cout, atteint) = game.get_info_path(candidate)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"

        print "Init best-first search :", str(init)
        (nash, coal) = game.is_a_Nash_equilibrium(init)
        print "Est un EN? ", nash
        (cout, atteint) = game.get_info_path(init)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
        print"\n"

        print "Random : ", str(random_result)
        (nash1, coal) = game.is_a_Nash_equilibrium(random_result)
        print "Est un EN? ", nash1
        (cout, atteint) = game.get_info_path(random_result)
        print "Information sur l'outcome :\nCout pour chaque joueur: ", cout, " joueur ayant atteint leur objectif ", atteint
        print"\n"



def test_best_first_search_2():

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

    graph = Graph(vertex, None, list_pred, list_succ, 4)
    goals = [{3}, {0}]
    init = v1
    game = ReachabilityGame(2, graph, init, goals, None)

    a_star = game.best_first_search(ReachabilityGame.a_star_positive, None, 5)


    random = game.test_random_path(100, game.compute_max_length())
    path_random = game.filter_best_result(random)
    init = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star_positive, 5)
    candidate = game.best_first_search(ReachabilityGame.short_path_evaluation, None, 5)


    print "A_star :"
    print str(a_star)
    (nash1, coal) = game.is_a_Nash_equilibrium(a_star)
    print "est un EN? ", nash1
    print "info path ", game.get_info_path(a_star)
    print "\n"


    print "random :"
    print str(path_random)
    (nash2, coal) = game.is_a_Nash_equilibrium(path_random)
    print " est un EN? ", nash2
    print "info path ", game.get_info_path(path_random)
    print "\n"

    print "Best-first search shortest_path ", str(candidate)
    (nash2, coal) = game.is_a_Nash_equilibrium(candidate)
    print "Est un EN? ", nash2
    (cout, atteint) = game.get_info_path(candidate)
    print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
    print"\n"

    print "Init best-first search ", str(init)
    (nash, coal) = game.is_a_Nash_equilibrium(init)
    print "Est un EN? ", nash
    (cout, atteint) = game.get_info_path(init)
    print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
    print"\n"



def test_best_first_search3():

    # exemple du rapport modifie

    v0 = Vertex(0, 2)
    v1 = Vertex(1, 1)
    v2 = Vertex(2, 2)
    v3 = Vertex(3, 1)
    v4 = Vertex(4, 1)
    v5 = Vertex(5, 2)
    v6 = Vertex(6, 1)
    v7 = Vertex(7, 2)

    vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

    # pred tableau des pred tq (u,k) u = index du pred, et k = valeur de l'arc

    pred0 = [(1, 1), (2, 4), (3, 5)]
    pred1 = [(2, 1)]
    pred2 = [ (0,1), (3, 4), (4, 5)]
    pred3 = [(4, 1),(2,1)]
    pred4 = [(5, 1), (3, 1)]
    pred5 = [(4, 1), (6, 2)]
    pred6 = [(5, 1), (7, 1)]
    pred7 = [(6, 1)]

    list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7]
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    graph = Graph(vertices, None, list_pred, list_succ, 5)
    goal = [{6}, {0}]
    game = ReachabilityGame(2, graph, v3, goal, None)

    a_star = game.best_first_search(ReachabilityGame.a_star_positive, None, 5)


    random = game.test_random_path(100, game.compute_max_length())
    random_result = game.filter_best_result(random)

    print "A_star :", str(a_star)
    (nash1, coal) = game.is_a_Nash_equilibrium(a_star)
    print "Est un EN? ", nash1
    (cout, atteint) = game.get_info_path(a_star)
    print "Information sur l'outcome : \nCout pour chaque joueur: ", cout, " joueurs ayant atteint leur objectif ", atteint
    print"\n"

    print "Random : ", str(random_result)
    (nash1, coal) = game.is_a_Nash_equilibrium(random_result)
    print "Est un EN? ", nash1
    (cout, atteint) = game.get_info_path(random_result)
    print "Information sur l'outcome :\nCout pour chaque joueur: ", cout, " joueur ayant atteint leur objectif ", atteint
    print"\n"



def general_test():

    nb_vertex = 10
    poids_min = 1
    poids_max = 100
    allowed_time = 5

    possible_goal = range(0, nb_vertex)
    random.shuffle(possible_goal)
    goal_1 = possible_goal.pop()
    goal_2 = possible_goal.pop()

    possible_init = range(0, nb_vertex)
    random.shuffle(possible_init)
    init = possible_init.pop()
    print "Noeud initial v"+str(init)

    print "Objectifs : joueur 1: v"+str(goal_1)+" joueur 2: v"+str(goal_2)
    game = ReachabilityGame.generate_game(2, nb_vertex, init,[{goal_1}, {goal_2}], poids_min, poids_max)
    print " Partition", game.part
    mat = Graph.list_succ_to_mat(game.graph.succ)
    print "\n"
    print Graph.mat_to_string(mat)
    print "\n"
    print "----------"






    init = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star_positive, allowed_time)
    if init is not None:
        print "init :", str(init)
        (cout1, atteint1) = game.get_info_path(init)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout1, " joueurs ayant atteint leur objectif ", atteint1
        print "En?:", game.is_a_Nash_equilibrium(init)
        print "\n"

    else:
        print "init a echoue \n"

    a_star = game.best_first_search(ReachabilityGame.a_star_positive, None, allowed_time)
    if a_star is not None:
        print "A_star :", str(a_star)
        (cout2, atteint2) = game.get_info_path(a_star)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout2, " joueurs ayant atteint leur objectif ", atteint2
        print "En?:", game.is_a_Nash_equilibrium(a_star)
        print "\n"

    else:
        print "a_star a echoue \n"

    res = game.test_random_path(100, game.compute_max_length())
    result_random = game.filter_best_result(res)
    if result_random is not None:
        print "Methode aleatoire :",str(result_random)
        (cout3, atteint3) = game.get_info_path(result_random)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout3, " joueurs ayant atteint leur objectif ", atteint3
        print "En?:", game.is_a_Nash_equilibrium(result_random)
        print "\n"

    else:
        print "Methode aleatoire a echoue \n"


    breadth = game.breadth_first_search(True, allowed_time)
    if breadth is not None:
        print "Breadth-first search :", str(breadth)
        (cout4, atteint4) = game.get_info_path(breadth)
        print "Information sur l'outcome : \nCout pour chaque joueur: ", cout4, " joueurs ayant atteint leur objectif ", atteint4
        print "EN ? :", game.is_a_Nash_equilibrium(breadth)
        print "\n"

    else:
        print "Breadth-first a echoue\n"



def echec_init():

    v0 = Vertex(0,2)
    v1 = Vertex(1,2)
    v2 = Vertex(2,2)
    v3 = Vertex(3,2)
    v4 = Vertex(4,2)
    v5 = Vertex(5,1)
    v6 = Vertex(6,1)
    v7 = Vertex(7,1)
    v8 = Vertex(8,2)
    v9 = Vertex(9,1)

    vertex = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]

    init = v2

    mat  = [[np.inf, 62, 16, 50, 25, 72, 19, 53, 13, 96],[65, np.inf, 87, 48, 99, 9, 98, 21, 67, 87],[25, 8, np.inf, 79, 92, 27, 62, 8, 80, 80],[86, 60, 96, np.inf, 78, 11, 78, 67, 61, 28],[19, 76, 56, 87, np.inf, 29, 11, 7, 42, 60],[20, 56, 58, 28, 63, np.inf, 86, 71, 65, 8],[63, 51, 1, 58, 5, 63, np.inf, 54, 81, 9],[6, 24, 70, 69, 7, 53, 5, np.inf, 93, 24],[79, 7, 41, 91, 11, 28, 92, 91, np.inf, 51],[72, 88, 63, 80, 30, 9, 94, 57, 42, np.inf]]
    list_pred = Graph.matrix_to_list_pred(mat)
    list_succ = Graph.list_pred_to_list_succ(list_pred)
    graph = Graph(vertex, mat, list_pred, list_succ, 100)
    game = ReachabilityGame(2, graph, init, [{6},{8}] , None)

    res = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star_positive, 30)
    print res
    res2 = game.best_first_search(ReachabilityGame.a_star_positive, None, 30)
    print res2
    print game.is_a_Nash_equilibrium(res2)

def test_slide():
    v0 = Vertex(0, 2)
    v1 = Vertex(1, 1)
    v2 = Vertex(2, 2)
    v3 = Vertex(3, 1)
    v4 = Vertex(4, 1)
    v5 = Vertex(5, 2)
    v6 = Vertex(6, 1)
    v7 = Vertex(7, 2)

    vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

    pred0 = [(1, 1), (2, 1), (3, 5)]
    pred1 = [(0, 1), (2, 1)]
    pred2 = [(3, 1), (4, 5)]
    pred3 = [(4, 1)]
    pred4 = [(1, 1), (5, 1)]
    pred5 = [(4, 1)]
    pred6 = [(5, 1), (7, 1)]
    pred7 = [(6, 1)]

    list_pred = [pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7]
    list_succ = Graph.list_pred_to_list_succ(list_pred)

    graph = Graph(vertices, None, list_pred, list_succ, 5)
    goal1 = {6}
    goal2 = {0}
    init = v3

    game = ReachabilityGame(2, graph, init, [goal1, goal2] , None)
    a_star = game.best_first_search(ReachabilityGame.a_star_positive, None, 30)
    print a_star




if __name__ == '__main__':


    #a_star_test1()
    #a_star_test2()

    #test_best_first_search()
    #test_best_first_search_2()
    #test_best_first_search3()

    #general_test()
    test_slide()

    #echec_init()
