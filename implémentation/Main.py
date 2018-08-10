from GraphGame import *
from Value import *

def one_aleatory_test(nb_vertex,min_weight = 1, max_weight = 100):

    """
    Genere aleatoirement un jeu a deux joueurs pour lequel chaque joueur a un objectif (ces deux objectis etant differents)

    :param nb_vertex: nombre de sommets du graphe du jeu
    :param min_weight: poids minimal sur les aretes du graphe
    :param max_weight: poids maximal sur les aretes du graphe
    :return: un jeu
    """


    possible_goal = range(0, nb_vertex)
    random.shuffle(possible_goal)
    goal_1 = possible_goal.pop()
    goal_2 = possible_goal.pop()

    possible_init = range(0, nb_vertex)
    random.shuffle(possible_init)
    init = possible_init.pop()

    game = ReachabilityGame.generate_game(2, nb_vertex, init, [{goal_1}, {goal_2}], min_weight, max_weight)

    return game


def compute_a_star(game, allowed_time = float("infinity")):

    res = game.best_first_search(ReachabilityGame.a_star_positive, None, allowed_time)
    return res


def compute_random_method(game, nbr_path=100):

    set_res = game.test_random_path(nbr_path, game.compute_max_length())
    best_res = game.filter_best_result(set_res)
    return best_res


def compute_init_best_first_search(game, allowed_time = float("infinity")):
    init = game.best_first_search_with_init_path_both_two(ReachabilityGame.a_star_positive, allowed_time)
    return init

def compute_breadth_first_search_first(game):
    first = game.breadth_first_search()
    return first

def compute_breadth_first_search(game, allowed_time = float("infinity")):
    breadth = game.breadth_first_search(False, allowed_time)
    breadth_result = game.filter_best_result(breadth)
    return breadth_result

BEST = 0
SAME = 1
EQUIVALENT = 2
BOTH_FAIL = 3
WORST = 4

def compare_one_method_to_other_method(game, path1, path2):

    """
    Compare si pour un certain jeu donne un resultat: path1 et un autre resultat. Determine sur path1 est meilleur, pire,
    egal, equivalent a path2.
    :param game: un jeu d atteignabilite
    :param path1: un resultat
    :param path2: un autre resultat
    :return: la comparaison entre les deux resultats
    """


    if path1 is None and path2 is None:
        return BOTH_FAIL
    else:
        if path1 is None:
            return WORST
        else:
            if path2 is None:
                return BEST
            else:

                (_cost1,reached_player1) = game.get_info_path(path1)
                (_cost2,reached_player2) = game.get_info_path(path2)

                cost1 = game.cost_for_all_players(path1, True)
                cost2 = game.cost_for_all_players(path2, True)

                if len(reached_player1) > len(reached_player2) :
                    return BEST
                else:
                    if len(reached_player1) < len(reached_player2):
                        return WORST
                    else:
                        sum_1 = 0
                        for p in cost1.keys():
                            sum_1 += cost1[p]
                        sum_2 = 0
                        for j in cost2.keys():
                            sum_2 += cost2[j]

                        if sum_1 < sum_2:
                            return BEST
                        else:
                            if sum_1 > sum_2:
                                return WORST
                            else:
                                if ReachabilityGame.same_paths(path1,path2):
                                    return SAME
                                else:
                                    return EQUIVALENT


def compute_stat(file_name, allowed_time = float("infinity")):

    """
    Test l' algorithme best-first search auquel on associe la fonction d evaluation de type A* sur des graphes a n noeuds
    ou 5 <= n  <= 20, 100 tests pour chaque n differents
    :param file_name: nom du fichier dans lequel stocker les resultats
    :param allowed_time: temps permis pour l execution d une methode
    """

    stat = open(file_name, "a")

    for nb_v in range(5, 21):
        stat.write("******************* \n")

        stat.write(" test avec "+str(nb_v)+" sommets \n")
        stat.write(" temps permis pour chaque recherche "+str(allowed_time)+"\n")

        compt_find = 0
        compt_not_find = 0
        debut = time.time()

        for i in range(0, 100):
            game = one_aleatory_test(nb_v)
            res = compute_a_star(game, allowed_time)

            if res is not None:
                compt_find += 1
            else:
                compt_not_find +=1

        fin = time.time()

        print "fini pour ", nb_v, " sommets en ", fin - debut, " secondes"

        stat.write(" nb EN trouves "+str(compt_find)+"\n")
        stat.write(" nbr EN manques "+str(compt_not_find)+"\n")
        stat.write("temps ecoule "+str(fin - debut)+"\n")

    stat.close()

def compute_stat_all_method(file_name, allowed_time = float("infinity")):

    """
    Execute les differents resultats sur 100 jeux donc le graphe est compose de 5 noeuds et genere aleatoirement et
    ecrit les resultats dans un fichier.
    :param file_name: nom du fichier dans lequel inscrire les resultats
    :param allowed_time: temps permis pour l execution d une methode

    """

    stat = open(file_name, "a")

    for nb_v in range(5, 6):
        stat.write("******************* \n")

        stat.write(" test avec " + str(nb_v) + " sommets \n")
        stat.write(" temps permis pour chaque recherche " + str(allowed_time) + "\n")

        compt_find_a_star = 0
        compt_find_random = 0
        compt_find_init = 0
        compt_find_breadth_first = 0
        compt_find_breadth = 0

        real_debut = time.time()


        fin_a_star = 0
        fin_random = 0
        fin_init = 0
        fin_breadth_first = 0
        fin_breadth = 0

        best_vs_random = 0
        best_vs_init = 0
        best_vs_breadth_first = 0
        best_vs_breadth = 0

        worst_vs_random = 0
        worst_vs_init = 0
        worst_vs_breadth_fisrt = 0
        worst_vs_breadth = 0

        both_fail_vs_random = 0
        both_fail_vs_init = 0
        both_fail_vs_breadth_first = 0
        both_fail_vs_breadth = 0

        equivalent_vs_random = 0
        equivalent_vs_init = 0
        equivalent_vs_breadth_first = 0
        equivalent_vs_breadth = 0

        same_vs_random = 0
        same_vs_init = 0
        same_vs_breadth_first = 0
        same_vs_breadth = 0




        for i in range(0, 20):
            print "------"
            print "Jeu numero: ", i + 1
            game = one_aleatory_test(nb_v)

            debut = time.time()
            a_star = compute_a_star(game, allowed_time)
            fin = time.time()
            fin_a_star = fin_a_star + (fin - debut)

            print " a_star"

            debut = fin
            random = compute_random_method(game, 100)
            fin = time.time()
            fin_random = fin_random + (fin - debut)

            print "random"

            debut = fin
            init = compute_init_best_first_search(game, allowed_time)
            fin = time.time()
            fin_init  = fin_init + (fin - debut)

            print "init"

            debut = fin
            breadth_first = compute_breadth_first_search_first(game)
            fin = time.time()
            fin_breadth_first = fin_breadth_first + (fin - debut)

            print "breadth"

            #debut = fin
            #breadth = compute_breadth_first_search(game, allowed_time)
            #fin = time.time()
            #fin_breadth = fin_breadth + (debut - fin)

            if a_star is not None:
                compt_find_a_star += 1
            if random is not None:
                compt_find_random += 1
            if init is not None:
                compt_find_init += 1
            if breadth_first is not None:
                compt_find_breadth_first += 1
            #if breadth is not None:
            #    compt_find_breadth += 1

            #on compare les resultats obtenus avec celui de A*

            vs_random = compare_one_method_to_other_method(game, a_star, random)

            if vs_random == BEST:
                best_vs_random += 1
            if vs_random == WORST:
                worst_vs_random += 1
            if vs_random == BOTH_FAIL:
                both_fail_vs_random += 1
            if vs_random == SAME:
                same_vs_random += 1
            if vs_random == EQUIVALENT:
                equivalent_vs_random += 1

            vs_init = compare_one_method_to_other_method(game, a_star, init)

            if vs_init == BEST:
                best_vs_init += 1
            if vs_init == WORST:
                worst_vs_init += 1
            if vs_init == BOTH_FAIL:
                both_fail_vs_init += 1
            if vs_init == SAME:
                same_vs_init += 1
            if vs_init == EQUIVALENT:
                equivalent_vs_init += 1

            vs_breadth_first = compare_one_method_to_other_method(game, a_star, breadth_first)
            if vs_breadth_first == BEST:
                best_vs_breadth_first += 1
            if vs_breadth_first == WORST:
                worst_vs_breadth_fisrt += 1
            if vs_breadth_first == BOTH_FAIL:
                both_fail_vs_breadth_first += 1
            if vs_breadth_first == SAME:
                same_vs_breadth_first += 1
            if vs_breadth_first == EQUIVALENT:
                equivalent_vs_breadth_first += 1

            #vs_breadth = compare_one_method_to_other_method(game, a_star, breadth)
            #if vs_breadth == BEST:
            #    best_vs_breadth += 1
            #if vs_breadth == WORST:
            #    worst_vs_breadth +=1
            #if vs_breadth == BOTH_FAIL:
            #    both_fail_vs_breadth += 1
            #if vs_breadth_first == SAME:
            #    same_vs_breadth += 1
            #if vs_breadth == EQUIVALENT:
            #    equivalent_vs_breadth += 1

        real_fin  = time.time()
        print "fini en ", real_fin - real_debut, " secondes"

        stat.write("A_STAR : \n")
        stat.write("Nb EN trouves " + str(compt_find_a_star) + "\n")
        stat.write("temps ecoule " + str(fin_a_star) + "\n")
        stat.write("----------------------- \n")
        stat.write("RANDOM : \n")
        stat.write("Nb EN trouves " + str(compt_find_random) + "\n")
        stat.write("temps ecoule " + str(fin_random) + "\n")
        stat.write("----------------------- \n")
        stat.write("INIT BEST : \n")
        stat.write("Nb EN trouves " + str(compt_find_init) + "\n")
        stat.write("temps ecoule " + str(fin_init) + "\n")
        stat.write("----------------------- \n")
        stat.write("BREADTH FIRST : \n")
        stat.write("Nb EN trouves " + str(compt_find_breadth_first) + "\n")
        stat.write("temps ecoule " + str(fin_breadth_first) + "\n")
        stat.write("----------------------- \n")
        #stat.write("BREADTH MULTI : \n")
        #stat.write("Nb EN trouves " + str(compt_find_breadth) + "\n")
        #stat.write("temps ecoule " + str(fin_breadth) + "\n")
        #stat.write("----------------------- \n")
        stat.write("STAT SUR A_STAR : \n")
        stat.write("A_star vs random: meilleur " + str(best_vs_random)+" pire "+ str(worst_vs_random) +" pareil :" + str(same_vs_random) + " equivalent :"+ str(equivalent_vs_random) + " manque tous les deux :"+ str(both_fail_vs_random)+ "\n")
        stat.write("A_star vs init : meilleur " + str(best_vs_init)+" pire "+ str(worst_vs_init) +" pareil " + str(same_vs_init) + " equivalent "+ str(equivalent_vs_init) + " manque tous les deux "+ str(both_fail_vs_init)+ "\n")
        stat.write("A_star vs breadth first: meilleur " + str(best_vs_breadth_first)+ " pire " + str(worst_vs_breadth_fisrt) +" pareil " + str(same_vs_breadth_first) + " equivalent "+ str(equivalent_vs_breadth_first) + " manque tous les deux "+ str(both_fail_vs_breadth_first)+ "\n")
        #stat.write("A_star vs breadth multi: meilleur " + str(best_vs_breadth) +" pareil " + str(same_vs_breadth) + " equivalent "+ str(equivalent_vs_breadth) + "manque tous les deux"+ str(both_fail_vs_breadth)+ "\n")
        stat.write("----------------------- \n")
        stat.close()


def real_EN_test(nb_v, allowed_time = float("infinity")):

    stat = open("real_EN.txt", "a")
    stat.write("*****************\n")
    stat.write("Nouveau test \n")

    compt_find = 0
    compt_not_find = 0
    real_EN = 0
    debut = time.time()

    for i in range(0, 100):
        game = one_aleatory_test(nb_v)
        res = compute_a_star(game, allowed_time)
        (nash,coal) = game.is_a_Nash_equilibrium(res)

        if res is not None:
            compt_find += 1
            if nash:
                real_EN += 1
        else:
            compt_not_find += 1

    fin = time.time()
    print "fini pour ", nb_v, " sommets en ", fin - debut, " secondes"

    stat.write(" nb EN trouves " + str(compt_find) + "\n")
    stat.write(" nbr EN manques " + str(compt_not_find) + "\n")
    stat.write(" EN reel " + str(real_EN) + "\n")
    stat.write("temps ecoule " + str(fin - debut) + "\n")


    stat.close()

def restart_test(times):

    for i in range(0, times):
        real_EN_test(5)


if __name__ == '__main__':
    #compute_stat(30)
    #restart_test(10)
    compute_stat_all_method("stat_all_method.txt",30)





