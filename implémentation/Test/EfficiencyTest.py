import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from RandomGame import random_game
from GraphGame import ReachabilityGame, Graph
import numpy as np

if __name__ == "__main__":
    from GraphToDotConverter import random_graph_to_dot
    import time
    import os

    if os.name != "posix":
        print "This script only works under an Unix(-like) OS"
        exit()

    nTests = 1000

    size = 20
    lowOutgoing = 1
    upOutgoing = 2
    probaCycle = 0.1
    negativeWeight = False
    maxWeight = 30
    tupleWeight = False
    shareTarget = True
    nPlayers = 2

    print "Size", "Mean", "Median"
    for size in xrange(2, 51):
        allTimes = [0] * nTests
        game_setup = ""
        for i in xrange(nTests):
            game = random_game(size, lowOutgoing, upOutgoing, probaCycle, negativeWeight, maxWeight, tupleWeight, shareTarget, nPlayers)
            t = time.clock()
            try:
                game.best_first_search(heuristic=ReachabilityGame.a_star_positive, tuple_=False, negative_weight=False, allowed_time=10)
            except:
                print game.init
                random_graph_to_dot(game, sys.stdout)
                exit()
            allTimes[i] = time.clock() - t
        print size, np.mean(allTimes), np.median(allTimes)

    