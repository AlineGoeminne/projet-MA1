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

    nTests = 10

    for size in xrange(2, 20):
        allTimes = [0] * nTests
        game_setup = ""
        for i in xrange(nTests):
            game = random_game(size, 1, 2, 0.1, False, 30, False, True, 2)
            t = time.clock()
            game.best_first_search(heuristic=ReachabilityGame.a_star_positive, tuple_=False, negative_weight=False)
            allTimes[i] = time.clock() - t
        print size, np.median(allTimes)

    