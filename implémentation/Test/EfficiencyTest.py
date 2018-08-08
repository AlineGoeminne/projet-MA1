import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from RandomGame import random_game
from GraphGame import ReachabilityGame, Graph
import numpy as np

if __name__ == "__main__":
    from GraphToDotConverter import random_graph_to_dot
    import timeit

    for size in xrange(2, 20):
        time = []
        for i in xrange(100):
            print "start", i
            game = random_game(size, 1, 2, 0.1, False, 30, True, 2)
            print "graph ok"
            t = timeit.default_timer()
            game.best_first_search(heuristic=ReachabilityGame.a_star_negative, tuple_=True, negative_weight=True)
            time.append(timeit.default_timer() - t)
            print "end", i
        print size, np.mean(time)