from GraphGame import Graph, Vertex, ReachabilityGame
import random
import numpy as np
from copy import deepcopy

def random_graph(size, lowOutgoing, upOutgoing, maxWeight, nPlayers, probaPlayers, probaTarget, maximumTarget):
    """
    :param size: le nombre de noeuds du graphe
    :type size: integer

    :param lowOutgoing: la borne inferieur sur le nombre d'arcs sortants par noeud
    :type lowOutgoing: integer

    :param upOutgoing: la borne superieure sur le nombre d'arcs sortants par noeud
    :type upOutgoing: integer

    :param maxWeight: le poids maximal

    :param nPlayers: le nombre de joueurs
    :type nPlayers: integer

    :param probaPlayers: pour chaque joueur, la probabilite qu'un noeud lui appartienne. La somme du tableau doit valoir 1.
    :type probaPlayers: tableau de float de taille nPlayers

    :param probaTarget: pour chaque joueur, la probabilite qu'un noeud soit un goal pour lui

    :param maximumTarget: pour chaque joueur, le nombre maximum de goals qu'il peut avoir

    :return (graph, init, target, partition)
    """
    assert(np.sum(probaPlayers) == 1)

    vertex = []
    succ = []

    target = []
    partition = []

    for i in xrange(nPlayers):
        partition.append([])
        target.append([])

    playersList = np.arange(1, nPlayers+1)

    for i in xrange(size):
        # On donne le noeud a un joueur
        player = np.random.choice(playersList, p=probaPlayers)
        node = Vertex(i, player)
        partition[player-1].append(node)
        vertex.append(node)

        # Le noeud peut etre un goal
        for p in xrange(len(probaTarget)):
            if random.random() <= probaTarget[p]:
                # Si l'aleatoire fait que c'est possible de declarer node comme goal pour le joueur p
                if len(target[p]) < maximumTarget[p]:
                    # Si on n'a pas encore depasse la limite
                    target[p].append(node)

        # On cree les arcs sortants
        outgoing = []
        e = random.randint(lowOutgoing, upOutgoing)
        for _ in xrange(e):
            weigth = random.randint(0, maxWeight)
            if random.random() <= 1/2:
                weigth = -weigth
            outgoing.append((random.randint(0, size), weigth))
        succ.append(outgoing)

    init = random.choice(vertex)

    # Si un joueur n'a pas de target, on lui en donne un
    for i in xrange(nPlayers):
        if len(target[i]) == 0:
            target[i].append(random.choice(vertex))

    graph = Graph(vertex, None, [], succ)
    return graph, init, target, partition

if __name__ == "__main__":
    from GraphToDotConverter import random_graph_to_dot

    graph, init, target, partition = random_graph(10, 1, 3, 10, 2, [0.5, 0.5], [0.1, 0.05], [4, 2])
    print(target)

    random_graph_to_dot(graph, init, target, "/tmp/test.dot")
