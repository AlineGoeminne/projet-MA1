from GraphGame import Graph, Vertex, ReachabilityGame
import random
import numpy as np
from copy import deepcopy

maximumUsedWeight = None

def has_path_to(succ, start, goal, visited):
    """
    Determine s'il existe un chemin entre start et goal
    """
    if start >= len(succ):
        return False

    for (s, _) in succ[start]:
        if s == goal:
            return True
        if s not in visited:
            visited.append(s)
            if has_path_to(succ, s, goal, visited):
                return True
            visited.remove(s)
    return False

def random_weight(maxWeight, negativeWeight):
    weigth = random.randint(0, maxWeight)
    if negativeWeight and random.random() <= 0.5:
        weigth = -weigth
    return weigth

def player_node(i, playersList, probaPlayers, partition):
    player = np.random.choice(playersList, p=probaPlayers)
    node = Vertex(i, player)
    partition[player-1].append(node)
    return node

def target_node(nPlayers, node, probaTarget, target, maximumTarget, shareTarget):
    for p in xrange(nPlayers):
        if random.random() <= probaTarget[p]:
            # Si l'aleatoire fait que c'est possible de declarer node comme goal pour le joueur p
            if len(target[p]) < maximumTarget[p]:
                # Si on n'a pas encore depasse la limite
                target[p].add(node.id)
                if not shareTarget:
                    # Si on ne peut pas partager le goal, on s'arrete
                    return

def generate_weight(tupleWeights, maxWeight, nPlayers, negativeWeight):
    global maximumUsedWeight
    if tupleWeights:
        # On genere un poids different par joueur
        weights = []
        for k in xrange(nPlayers):
            weight = random_weight(maxWeight, negativeWeight)
            maximumUsedWeight[k] = max(abs(weight), maximumUsedWeight[k])
            weights.append(weight)
        return weights
    else:
        # On genere un seul poids
        weight = random_weight(maxWeight, negativeWeight)
        maximumUsedWeight = max(abs(weight), maximumUsedWeight)
        return weight
        

def random_game(size, lowOutgoing, upOutgoing, probaCycle, negativeWeight, maxWeight, tupleWeight, shareTarget, nPlayers, probaPlayers=None, probaTarget=None, maximumTarget=None):
    """
    Construit aleatoirement un jeu d'atteignabilite. Le graphe sous-jacent est bien construit et permet de creer des chemins infinis.

    :param size: le nombre de noeuds du graphe
    :type size: integer

    :param lowOutgoing: la borne inferieur sur le nombre d'arcs sortants par noeud
    :type lowOutgoing: integer

    :param upOutgoing: la borne superieure sur le nombre d'arcs sortants par noeud
    :type upOutgoing: integer

    :param probaCycle: la probabilite qu'un cycle est valide s'il est possible d'en creer un
    :type probaCycle: float

    :param negativeWeight: si les poids peuvent etre negatifs
    :type negativeWeight: integer

    :param maxWeight: le poids maximal
    :type maxWeight: integer

    :param tupleWeight: si les poids sur les arcs sont des tuples (une valeur par joueur) ou non
    :type tupleWeight: boolean

    :param shareTarget: si plusieurs joueurs peuvent partager la meme cible ou non
    :type shareTarget: boolean

    :param nPlayers: le nombre de joueurs
    :type nPlayers: integer

    :param probaPlayers: pour chaque joueur, la probabilite qu'un noeud lui appartienne. La somme du tableau doit valoir 1. Si None est donne, les probabilites sont les memes pour chaque joueur (1/nPlayers)
    :type probaPlayers: tableau de float de taille nPlayers ou None

    :param probaTarget: pour chaque joueur, la probabilite qu'un noeud soit un goal pour lui. Si None est donne, chaque joueur a une probabilite de 0.1
    :type probaTarget: tableau de float de taille nPlayers ou None

    :param maximumTarget: pour chaque joueur, le nombre maximum de goals qu'il peut avoir. Si None est donne, il n'y a pas de contrainte
    :type probaTarget: tableau de integers de taille nPlayers ou None

    :return un ReachabilityGame construit aleatoirement
    """
    if maximumTarget is None:
        maximumTarget = [float("inf")] * nPlayers
    if probaPlayers is None:
        probaPlayers = [1./nPlayers] * nPlayers
    if probaTarget is None:
        probaTarget = [0.1] * nPlayers

    assert(lowOutgoing >= 0)
    assert(upOutgoing >= 0)
    assert(upOutgoing <= size)
    assert(len(probaPlayers) == nPlayers)
    assert(len(probaTarget) == nPlayers)
    assert(len(maximumTarget) == nPlayers)
    assert(np.sum(probaPlayers) == 1)

    # Creation des variables utilisees
    vertex = []
    succ = []
    global maximumUsedWeight
    if tupleWeight:
        maximumUsedWeight = [0] * nPlayers
    else:
        tupleWeight = 0

    target = []
    partition = []

    for i in xrange(nPlayers):
        partition.append([])
        target.append(set())

    playersList = np.arange(1, nPlayers+1)

    for i in xrange(size):
        # On donne le noeud a un joueur
        node = player_node(i, playersList, probaPlayers, partition)
        vertex.append(node)

        # Le noeud peut etre un goal
        target_node(nPlayers, node, probaTarget, target, maximumTarget, shareTarget)

        # On cree les arcs sortants
        outgoing = []
        e = random.randint(lowOutgoing, upOutgoing)
        while e > 0:
            weights = generate_weight(tupleWeight, maxWeight, nPlayers, negativeWeight)
            edge = (random.randint(0, size-1), weights)

            # Si on n'essaye pas de creer un cycle
            # OU (si on essaye de creer un cycle et) si l'aleatoire permet de creer un cycle
            if not has_path_to(succ, edge[0], i, []) or random.random() <= probaCycle:
                # Si on n'a pas encore d'arcs en sortie, alors on ajoute systematiquement
                if len(outgoing) == 0:
                    outgoing.append(edge)
                    e -= 1
                else:
                    # On ne peut pas avoir 2 arcs dans la meme direction entre deux noeuds
                    alreadyIn = False
                    for (ed, _) in outgoing:
                        if ed == edge[0]:
                            alreadyIn = True
                    if not alreadyIn:
                        outgoing.append(edge)
                        e -= 1

        # Si le noeud n'a aucun successeur, on ajoute une boucle sur elle-meme
        if len(outgoing) == 0:
            weights = generate_weight(tupleWeight, maxWeight, nPlayers, negativeWeight)
            outgoing.append((random.randint(0, size-1), weights))

        succ.append(outgoing)

    init = random.choice(vertex)

    # Si un joueur n'a pas de target, on lui en donne un
    for i in xrange(nPlayers):
        while len(target[i]) == 0:
            t = random.choice(vertex)
            if shareTarget:
                # On autorise le partage, donc on prend d'office
                target[i].add(t.id)
            else:
                # On interdit le partage, donc on doit verifier si le noeud est deja une cible
                used = False
                for j in xrange(nPlayers):
                    if t.id in target[j]:
                        used = True
                if not used:
                    target[i].add(t.id)

    graph = Graph(vertex, None, [], succ)
    graph.mat = Graph.list_succ_to_mat(graph.succ, tuple=tupleWeight, nb_player=nPlayers)
    graph.pred = Graph.matrix_to_list_pred(graph.mat)
    graph.max_weight = maximumUsedWeight

    game = ReachabilityGame(nPlayers, graph, init, target, partition)
    return game

if __name__ == "__main__":
    from GraphToDotConverter import random_graph_to_dot
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Randomly generate reachability game", epilog="Warning: if the number of vertices and/or the bounds for the out-degree range and/or the probability to create a cycle are too low, the program may not stop")
    parser.add_argument("size", type=int, help="Size of the graph")
    parser.add_argument("lowOutgoing", type=int, help="Lower bound for the out-degree range. If it's possible to have 0 successors for a node v, an edge from v to v is added (because we need infinite pathes)")
    parser.add_argument("upOutgoing", type=int, help="Upper bound for the out-degree range. It must be <= size")
    parser.add_argument("probaCycle", type=float, help="If a cycle can be created, gives the probability the cycle is effectively created. Since we need to be able to create infinite pathes, the cycles of length 1 are always created. 0 is not recommended")
    parser.add_argument("maxWeight", type=int, help="Maximum absolute weight")
    parser.add_argument("nPlayers", type=int, help="The number of players")
    parser.add_argument("--probaPlayers", type=float, nargs="*", help="For each player, the probability a node belongs to the player. The sum of the values must be 1. By default, every player has the same probability (1/nPlayers)")
    parser.add_argument("--probaTarget", type=float, nargs="*", help="For each player, the probability a node is a target for the player. The sum of the values doesn't have to be 1. By default, it's 0.1 for each player")
    parser.add_argument("--maximumTarget", type=int, nargs="*", help="For each player, the maximum number of targets the player can have. By default, there are no limits")
    parser.add_argument("-s", "--share-targets", action="store_true", help="If set, multiple players can share a same target. If not set, the target sets are disjoint")
    parser.add_argument("-n", "--negative-weights", action="store_true", help="If set, negative weights are allowed (probability: 0.5)")
    parser.add_argument("-t", "--tuple-weights", action="store_true", help="If set, weights are tuples")
    parser.add_argument("--file", type=argparse.FileType("w"), nargs="?", default=sys.stdout, help="The output DOT file")

    args = parser.parse_args()

    game = random_game(args.size, args.lowOutgoing, args.upOutgoing, args.probaCycle, args.negative_weights, args.maxWeight, args.tuple_weights, args.share_targets, args.nPlayers, args.probaPlayers, args.probaTarget, args.maximumTarget)

    random_graph_to_dot(game, args.file)
