Main :
permet la génération de tests aléatoires. Par défaut, son exécution lance une série de 100 tests sur des jeux dont le graphe possède 5 noeuds. Les méthodes lancées sur ces jeux sont:
méthode aléatoire, best-first search avec fonction d’évaluation de type A*, best-first search initialisé et breadth-first search (ne retournant que le premier EN trouvé). Les résultats de ces tests sont stockés dans un fichier « stat_all_method.txt » et contient le nombre d’EN trouvés pour chaque méthode ainsi que le temps nécessaire.  S’y trouve aussi la comparaison du résultat de chaque méthode avec celui du best-first search avec A*. On comptabilise si ce dernier est meilleur, pire, égal, équivalent ou si aucune des deux méthodes ne trouve un EN. 
Une méthode de test uniquement sur A* mais sur des graphes dont le nombre de noeuds varie de 5 à 20 s’y retrouve également.


GraphGrame:
- définition de l'objet Graph
- définition de l'objet Vertex
- définition de l'objet ReachabilityGame : best-first search, breadth-first search, méthode random, ... (rem: la méthode de best-first search a été entendue à l'utilisation de tuple pour pondérer les arcs ainsi que la présence de poids négatifs. Les méthodes utilisées au sein du best-first search ont donc dû être également modifiées. ATTENTION, les autres méthodes n'ont pas été mises à jour pour une telle configuration de jeu!) + algorithme de backward induction (à utiliser si le graphe est sous la forme d'un arbre).

GraphToDotConverter : 
permet de convertir certains types de graphes de jeu en .dot afin de pouvoir les visualiser (utile pour le debug).


HousesGame :
contient tout ce qui permet la modélisation du jeu des maisons.

MinHeap:
implémentation de la structure de donnée de tas

Value:

permet de calculer les valeurs des jeux d'atteignabilité que ce soit à coûts positifs (dijkstraMinMax) ou avec des coûts négatifs (compute_value_with_negative_weight)


Dans le dossier Test:

Fichier TestMinHeap: regroupe quelques tests effectués sur la structure MinHeap
Fichier DijkstraMinMaxTest: regroupe quelques tests effectués sur l’algorithme DijkstraMinMax
Fichier ReachabilityGameTest: regroupe quelques tests sur les ReachabilityGame
Fichier MethodsTest: regoupe des tests sur l’exécution des différentes méthodes de recherche d’EN. Les résultats sont affichés. Par défaut, exécuter ce fichier génère un test aléatoire sur les quatre méthodes testées.

Fichier HousesGameTest: regroupe les quelques tests effectués sur le jeu des maisons. Les documents sources permettant de lancer ceux-ci se trouvent dans le dossier Maisons. Puisque peu de tests ont été effectués, il s'agit, à l'heure actuelle, de la seule manière de lancer les tests pour le jeu des maisons. 
Fichier ShortestPathTest: regroupe quelques tests pour le calcul des plus courts chemins en utilisant l'algorithme de Floyd-Warshall (avec et sans tuple sur les arcs)
Fichier ValueWithNegWeightTest: regroupe queqlues tests poru le calcul des valeurs dans des graphes pouvant avoir des arcs de poids négatifs.


Dans le dossier Maisons:

contient les fichiers .txt qui contiennent toutes les données nécessaires à la génération du jeu des maisons
Les fichiers sont formatés de la manière suivante:

- première ligne : nbr_intervalles production_energie cout_exterieur cout_interne
-chaque nouvelle maison est marqué par un "&"
-chaque ligne suivant le "&" est une nouvelle tâche avec : consommation_energie_de_la_tache contrainte_1 contrainte_2 contrainte_3 ...