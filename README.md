# Objectif d'atteignabilité et équilibres de Nash dans les jeux sur grahe (Projet-MA1)

Projet réalisé dans le cadre de ma première année de master en mathématiques.

Le but était dans un premier temps d'expliquer des notions bien connues de la théorie des jeux: les équilibres de Nash
et ce dans le cadre d'un certain type de jeux: les jeux sur graphe avec objectif d'atteignabilité et quantitatif. 

Une fois un background clairement établi, je me suis interrogée à la notion d'équilibres pertinents et à un processus
algorithmique rapide afin de déterminer l'équilibre le plus pertinent possible sur un jeu donné.

Tout ceci a été abordé d'un point de vue théorique et est expliqué dans la partie rapport du projet.
Un prototype a ensuite été mis au point afin de tester les approches théoriques proposées. Un code en python se retrouve donc
dans la partie implémentation.

## Générateur aléatoire
Le fichier `RandomGame` permet de générer aléatoirement un jeu d'atteignabilité. La liste des paramètres peut être obtenue via 

    python RandomGame.py -h

Admettons qu'on veuille générer un jeu avec les paramètres suivants :
  - le graphe doit respecter :
    - 20 sommets
    - chaque sommet peut avoir 2 et 10 arcs en sortie
    - une boucle peut apparaître avec une probabilité de 0.5
    - la valeur absolue du poids maximal autorisé est 50
  - pour 2 joueurs
  - la probabilité qu'un noeud appartienne au joueur 1 est de 0.7 et au joueur 2 est de 0.3
  - la probabilité qu'un noeud soit une cible pour le joueur 1 est de 0.1 et pour le joueur 2 est de 0.4
  - le nombre maximal de cibles pour le joueur 1 est 5 et pour le joueur 2 est 8
  - on permet de partager une cible entre plusieurs joueurs

et stocker le fichier DOT résultant dans sortie.dot

alors, il faut écrire

    python RandomGame.py 20 2 10 0.5 50 2 -s --probaPlayers 0.7 0.3 --probaTarget 0.1 0.4 --maximumTarget 5 8 --dot sortie.dot