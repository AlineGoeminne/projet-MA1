set terminal pngcairo size 1920,1080
set output "players.png"

set title "Temps d'exécution médian en fonction du nombre de joueurs"

set xtics nomirror
set logscale y

set xlabel "nombre de joueurs"
set ylabel "temps (échelle logarithmique)"

set style data linespoints

plot 'players-positive.data' using 2:xtic(1) title "Valeurs positives (moyennes)",\
    "" using 3:xtic(1) title "Valeurs positives (médiannes)"

# 4,00Ghz; limite de temps: 10 secondes
# size=20