set terminal pngcairo size 1920,1080
set output "taille.png"

set title "Temps d'exécution médian en fonction du nombre de nœuds"

set xtics nomirror
set logscale y

set xlabel "taille"
set ylabel "temps (échelle logarithmique)"

set style data linespoints

plot 'taille-positive.data' using 2:xtic(1) title "Valeurs positives (moyennes)",\
    "" using 3:xtic(1) title "Valeurs positives (médiannes)"

# 4,00Ghz; limite de temps: 10 secondes