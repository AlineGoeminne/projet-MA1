set terminal pngcairo
set output "taille.png"

set title "Temps d'exécution médian en fonction du nombre de nœuds"

set xtics nomirror
set logscale y

set xlabel "taille"
set ylabel "temps (médianne; échelle logarithmique)"

set style data linespoints

plot 'taille-positive.data' using 2:xtic(1) title "Valeurs positives"