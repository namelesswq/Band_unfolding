# Band_unfolding
Some python script for plot band unfolding picture

## For banduppy software
 python plot_band_unfold.py bandstructure_unfolded.dat --mode density --bins-k 200 --bins-e 200 --cmap viridis --vmin 0 --vmax 0.05 --smooth-sigma 1.0  --ylim -5 5 --ef 13.558 --ef-overlay 14.7758 --overlay-bands bands.out.gnu --overlay-color red --overlay-lw 1.0 --overlay-alpha 1.0 --out dens.png

python plot_band_unfold.py bandstructure_unfolded.dat --mode fatband --color 'r' --size-min 10 --size-max 50 --alpha-min 0.01 --alpha-max 1.0 --ylim -5 5 --ef 13.558 --overlay-bands bands.out.gnu --overlay-color green --overlay-lw 1.0 --overlay-alpha 1.0 --ef-overlay 14.7758 --out fat.png
