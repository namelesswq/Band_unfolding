# Band_unfolding
Some python script for plot band unfolding picture

## For banduppy software
python plot_band_unfold.py bandstructure_unfolded.dat --mode density --bins-k 200 --bins-e 200 --cmap viridis --vmin 0 --vmax 0.08 --smooth-sigma 1.0  --ylim 8.558 18.558 --ef 13.558 --out dens.png

python plot_band_unfold.py bandstructure_unfolded.dat --mode fatband --color 'r' --size-min 10 --size-max 50 --alpha-min 0.01 --alpha-max 1.0 --ylim 8.558 18.558 --ef 13.558 --out fat.png
