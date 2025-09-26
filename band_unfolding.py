import shutil
import numpy as np
import pickle
import sys
import banduppy

print(f'- BandUppy version: {banduppy.__version__}')

#-------------------------- Set job -------------------------------------
do_generate_SC_kpts = True
read_bandstructure = True
do_unfold = True
do_plot = True

SimulationParentFolder = '/data/home/wangqian/workplace/Zr2AC/Zr2SC_defect_Zr/band_unfolding'

#-------------------------- Define variables ---------------------------------
# supercell matrix
super_cell_size = [[2,0,0],[0,2,0],[0,0,1]]
# k-path: L-G-X-U,K-G. If the segmant is skipped, put a None between nodes.
PC_BZ_path = [[0,0,0],[0.5,0,0],[1/3, 1/3, 0],[0,0,0],[0,0,0.5],[0.5,0,0.5],[1/3,1/3,0.5],[0,0,0.5],None,[0.5,0,0.5],[0.5,0,0],None,[1/3,1/3,0.5],[1/3,1/3,0]]
# Number of k-points in each path segments. or, one single number if they are same.
npoints_per_path_seg = (50,50,50,50,50,50,50,50,50)
# Labels of special k-points: list or string. e.g ['L','G','X','U','K','G'] or 'LGXUKG'
special_k_points = "GMKGALHALMHK"
# Weights of the k-points to be appended in the final generated k-points files
kpts_weights = 1
# Save the SC kpoints in a file
save_to_file = True
# Directory to save file
save_to_dir = f'{SimulationParentFolder}/input'
# Wavefunction file path
sim_folder = f'{SimulationParentFolder}/tmp' # '<path where the vasp output files are>'
# File format of kpoints file that will be created and saved
kpts_file_format = 'qe' # This will generate qe KPOINTS file format
# Unfolding results directory
results_dir = f'{SimulationParentFolder}/results'
# QE file prefix
pw_file = 'zr2sc'

#---------------------- Initiate Unfolding method ----------------------------
if do_generate_SC_kpts:
    print (f"{'='*72}\n- Generating SC Kpoints...")
    band_unfold = banduppy.Unfolding(supercell=super_cell_size, 
                                      print_log='high')
    
    # ------------ Creating SC folded kpoints from PC band path -------------
    kpointsPBZ_full, kpointsPBZ_unique, kpointsSBZ, \
    SBZ_PBZ_kpts_mapping, special_kpoints_pos_labels \
    = band_unfold.generate_SC_Kpts_from_pc_k_path(pathPBZ = PC_BZ_path,
                                                  nk = npoints_per_path_seg,
                                                  labels = special_k_points,
                                                  kpts_weights = kpts_weights,
                                                  save_all_kpts = save_to_file,
                                                  save_sc_kpts = save_to_file,
                                                  save_dir = save_to_dir,
                                                  file_name_suffix = '',
                                                  file_format=kpts_file_format)
    print("- Generating SC Kpoints - done.")

#-------------------- Read wave function file ----------------------------
if read_bandstructure:
    print (f"{'='*72}\n- Reading band structure...")
    bands = banduppy.BandStructure(code = "espresso", spinor = False, prefix = f'{sim_folder}/{pw_file}')
    pickle.dump(bands, open(f"{results_dir}/bandstructure.pickle","wb"))
    print("- Reading band structure - done.")
else:
    print (f"{'='*72}\n- Unpickling band structure...")
    bands = pickle.load(open(f"{results_dir}/bandstructure.pickle","rb"))
    print("- Unpickling band structure - done.")

#----------------------- Unfold the band structures -------------------------
if do_unfold:
    print (f"{'='*72}\n- Unfolding band structure...")
    unfolded_bandstructure_, kpline \
    = band_unfold.Unfold(bands, kline_discontinuity_threshold = 0.1, 
                         save_unfolded_kpts = {'save2file': True, 
                                              'fdir': results_dir,
                                              'fname': 'kpoints_unfolded',
                                              'fname_suffix': ''},
                         save_unfolded_bandstr = {'save2file': True, 
                                                'fdir': results_dir,
                                                'fname': 'bandstructure_unfolded',
                                                'fname_suffix': ''})
    print ("- Unfolding - done")
else:
    print (f"{'='*72}\n- Reading band structure data from saved file...")
    unfolded_bandstructure_ = np.loadtxt(f'{results_dir}/bandstructure_unfolded.dat')
    kpline = np.loadtxt(f'{results_dir}/kpoints_unfolded.dat')[:,1]
    with open(f'{save_to_dir}/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
        special_kpoints_pos_labels = pickle.load(handle)
    print ("- Reading band structure file - done")

#----------------------- Plot the band structures ----------------------------
if do_plot:
    print (f"{'='*72}\n- Plotting band structure...")
    # Fermi energy
    Efermi = 13.5580
    # Minima in Energy axis to plot
    Emin = -5
    # Maxima in Energy axis to plot
    Emax = 5
    # Filename to save the figure. If None, figure will not be saved
    save_file_name = 'unfolded_bandstructure.png'

    plot_unfold = banduppy.Plotting(save_figure_dir=results_dir)
    
    fig, ax, CountFig \
    = plot_unfold.plot_ebs(kpath_in_angs=kpline,
                           unfolded_bandstructure=unfolded_bandstructure_,
                           save_file_name=save_file_name, CountFig=None,
                           Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5,
                           mode="density", special_kpoints=special_kpoints_pos_labels,
                           plotSC=True, fatfactor=20, nE=100,smear=0.2,
                           color='red', color_map='viridis', show_colorbar=False)
    
#%%
    fig, ax, CountFig \
    = band_unfold.plot_ebs(save_figure_dir=results_dir,
                           save_file_name='f_'+save_file_name, CountFig=None, 
                           Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                           mode="fatband", special_kpoints=special_kpoints_pos_labels, 
                           plotSC=True, fatfactor=20, nE=100,smear=0.2, 
                           color='red', color_map='viridis', show_colorbar=False)
    print ("- Plotting band structure - done")
