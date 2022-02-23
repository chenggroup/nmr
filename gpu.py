import numpy as np
from ase.io import read
from nn_nmr import NMRModel

traj_path = '' # Please set the directory where the long trajectory is
model_path = '' # Please set the directory where the model is, eg. ./results/model_test

traj = read(traj_path, ':')
expression = "soap cutoff=5.5 cutoff_transition_width=0.5 n_max=9 l_max=9 atom_sigma=0.55 n_Z=1 n_species=4 species_Z={11, 12, 25, 8}"
elements = ['Na']

nmr = NMRModel()
nmr.load_model(model_path)
fcshifts = nmr.predict_fcshifts_from_traj(traj[:], expression=expression, elements=elements)
np.save('fcshifts_pred.npy', fcshifts)
