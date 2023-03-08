from math import ceil
import numpy as np
from ase.io import read
from quippy import descriptors
from warnings import warn
import joblib
import pandas as pd
import os
import re
from copy import deepcopy
from itertools import compress
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


# Physical Constants
UB=9.274e-24
K=1.381e-23
GE=2.0023
# Epochs & Batch_size
EPOCHS=500
BATCH_SIZE=500


def get_HFFs(filename):
    """
    Parameters
    ----------
    filename: str, eg. 'OUTCAR'
    
    Returns
    -------
    HFFs: numpy.array
    """
    hff_reg = re.compile(r"Fermi contact \(isotropic\) hyperfine coupling parameter \(MHz\)\s+-+\s+[\w|\s]+-+\s+(.*?)---", re.DOTALL)
    with open(filename) as f:
        lines = f.read()
        hff_string = re.search(hff_reg, lines).group(1)
    
    HFFs = []
    for line in hff_string.strip().splitlines():
        HFFs.append(line.strip().split()[-1])
    return np.array(HFFs, dtype=float)


def HFFs2FCShifts(HFFs, gamma, mu_eff, cell_S, s, T, sigma):
    """
    Parameters
    ----------
    HFFs: numpy.array
    gamma: float
    mu_eff: float
    cell_S: float
    s: float
    T: float
    sigma: float
    
    Returns
    -------
    FCshifts: numpy.array
    """
    FCshifts=1000000*(HFFs*mu_eff**2*cell_S*UB)/(gamma*2*s*GE*3*K*(T-sigma))
    return FCshifts


# calculating the FC shifts
def get_FCShifts(filename, gamma=100, mu_eff=3.87, cell_S=48, s=1.5, T=320, sigma=-14.8):
    HFFs = get_HFFs(filename)
    FCshifts = HFFs2FCShifts(HFFs, gamma, mu_eff, cell_S, s, T, sigma)
    return FCshifts


# TODO: can be restructure, like read traj from folder/OUTCARs
# positions: iterable; elements: set
def get_soaps_from_traj(filename, expression, positions=None, elements=None):
    traj = read(filename, ':')
    desc = descriptors.Descriptor(expression)
    soaps_ = np.array(desc.calc_descriptor(traj))[:, :, :-1]
    if positions and elements:
        warn("positions and elements are simultaneously exist, elements will be ignored")
    elif positions:
        soaps = soaps_[:, positions, :]
    elif elements:
        if not isinstance(elements, list):
            raise TypeError("param elements should be set")
        else:
            cs = np.array(traj[0].get_chemical_symbols())
            soaps = np.concatenate([soaps_[:, np.where(cs == e)[0], :] for e in elements], axis=1)
    else:
        soaps = soaps_
    return soaps


def get_soaps_from_atoms(atoms, expression, positions=None, elements=None):
    desc = descriptors.Descriptor(expression)
    soaps_ = desc.calc_descriptor(atoms)[:, :-1]
    if positions and elements:
        warn("positions and elements are simultaneously exist, elements will be ignored")
    elif positions:
        soaps = soaps_[positions]
    elif elements:
        if not isinstance(elements, list):
            raise TypeError("param elements should be set")
        else:
            cs = np.array(atoms.get_chemical_symbols())
            soaps = np.concatenate([soaps_[np.where(cs == e)[0]] for e in elements])
    else:
        soaps = soaps_
    return soaps


def get_df_from_outcar(filename, expression, positions=None, elements=None, **kwargs):
    """
    Parameters
    ----------
    filename: str, eg. 'OUTCAR'
    
    Returns
    -------
    DataFrame, colunm X is soap descriptors, colunm y is fcshifts
    """
    atoms = read(filename)
    soaps = get_soaps_from_atoms(atoms, expression, positions, elements)
    fcshifts_ = get_FCShifts(filename, **kwargs)
    if positions and elements:
        warn("positions and elements are simultaneously exist, elements will be ignored")
    elif positions:
        fcshifts = fcshifts_[positions]
    elif elements:
        if not isinstance(elements, list):
            raise TypeError("param elements should be set")
        else:
            cs = np.array(atoms.get_chemical_symbols())
            fcshifts = np.concatenate([fcshifts_[np.where(cs == e)[0]] for e in elements])
    else:
        fcshifts = fcshifts_
    df = pd.DataFrame(data={'X': list(soaps), 'y': fcshifts})
    return df


def get_df_from_outcars(filenames, expression, positions=None, elements=None, outcar_filter=None, **kwargs):
    valid_filenames = get_valid_outcar(filenames, outcar_filter)
    df = pd.concat([get_df_from_outcar(filename, expression, positions, elements, **kwargs) for filename in valid_filenames], ignore_index=True)
    return df


def sample_from_df(df, n, filename=None):
    """
    random select n samples from df
    
    Parameters
    ----------
    df: DataFrame
    n: int
    
    Returns
    -------
    df_select: DataFrame
    """
    df_select = df.sample(n)
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_select.to_pickle(filename)
    return df_select


def filter_total_energy_change(filename, maxlimit=1e-5):
    with open(filename) as f:
        content = f.read()
    total_energy_change = [float(i.strip()) for i in re.findall(r'total energy-change.*:(.*)\(', content)]
    if total_energy_change[-1] <= maxlimit:
        return filename
    else:
        return None


def get_valid_outcar(outcars, outcar_filter=None):
    if outcar_filter:
        return compress(outcars, [outcar_filter(outcar) for outcar in outcars])
    else:
        return outcars


def plot_rmse_datasets(n_dataset, n_rmses, xlabel, ylabel, title):
    rmse_means = [np.mean(rmses) for rmses in n_rmses]
    rmse_stds = [np.std(rmses) for rmses in n_rmses]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    ax.plot(n_dataset,rmse_means,c='b',ls='--')
    ax.scatter(n_dataset, rmse_means, alpha=0.5,lw=1, s=20, c='b', marker='o', edgecolors='b')
    ax.errorbar(x=n_dataset, y=rmse_means, yerr=np.array(rmse_stds), fmt='none', ecolor='k', capsize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_dft_validation(dft, dnns, rmse=None, xlabel='DFT (ppm)', ylabel='DNNs (ppm)', title='shift deviation'):
    rmse_ = rmse or mean_squared_error(dft, dnns, squared=False)
    print(f'rmse: {rmse_} ppm')
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=150)
    ax.scatter(dft, dnns, alpha=0.1, s=10, c='r')
    ax.set_xlabel('DFT (ppm)')
    ax.set_ylabel('DNNs (ppm)')
    ax.set_title('shift deviation')
    line_min = int(min(ax.get_xlim()[0], ax.get_ylim()[0])/1000)*1000
    line_max = ceil(max(ax.get_xlim()[1], ax.get_ylim()[1])/1000)*1000
    ax.plot((line_min, line_max), (line_min, line_max), c='k', ls='--')
    return ax

