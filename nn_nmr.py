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
from tensorflow import keras
import tensorflow as tf
from tensorflow_addons.callbacks import TQDMProgressBar
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


class NMRModel():
    def __init__(self, X=None, y=None, model=None):
        #self.vector_dim = vector_dim
        #self.hasmodel = True if model else False
        self.X = X
        self.y = y
        self._model = deepcopy(model)
        self.model = self.build_model() if (X is not None) or (model is not None) else None
        self.dataset = None
    
    def load_model(self, filename):
        self.model = keras.models.load_model(filename)
        return self.model
    
    def load_weights(self, filename):
        self.model.load_weights(filename)
        return self.model
    
    def build_model(self):
        if self._model:
            model = self._model
        else:
            lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
                          0.0001,
                          decay_steps=100000,
                          decay_rate=5,
                          staircase=False)
            model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=[int(self.X[0].shape[0])], kernel_regularizer=keras.regularizers.l2(0.01)), 
            keras.layers.Dropout(0.1),
            keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1)
            ])
            optimizer = keras.optimizers.RMSprop(lr_schedule)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model
    
    def train_test_split(self, test_size=0.2, train_size=None, random_state=None, shuffle=True, stratify=None):
        self.dataset = train_test_split(self.X, self.y, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        return self.dataset
    
    def kfolder_split(self, n_split=5, shuffle=True, random_state=None, foldername=None, is_serialized=False):
        kf = KFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
        self.datasets = []
        train_test_index = []
        for train_index, test_index in kf.split(self.X):
            train_test_index.append((train_index, test_index))
            train_X, test_X = self.X[train_index], self.X[test_index]
            train_y, test_y = self.y[train_index], self.y[test_index]
            self.datasets.append((train_X, test_X, train_y, test_y))
        if foldername:
            os.makedirs(foldername, exist_ok=True)
            {f'{i}.dat':{'train_index': train_test_i[0], 'test_index': train_test_i[1]} for i, train_test_i in enumerate(train_test_index)}
            if is_serialized:
                for i, dataset in enumerate(self.datasets):
                    joblib.dump(dataset, f'{foldername}/{i}.dat')
        self.dataset = self.datasets[0]
        return self.datasets
    
    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE, callback_class=TQDMProgressBar(show_epoch_progress=False), filename=None, save_weights=False):
        """
        Training NN-NMR Model
        
        Parameters
        ----------
        epochs: int
        batch_size: int
        callback_class: tqdm
        filename: str, path of model, if set, model and history will be saved
        save_weights: bool, if True, model will be saved as weights file; if False, model will be saved as model folder

        Returns
        -------
        df_select: DataFrame
        """
        # early stop, 防止过拟合，patience 值用来检查改进 epochs 的数量
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        history = self.model.fit(self.dataset[0], self.dataset[2], validation_data=(self.dataset[1], self.dataset[3]), epochs=epochs, verbose=0, callbacks=[early_stop, callback_class], batch_size=batch_size)
        if filename:
            if save_weights:
                self.model.save_weights(filename)
            else:
                self.model.save(filename)
            pd.DataFrame(history.history).to_pickle(filename + '_hist.pkl')
        return history
    
    def train_model(self, test_size=0.2, train_size=None, random_state=None, shuffle=True, stratify=None, epochs=EPOCHS, batch_size=BATCH_SIZE, callback_class=TQDMProgressBar(show_epoch_progress=False), filename=None, save_weights=False, show_ax=False):
        self.train_test_split(test_size, train_size, random_state, shuffle, stratify)
        history = self.train(epochs, batch_size, callback_class, filename, save_weights)
        foldername = os.path.dirname(filename)
        rmse_testdata = self.get_rmse()
        np.save(f'{foldername}/rmse_and_test_data.npy', rmse_testdata)
        if show_ax:
            ax = plot_dft_validation(dft=rmse_testdata['test_y'], dnns=rmse_testdata['test_y_pred'], rmse=rmse_testdata['rmse'])
            return history, ax
        else:
            return history

    def get_rmse(self):
        test_pred = self.model.predict(self.dataset[1])
        rmse = mean_squared_error(self.dataset[3], test_pred, squared=False)
        rmse_d = {'rmse': rmse, 'test_y': self.dataset[3], 'test_y_pred': test_pred.flatten()}
        return rmse_d
    
    def train_kfolder_models(self, n_split=5, shuffle=True, random_state=None, epochs=EPOCHS, batch_size=BATCH_SIZE, callback_class=TQDMProgressBar(show_epoch_progress=False), foldername=None):
        if not foldername:
            raise ValueError('Please setup foldername to store output files')
        os.makedirs(foldername, exist_ok=True)
        datasets = self.kfolder_split(n_split, shuffle, random_state, foldername)
        hists = []
        rmse_testdatas = []
        for i, dataset in enumerate(tqdm(datasets, desc='models')):
            self.model = self.build_model()
            history = self.train_model(filename=f'{foldername}/model_{i}')
            hists.append(history)
            rmse_testdatas.append(self.get_rmse())
        np.save(f'{foldername}/rmses_and_test_data.npy', rmse_testdatas)
        return hists
    
    # TODO: now load index
    @staticmethod
    def load_dataset(filename):
        train_X, test_X, train_y, test_y = joblib.load(filename)
        return (train_X, test_X, train_y, test_y)
    
    @staticmethod
    def load_testset(filename):
        test_X, test_y = joblib.load(filename)
        return (test_X, test_y)

    @staticmethod
    def get_X_from_df(df):
        X = np.array(df['X'].to_list())
        return X
    
    @staticmethod
    def get_y_from_df(df):
        y = df['y'].values
        return y
    
    @classmethod
    def get_Xy_from_df(cls, df):
        X, y = cls.get_X_from_df(df), cls.get_y_from_df(df)
        return X, y
    
    def predict_fcshifts(self, atoms, expression, positions=None, elements=None):
        soaps = get_soaps_from_atoms(atoms, expression, positions, elements)
        fcshifts = self.model.predict(soaps)
        return fcshifts.flatten()
    
    def predict_fcshifts_from_traj(self, traj, expression, positions=None, elements=None):
        fcshifts = [self.predict_fcshifts(atoms, expression, positions, elements) for atoms in traj]
        return np.concatenate(fcshifts)
