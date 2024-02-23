# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
from pathlib import Path
from itertools import product

import numpy as np
import xarray as xa

import matplotlib.pyplot as plt
import seaborn as sns

STANDARD_PARAMETER_SET = {
    'axes.axisbelow': False,
    'axes.labelsize': 23,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
    'axes.edgecolor': 'lightgrey',
    'axes.facecolor': 'None',

    'axes.grid.axis': 'y',
    'grid.color': 'lightgrey',
    'grid.alpha': 0.5,

    'axes.labelcolor': 'dimgrey',

    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,

    'figure.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',

    'xtick.bottom': True,
    'xtick.color': 'dimgrey',
    'xtick.direction': 'out',
    'xtick.top': False,
    'xtick.labelbottom': True,

    'ytick.major.width': 0.4,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'out',
    'ytick.left': False,
    'ytick.right': False
}


def anomalise(inarr):
    ntime = inarr.shape[0]

    outarr = np.zeros(inarr.shape)

    months = np.zeros((ntime))
    years = np.zeros((ntime))

    y = 1850
    m = 1
    for t in range(ntime):
        months[t] = m
        years[t] = y
        m = m + 1
        if m > 12:
            m = 1
            y = y + 1

    for m in range(1, 13):
        clim = (years >= 1991) & (years <= 2020) & (months == m)
        this_month = (months == m)
        selection = inarr[clim, :, :]
        selection = np.nanmean(selection, 0)
        outarr[this_month, :, :] = inarr[this_month, :, :] - selection[:, :]

    return outarr


def area_average(inarr, weights):
    ntime = inarr.shape[0]
    outarr = np.zeros((ntime))

    for t in range(ntime):
        selector = inarr[t, :, :]
        mask = ~np.isnan(selector)
        outarr[t] = np.sum(selector[mask] * weights[mask]) / np.sum(weights[mask])

    return outarr


def annualise(inarr):
    ntime = len(inarr)
    nyears = int(ntime / 12)
    outarr = np.zeros((nyears))

    for i in range(nyears):
        outarr[i] = np.mean(inarr[i * 12:(i + 1) * 12])

    return outarr


def reblend_datasets():
    data_dir = Path(os.getenv('DATADIR')) / 'ManagedData' / 'Data'

    noaa_dir = data_dir / 'NOAA Interim'
    hadley_dir = data_dir / 'HadCRUT5'
    hadisst_dir = data_dir / 'HadISST2'

    noaa = xa.open_dataset(noaa_dir / 'NOAAGlobalTemp_v5.1.0_gridded_s185001_e202312_c20240108T150239.nc')
    hadley = xa.open_dataset(hadley_dir / 'HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc')
    hadisst = xa.open_dataset(hadisst_dir / 'HadISST.2.2.0.0_sea_ice_concentration.nc')

    land_ice_mask = hadisst['sic'].data
    land_ice_mask[np.isnan(land_ice_mask)] = 1.0

    ntime = land_ice_mask.shape[0]

    land_ice_mask_5x5 = np.zeros((2088, 36, 72))

    for t, x, y in product(range(ntime), range(72), range(36)):
        land_ice_mask_5x5[t, y, x] = np.mean(land_ice_mask[t, y * 5:(y + 1) * 5, x * 5:(x + 1) * 5])

    for t in range(ntime, 2088):
        land_ice_mask_5x5[t, :, :] = land_ice_mask_5x5[t - 48, :, :]

    land_ice_mask_5x5 = np.flip(land_ice_mask_5x5, 1)

    hadley = hadley['tas_mean'].data
    hadley = np.roll(hadley, 36, 2)

    noaa = noaa['anom'].data[:, 0, :, :]

    hadley = anomalise(hadley)
    noaa = anomalise(noaa)

    weights = np.zeros((36, 72))
    for y in range(36):
        weights[y, :] = np.cos(np.deg2rad(y * 5.0 - 87.5))

    # NOAA land, Hadley ocean
    combo1 = noaa * land_ice_mask_5x5 + hadley * (1 - land_ice_mask_5x5)
    mask = (land_ice_mask_5x5 > 0) & (np.isnan(hadley))
    combo1[mask] = noaa[mask]

    # Hadley land, NOAA ocean
    combo2 = hadley * land_ice_mask_5x5 + noaa * (1 - land_ice_mask_5x5)
    mask = (land_ice_mask_5x5 == 0) & (np.isnan(hadley))
    combo2[mask] = noaa[mask]

    ts_had = annualise(area_average(hadley, weights))
    ts_noaa = annualise(area_average(noaa, weights))
    ts_1 = annualise(area_average(combo1, weights))
    ts_2 = annualise(area_average(combo2, weights))

    time = np.arange(1850, 2024, 1.0)

    sns.set(font='Franklin Gothic Book', rc=STANDARD_PARAMETER_SET)

    plt.figure(figsize=[16, 9])

    plt.plot(time, ts_had - np.mean(ts_had[0:51]), color='blue', label='HadCRUT5')
    plt.plot(time, ts_noaa - np.mean(ts_noaa[0:51]), color='orange', label='NOAAGlobalTemp v5.1')
    plt.plot(time, ts_1 - np.mean(ts_1[0:51]), color='pink', label='GHCN + HadSST')
    plt.plot(time, ts_2 - np.mean(ts_2[0:51]), color='purple', label='CRUTEM + ERSST')

    plt.legend(frameon=False, prop={'size': 20}, labelcolor='linecolor', handlelength=0, handletextpad=0.3)

    plt.savefig(Path(os.getenv('DATADIR')) / 'ManagedData' / 'Figures' / 'decoupling.png')

    plt.close()


if __name__ == '__main__':
    reblend_datasets()
