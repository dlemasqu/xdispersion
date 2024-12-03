# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram


"""
Core classes are defined below
"""
# @nb.jit(nopython=True, cache=False)
def geodist(lon1, lon2, lat1, lat2):
    """Calculate great-circle distance on a unit sphere.
    
    Parameters
    ----------
    lon1: float or numpy.ndarray
        longitude of particle 1 (in radian).
    lon2: float or numpy.ndarray
        longitude of particle 2 (in radian).
    lat1: float or numpy.ndarray
        latitude of particle 1 (in radian).
    lat2: float or numpy.ndarray
        latitude of particle 2 (in radian).
    
    Returns
    -------
    dis: float or numpy.ndarray
        Great-circle distance.
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2.0 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2.0
    
    dis = 2.0 * np.arcsin(np.sqrt(a))
    
    return dis


def semilog_fit(t, y):
    """semi-log fit of a timeseries y=f(t)
    
    Parameters
    ----------
    t: numpy.ndarray
        Relative time axis.
    y: numpy.ndarray
        Values at each time.
    
    Returns
    -------
    slope: numpy.ndarray
        Slope of the semi-log fit.
    inter: numpy.ndarray
        Intersection of the fit.
    rmse: numpy.ndarray
        Root mean squared error of the fit.
    """
    idx = ~np.isnan(y)

    # select non-nan points to fit
    yy = y[idx]
    tt = t[idx]
    
    if len(yy) > 1:
        try:
            if tt[0] == 0:
                tt[0] = 1e-20
            slope, inter = np.polyfit(np.log(tt), yy, 1)
        except:
            print('polyfit error')
            return np.nan, np.nan, np.nan
        fitted = slope * tt + inter;
        rmse = np.sqrt(np.sum((fitted - yy) ** 2.0) / len(tt));
        
        return slope, inter, rmse
    else:
        return np.nan, np.nan, np.nan


def hist_mean(var, obins, nbins):
    """Re-bin a given variable into a new set of bins through xhistogram

    Since this is a histogram average, the new coordinate (nbins) is
    generally coarser than the original one (obins).
    
    Parameters
    ----------
    var: xr.DataArray
        A given variable
    obins: xr.DataArray
        original bins or coordinate
    nbins: xr.DataArray or numpy.array
        new bins or coordinate
    """
    ones = var - var + 1

    nbv = nbins.values if type(nbins) is xr.DataArray else nbins
    
    mean = histogram(obins.rename('rtmp'), bins=nbv, weights=var.rename('vtmp') , block_size=1) \
         / histogram(obins.rename('rtmp'), bins=nbv, weights=ones.rename('otmp'), block_size=1)

    mean['rtmp_bin'] = nbins[:-1]
    
    return mean.rename({'rtmp_bin':'sep'})


def get_overlap_indices(times1, times2):
    """get overlapping indices.
    
    Parameters
    ----------
    times1: numpy.array
        times of trajectory 1.
    times2: numpy.array
        times of trajectory 2.
    
    Returns
    -------
    i1, i2, j1, j2: int
        i1 and i2 are start and end indices for trajectory 1;
        j1 and j2 are start and end indices for trajectory 2
        one can slice using timesi[i1:i2], so the i2 is exclusive!!!
    """
    # cases for non-overlap
    if times2[0] > times1[-1]:
        return None, None, None, None

    # cases for non-overlap
    if times2[-1] < times1[0]:
        return None, None, None, None

    lts, sts = times1, times2

    switch = False
    if len(times1) < len(times2):
        switch = True
        lts = times2 # longer
        sts = times1 # shorter
    
    strIdx = np.where(lts == sts[ 0])[0]
    endIdx = np.where(lts == sts[-1])[0]

    hasStr = False
    hasEnd = False

    if len(strIdx) != 0:
        hasStr = True
        strIdx = strIdx[0]

    if len(endIdx) != 0:
        hasEnd = True
        endIdx = endIdx[0]

    if hasStr and hasEnd:
        if switch:
            i1, i2, j1, j2 = 0, len(sts), strIdx, endIdx+1
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
        else:
            i1, i2, j1, j2 = strIdx, endIdx+1, 0, len(sts)
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
    
    if (not hasStr) and hasEnd:
        strIdx = np.where(sts == lts[ 0])[0][0]
        if switch:
            i1, i2, j1, j2 = strIdx, len(sts), 0, endIdx+1
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
        else:
            i1, i2, j1, j2 = 0, endIdx+1, strIdx, len(sts)
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2

    if hasStr and (not hasEnd):
        endIdx = np.where(sts == lts[-1])[0][0]
        if switch:
            i1, i2, j1, j2 = 0, endIdx+1, strIdx, len(lts)
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
        else:
            i1, i2, j1, j2 = strIdx, len(lts), 0, endIdx+1
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2

    raise Exception(f'should not reach here: {hasStr}, {hasEnd}, {switch}')


"""
Helper (private) methods are defined below
"""
def _check_indices(i1, i2, j1, j2):
    if i2 - i1 != j2 - j1:
        raise Exception('invalid indices')

