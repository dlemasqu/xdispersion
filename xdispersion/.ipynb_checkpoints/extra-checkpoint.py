# -*- coding: utf-8 -*-
"""
Created on 2024.11.21

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import xarray as xr
import numpy as np
import numba as nb
import xrft as xrft
from .utils import geodist

"""
codes below are still under test
"""

class ParticleStatistics(object):
    """
    This class is designed as a basis for particle statistical analysis.
    """
    def power_spectrum(self, ps, vnames, detrend='linear', window=True):
        """Calculate power spectrum of a variable
        
        Parameters
        ----------
        ps: list of xr.Dataset
            A list of particles
        vnames: str or list of str
            Variable names to be analyzed
        detrend: str
            How to de-trend
        window: bool
            Windowing the data or not
        
        Returns
        -------
        specs: xr.DataArray or list of xr.DataArray
            Spectrum of each variable.
        """
        def spectrum(ps, vname):
            # calculate a single variable
            spec = []
            time = self.time
            
            for p in tqdm(ps, ncols=80):
                var = p[vname]
                
                spec.append(xrft.power_spectrum(var, dim=time,
                                                detrend=detrend, window=window))
            return xr.concat(spec, 'particle')
        
        # if errbar:
        #     N = len(ps)
            
        #     err_lo = 2.0 * N / chi2.ppf(    0.05/2.0, df=N*2)
        #     err_hi = 2.0 * N / chi2.ppf(1.0-0.05/2.0, df=N*2)
        
        if isinstance(vnames, str):
            return spectrum(ps, vnames)
        elif isinstance(vnames, list):
            return [spectrum(ps, v) for v in vnames]
        else:
            raise Exception('invalid type for vnames, \
                             should be str or list of str')
    


class RelativeDispersion(ParticleStatistics):
    """
    This class is designed for performing two-particle statistical analysis.
    """
    def __init__(self, xpos, ypos, uvel, vvel, time, coord, Rearth=6371.2):
        """
        Construct a RelativeDispersion class

        Parameters
        ----------
        xpos: str
            x-position name e.g., lon or longitude
        ypos: str
            y-position name e.g., lat or latitude
        uvel: str
            x-velocity name e.g., u or uvel
        vvel: str
            y-velocity name e.g., v or vvel
        time: str
            Name of time dimension
        coord: str
            Type of coordinates in ['cartesian' 'latlon'].
        Rearth: float
            The radius of the earth, either in m or in km, which determine the
            units of later statistics if coord is latlon.
        """
        super().__init__(xpos, ypos, uvel, vvel, time, coord, Rearth)
    
    
    
    def pairs_to_particles(self, pairs, ID):
        """convert a list of pairs to a list of non-repeating particles.
        
        One may need an ID field in the particle dataset to do this (sometimes
        it is in the attributes of the dataset).

        Parameters
        ----------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        ID: str
            Field ID for identifying a unique particle
        
        Returns
        -------
        particles: list of particles
            All unique particles in a given pairs.
        """
        pts = [p for pair in pairs for p in pair]

        tmp = {}

        for p in pts:
            tmp[p.attrs[ID]] = p

        pts_unique = list(tmp.values())
        
        print(f'there are {len(pts_unique)} unique particles in given pairs')
        
        return pts_unique




"""
codes below are still under test
"""
def cal_relative_dispersion(pinfo, trajs, Rearth=6371.2, dtype=np.float32):
    times = np.linspace(0, 90, 4*24*90+1)

    npair, ntime = len(pinfo['idx']), len(times)

    rd = np.zeros([npair, ntime], dtype=dtype) * np.nan
    
    idxI = pinfo.idxI.astype(np.int32).values
    idxJ = pinfo.idxJ.astype(np.int32).values

    lons = trajs['longitude'].values
    lats = trajs['latitude' ].values

    for i in range(npair):
        sI = slice(idxI[i][0], idxI[i][1])
        sJ = slice(idxJ[i][0], idxJ[i][1])
        
        lon1 = np.deg2rad(lons[sI])
        lat1 = np.deg2rad(lats[sI])
    
        lon2 = np.deg2rad(lons[sJ])
        lat2 = np.deg2rad(lats[sJ])

        r = _geodist(lon1, lon2, lat1, lat2) * Rearth
        #print(f'{pinfo.gID.values[i]}, {lons[sI][0]:.3f}, {lats[sI][0]:.3f}')
        
        end = min(len(r), ntime)
        rd[i, :end] = r[:end]

    rd = xr.DataArray(rd, name='rd', dims=['pair', 'time'],
                      coords={'pair':pinfo['idx'].values, 'time':times})

    return rd


def get_all_pairs(dset, dtype=np.float32):
    """get all available pairs information
    
    This extracts pair information from the ragged trajectory dataset.

    Parameters
    ----------
    dset: xarray.DataArray
        A ragged dataset of trajectory, usually generated from clouddrift.
    dtype: np.dtype
        Precision of the float data.

    Returns
    -------
    pinfo: xarray.Dataset
        Pair information as a xarray.Dataset.
    """
    gID, tlen, r0, lon0, lat0, idx1, idx2 = _get_all_pairs(dset.ID.values, dset.rowsize.values,
                                                           dset.longitude.values, dset.latitude.values,
                                                           dset.time.values)
    idx  = np.arange(len(r0), dtype=np.int32)
    pair = np.array([0, 1]  , dtype=np.int32)

    tlen = xr.DataArray(tlen, name='tlen', dims='idx', coords={'idx':idx}).astype(np.int32)
    r0   = xr.DataArray(r0  , name='r0'  , dims='idx', coords={'idx':idx}).astype(dtype)
    
    gID  = xr.DataArray(gID , name='gID' , dims=['idx','pair'], coords={'idx':idx, 'pair':pair}).astype(dtype)
    lon0 = xr.DataArray(lon0, name='lon0', dims=['idx','pair'], coords={'idx':idx, 'pair':pair}).astype(dtype)
    lat0 = xr.DataArray(lat0, name='lat0', dims=['idx','pair'], coords={'idx':idx, 'pair':pair}).astype(dtype)
    idx1 = xr.DataArray(idx1, name='idxI', dims=['idx','pair'], coords={'idx':idx, 'pair':pair}).astype(np.int32)
    idx2 = xr.DataArray(idx2, name='idxJ', dims=['idx','pair'], coords={'idx':idx, 'pair':pair}).astype(np.int32)

    return xr.merge([tlen, r0, gID, lon0, lat0, idx1, idx2])



#@nb.jit(nopython=True, cache=False)
def _get_all_pairs(ID, rowsize, lons, lats, times, Rearth=6371.2, dtype=np.float32):
    ntraj = 0

    for i in range(len(rowsize)):
        if rowsize[i] > 0:
            ntraj += 1

    if ntraj == len(ID):
        npair = ntraj * (ntraj - 1) // 2
    else:
        raise Exception(f'there are {len(ID)-ntraj} empty trajectories')
    
    r0   = [] # npair * 1
    tlen = [] # npair * 1
    gID  = [] # npair * 2
    lon  = [] # npair * 2
    lat  = [] # npair * 2
    idx1 = [] # npair * 3, global_start, relative_start, relative_end
    idx2 = [] # npair * 3, global_start, relative_start, relative_end
    
    p = 0
    idx = np.roll(rowsize.cumsum(), 1) # calculate start index for each trajectory
    idx[0] = 0
    
    for i in range(len(ID)):
        for j in range(i+1, len(ID)):
            # global start indices for a pair
            idxI, idxJ = idx[i], idx[j]
            
            tsI = times[idxI:idxI+rowsize[i]]
            tsJ = times[idxJ:idxJ+rowsize[j]]

            # relative indices (end indices i2, j2 are exclusive, slice is [i1:i2])
            i1, i2, j1, j2 = get_overlap_indices(tsI, tsJ)

            if i1 != None:
                gID.append([ID[i], ID[j]])
                lon.append([lons[idxI+i1], lons[idxJ+j1]])
                lat.append([lats[idxI+i1], lats[idxJ+j1]])
                tlen.append(i2-i1)

                lon1, lon2 = np.deg2rad(lon[-1])
                lat1, lat2 = np.deg2rad(lat[-1])
                
                r0.append(geodist(lon1, lon2, lat1, lat2))
                idx1.append([idxI+i1, idxI+i2]) # store global index
                idx2.append([idxJ+j1, idxJ+j2]) # store global index
    
                p += 1

    return np.array(gID , dtype=np.int32), np.array(tlen, dtype=np.int32),\
           np.array(r0  , dtype=dtype) * Rearth,\
           np.array(lon , dtype=dtype )  , np.array(lat , dtype=dtype ),\
           np.array(idx1, dtype=dtype )  , np.array(idx2, dtype=dtype )


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




def _new_DataArray(pinfo, times, name, dtype=np.float32):
    npair = len(pinfo['idx'])
    ntime = len(times)
    
    data = np.zeros([npair, ntime], dtype=dtype)
    data = xr.DataArray(data, name=name, dims=['pair', 'time'], coords={'pair':pinfo['idx'], 'time':times})

    return data


def _check_indices(i1, i2, j1, j2):
    if i2 - i1 != j2 - j1:
        raise Exception('invalid indices')
