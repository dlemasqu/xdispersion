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
from utils import geodist

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

