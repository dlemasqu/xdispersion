# -*- coding: utf-8 -*-
"""
Created on 2025.02.26

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
import itertools
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple, Literal
from .utils import semilog_fit, geodist, get_overlap_indices


"""
Core classes are defined below
"""

class RelativeDispersion(object):
    """
    This class is designed for performing relative dispersion analysis.
    """
    def __init__(self,
        ds_traj: xr.Dataset,
        xpos: str,
        ypos: str,
        uvel: str,
        vvel: str,
        time: str,
        coord: Literal['cartesian', 'latlon'],
        ID: str,
        maxtlen: Optional[int] = -1 ,
        Rearth: Optional[float] = 6371.2,
        ragged: Optional[bool] = False
    ) -> None:
        """
        Construct a RelativeDispersion class

        Parameters
        ----------
        ds_traj: xarray.Dataset
            A ragged trajectory dataset.
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
        ID: str
            Dimension name for particle IDs.
        maxtlen: int
            Set the maximum length of rtime (relative time).
        Rearth: float
            The radius of the earth, either in m or in km, which determine
            the units of later statistics if coord is latlon.
        ragged: boolean
            Whether the dataset is a ragged one.  Default is False so that
            each drifter is of the same length.
        """
        self.xpos    = xpos
        self.ypos    = ypos
        self.uvel    = uvel
        self.vvel    = vvel
        self.time    = time
        self.coord   = coord
        self.ID      = ID
        self.ragged  = ragged
        self.Rearth  = Rearth
        self.deg2m   = np.deg2rad(1.0) * Rearth
        self.ds_traj = ds_traj
        self.dtype   = ds_traj[uvel].dtype
        self.maxtlen = maxtlen

        times = ds_traj[time]
        if np.issubdtype(times.dtype, np.datetime64):
            # change unit to day
            self.dt = (times[1] - times[0]).astype('int').values / 1e9 / 86400
        else:
            self.dt = (times[1] - times[0])

        if ragged and maxtlen < 0:
            raise Exception('maxtlen should be positive when trajectories are a ragged dataset')

        if not ragged and maxtlen < 0:
            maxtlen = len(times)
        
        if coord not in ['cartesian', 'latlon']:
            raise Exception(f'invalid coord {coord}, should be [cartesian, latlon]')
    
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    "       Below are particle-related functions.        "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_all_pairs(self) -> xr.Dataset:
        """Get all available pairs into a Dataset
        
        This extracts pair information from the ragged trajectory dataset.
    
        Parameters
        ----------
        dset: xarray.DataArray
            A ragged dataset of trajectory, usually generated from clouddrift.
    
        Returns
        -------
        pinfo: xarray.Dataset
            Pair information as a xarray.Dataset.
        """
        dset = self.ds_traj
        
        if self.ragged: # for ragged drifter dataset
            pID , tlen, stim, r0, xpos0, ypos0,\
            idxI, idxJ = self._get_all(dset[self.ID].values,
                                       dset['rowsize'].values,
                                       dset[self.xpos].values,
                                       dset[self.ypos].values,
                                       dset[self.time].values)
            
            # pair index, starts from 0 to the total number of pairs
            pair = np.arange(len(r0)  , dtype=np.int32)
            # particle index in a single pair
            particle = np.array([0, 1], dtype=np.int32)
        
            tlen = xr.DataArray(tlen, name='tlen', dims='pair', coords={'pair':pair})
            r0   = xr.DataArray(r0  , name='r0'  , dims='pair', coords={'pair':pair})
            stim = xr.DataArray(stim, name='stim', dims='pair', coords={'pair':pair})
            
            pID   = xr.DataArray(pID , name='pID', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            xpos0 = xr.DataArray(xpos0, name='xpos0', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            ypos0 = xr.DataArray(ypos0, name='ypos0', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            idxI  = xr.DataArray(idxI, name='idxI', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            idxJ  = xr.DataArray(idxJ, name='idxJ', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            
            print(f'there are {len(r0)} pairs of particles')
            
            return xr.merge([tlen, stim, r0, pID, xpos0, ypos0, idxI, idxJ])
            
        else:
            no_pair = len(dset[self.ID])
            pair_idx = xr.DataArray(np.array(list(itertools.combinations(range(no_pair), 2))),
                                    dims=("pair", "particle"), name='pID',
                                    coords={'pair':np.arange(no_pair*(no_pair-1)/2, dtype='int32'),
                                            'particle':np.array([0,1], dtype='int32')})
            
            xpos0 = dset[self.xpos].isel({'time':0, 'ID':pair_idx}).drop_vars(['time','ID']).rename('xpos0')
            ypos0 = dset[self.ypos].isel({'time':0, 'ID':pair_idx}).drop_vars(['time','ID']).rename('ypos0')
            pID   = dset[self.ID].isel({'ID':pair_idx}).drop_vars('ID').rename('pID')
            tlen  = (xpos0 - xpos0).isel(particle=0).rename('tlen') + len(dset.time)
            stime = (dset.time.isel({'time':0}, drop=True) + (xpos0 - xpos0).isel({'particle':0})).rename('stime')
            r0    = np.hypot(xpos0.isel(particle=0) - xpos0.isel(particle=1),
                             ypos0.isel(particle=0) - ypos0.isel(particle=1)).rename('r0')
            
            return xr.merge([tlen, stime, r0, pID, xpos0, ypos0])
    
    
    def get_original_pairs(self,
        pairs: xr.Dataset,
        r0: List[float]
    ) -> xr.Dataset:
        """Get original pairs from a given pairs Dataset

        Original pairs are identified when initial separations are
        within the given range of r0 = [rmin, rmax].
        
        Parameters
        ----------
        pairs: xarray.Dataset
            Information of a given pairs.
        r0: float or list of float
            Range of initial separation for selecting original pairs.
        
        Returns
        -------
        pair_c: xarray.Dataset
            Information of chance pairs.
        """
        if isinstance(r0, float):
            if r0 <= 0:
                raise Exception('r0 should be larger than 0')
            r0 = [0, r0]
        
        if isinstance(r0, list):
            rmin, rmax = r0
        else:
            raise Exception(f'unsupported r0 {r0}, should be a list ' +
                            f'of two floats or a single float')
            
        cond = np.logical_and(pairs.r0>=rmin, pairs.r0<rmax)
        
        return pairs.where(cond, drop=True).astype(pairs.dtypes)
    
    
    def get_chance_pairs(self,
        pairs: xr.Dataset,
        r0: List[float]
    ) -> xr.Dataset:
        """Get chance pairs from a given pairs Dataset

        Chance pairs are identified when the separations are within
        the given range of r0 = [rmin, rmax] for the first time.
        
        Parameters
        ----------
        pairs: xarray.Dataset
            Information of a given pairs.
        r0: float or list of float
            Range of initial separation for selecting original pairs.
        
        Returns
        -------
        pair_c: xarray.Dataset
            Information of chance pairs.
        """
        if isinstance(r0, float):
            if r0 <= 0:
                raise Exception('r0 should be larger than 0')
            r0 = [0, r0]
        
        if isinstance(r0, list):
            rmin, rmax = r0
        else:
            raise Exception(f'unsupported r0 {r0}, should be a list ' +
                            f'of two floats or a single float')
        
        ds = pairs.copy(deep=True) # make a copy to modify
        idxI = pairs.idxI.values
        idxJ = pairs.idxJ.values
        
        xpos = self.ds_traj[self.xpos].values
        ypos = self.ds_traj[self.ypos].values
        
        for i in range(len(pairs['pair'])):
            idxIS, idxIE = idxI[i] # the first particle
            idxJS, idxJE = idxJ[i] # the second particle

            if self.coord == 'latlon':
                xi = np.deg2rad(xpos[idxIS:idxIE])
                xj = np.deg2rad(xpos[idxJS:idxJE])
                yi = np.deg2rad(ypos[idxIS:idxIE])
                yj = np.deg2rad(ypos[idxJS:idxJE])
    
                r = geodist(xi, xj, yi, yj) * self.Rearth
            else:
                r = np.hypot(xpos[idxIS:idxIE] - xpos[idxJS:idxJE],
                             ypos[idxIS:idxIE] - ypos[idxJS:idxJE])
            
            idx = r.argmin() # relative index of (the first) minimum separation
            rm  = r.min()    # minimum separation

            # idx == 0 means an original pair
            if idx > 0 and (rmin <= rm < rmax):
            #if (rmin <= rm <= rmax):
                ds.tlen[i] = ds.tlen[i] - idx
                ds.stim[i] = self.ds_traj[self.time][idxIS + idx]
                ds.r0[i]   = rm
                ds.xpos0[i,0] = xpos[idxIS + idx]
                ds.xpos0[i,1] = xpos[idxJS + idx]
                ds.ypos0[i,0] = ypos[idxIS + idx]
                ds.ypos0[i,1] = ypos[idxJS + idx]
                ds.idxI[i, 0] = idxIS + idx # only change start index
                ds.idxJ[i, 0] = idxJS + idx # only change start index
            else:
                ds.r0[i] = np.nan # assign nan so that we could drop them
        
        return ds.dropna(dim='pair').astype(pairs.dtypes)
    
    def get_variable(self,
        pairs: xr.Dataset,
        vname: str
    ) -> xr.DataArray:
        """get a variable from the ragged trajectory dataset
        
        The variable should be one of (xpos, ypos, uvel, vvel).  Its dimension
        should be v[pair, particle, rtime], where:
        - 'pair'     is for the dimension of different pairs;
        - 'particle' is for two particles [0, 1] in a single pair, and
        - 'rtime'    is for relative time starting when the pair is identified.
        
        Parameters
        ----------
        pairs: xarray.Dataset
            Information of a given pairs.
        vname: str
            Variable name in (xpos, ypos, uvel, vvel) for the ragged dataset.
        
        Returns
        -------
        re: xarray.DataArray
            A variable filled with data from ragged dataset.
        """
        if self.ragged:
            maxtlen = self.maxtlen
            
            v = self.ds_traj[vname]
            N = len(pairs['pair'])
            
            idxI = pairs.idxI.values
            idxJ = pairs.idxJ.values
            
            re = np.zeros((N, 2, maxtlen), dtype=self.dtype) + np.nan
            
            for i in range(N):
                idxIS, idxIE = idxI[i] # the first particle
                idxJS, idxJE = idxJ[i] # the second particle
                size = idxIE - idxIS
                
                if size <= maxtlen:
                    re[i, 0, :size] = v[idxIS:idxIE]
                    re[i, 1, :size] = v[idxJS:idxJE]
                else:
                    re[i, 0, :maxtlen] = v[idxIS:idxIS + maxtlen]
                    re[i, 1, :maxtlen] = v[idxJS:idxJS + maxtlen]
            
            return xr.DataArray(re, dims=['pair', 'particle', 'rtime'],
                                coords={'pair':pairs.pair, 'particle':[0,1],
                                        'rtime':np.arange(maxtlen)*self.dt})
        else:
            return self.ds_traj[vname].sel({self.ID:pairs.pID})\
                   .drop_vars('ID').rename({'time':'rtime'})
    

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    "          Below are the helper functions.           "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    def separation_measures(self,
        pairs: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray]:
        """Calculate separation measures from a set of pairs
    
        Parameters
        ----------
        pairs: xarray.Dataset
            Information of a given pairs.
    
        Returns
        -------
        rx: xarray.DataArray
            Zonal component of separation.
        ry: xarray.DataArray
            meridional component of separation.
        rxy: xarray.DataArray
            Cross component of separation.
        r: xarray.DataArray
            total separation.
        rpb: xarray.DataArray
            perturbation separation.
        """
        xpos = self.get_variable(pairs, self.xpos)
        ypos = self.get_variable(pairs, self.ypos)
        
        xi = xpos.isel(particle=0)
        xj = xpos.isel(particle=1)
        yi = ypos.isel(particle=0)
        yj = ypos.isel(particle=1)
        
        if self.coord == 'latlon':
            xi = np.deg2rad(xi)
            yi = np.deg2rad(yi)
            xj = np.deg2rad(xj)
            yj = np.deg2rad(yj)
            
            rx  = (xi - xj) * np.cos((yi + yj)/2.0) * self.Rearth
            ry  = (yi - yj) * self.Rearth
            rxy = rx * ry
            r   = geodist(xi, xj, yi, yj) * self.Rearth
            rxp = ((xi-xi.isel(rtime=0)) - (xj-xj.isel(rtime=0))) * np.cos((yi + yj)/2.0)
            ryp = ((yi-yi.isel(rtime=0)) - (yj-yj.isel(rtime=0)))
            rpb = np.hypot(rxp, ryp) * self.Rearth
        else:
            rx = xi - xj
            ry = yi - yj
            rxy = rx * ry
            r   = np.hypot(rx, ry)
            rpb = np.hypot((xi-xi.isel(rtime=0)) - (xj-xj.isel(rtime=0)),
                           (yi-yi.isel(rtime=0)) - (yj-yj.isel(rtime=0)))
        
        return rx, ry, rxy, r, rpb

    
    def velocity_measures(self,
        pairs: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray]:
        """Calculate separation measures from a set of pairs
        
        Parameters
        ----------
        pairs: xarray.Dataset
            Information of a given pairs.
    
        Returns
        -------
        du: xarray.DataArray
            delta u.
        dv: xarray.DataArray
            delta v.
        dul: xarray.DataArray
            longitudinal velocity.
        dut: xarray.DataArray
            transversal velocity.
        vsi: xarray.DataArray
            velocity magnitude of particle i.
        vsj: xarray.DataArray
            velocity magnitude of particle j.
        uv: xarray.DataArray
            inner product of two particle's velocities.
        """
        xpos = self.get_variable(pairs, self.xpos)
        ypos = self.get_variable(pairs, self.ypos)
        uvel = self.get_variable(pairs, self.uvel)
        vvel = self.get_variable(pairs, self.vvel)
        
        xi = xpos.isel(particle=0)
        xj = xpos.isel(particle=1)
        yi = ypos.isel(particle=0)
        yj = ypos.isel(particle=1)

        ui = uvel.isel(particle=0)
        uj = uvel.isel(particle=1)
        vi = vvel.isel(particle=0)
        vj = vvel.isel(particle=1)
        
        du  = ui - uj
        dv  = vi - vj
        vsi = np.hypot(ui, vi)
        vsj = np.hypot(uj, vj)
        uv  = ui * uj + vi * vj
        
        if self.coord == 'latlon':
            xi = np.deg2rad(xi)
            xj = np.deg2rad(xj)
            yi = np.deg2rad(yi)
            yj = np.deg2rad(yj)
            
            rx = (xi - xj) * np.cos((yi + yj)/2.0) * self.Rearth
            ry = (yi - yj) * self.Rearth
            r  = geodist(xi, xj, yi, yj) * self.Rearth
            
            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            
        else:
            rx = xi - xj
            ry = yi - yj
            r  = np.hypot(rx, ry)

            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            
        return du, dv, dul, dut, vsi, vsj, uv
    
    
    def r_based_measures(self,
        pairs: xr.Dataset,
        alpha: float,
        rbins: xr.DataArray,
        interpT: Optional[int] = 4
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """Calculate r-based measures using all available pairs
        
        r-based measures includes K2, S2, S2L, S2T, S3, FSLE, FAGR, FAGRp.
        No bootstrapping is done here, as the samples are quite large.
        
        Parameters
        ----------
        pairs: xarray.Dataset
            A given pairs.
        alpha: float
            ratio between neighbouring separation bin (for FSLE).
        rbins: xarray.DataArray
            A given separation bins.
        
        Returns
        -------
        K2: xarray.DataArray
            Relative diffusivity.
        S2: xarray.DataArray
            2nd-order velocity structure funciton.
        S2L: xarray.DataArray
            2nd-order longitudinal velocity structure funciton.
        S2T: xarray.DataArray
            2nd-order transversal velocity structure funciton.
        S3: xarray.DataArray
            3rd-order velocity structure funciton.
        FSLE: xarray.DataArray
            Finite-size Lyapunov exponent.
        FAGR: xarray.DataArray
            Finite-amplitude growth rate.
        FAGRp: xarray.DataArray
            Positive finite-amplitude growth rate (equivalent to FSLE).
        numS: xarray.DataArray
            Number of observations for K2, S2, S2L, S2T, S3, FAGR.
        numP: xarray.DataArray
            Number of observations for positive FAGR.
        numF: xarray.DataArray
            Number of observations for FSLE.
        """
        N = len(pairs['pair'])
        
        xpos = self.ds_traj[self.xpos]
        ypos = self.ds_traj[self.ypos]
        uvel = self.ds_traj[self.uvel]
        vvel = self.ds_traj[self.vvel]
        
        K2    = (rbins - rbins).rename('K2')
        S2    = (rbins - rbins).rename('S2')
        S2L   = (rbins - rbins).rename('S2L')
        S2T   = (rbins - rbins).rename('S2T')
        S3    = (rbins - rbins).rename('S3')
        FSLEO = (rbins - rbins).rename('FSLEO')
        FSLEI = (rbins - rbins).rename('FSLEI')
        FAGR  = (rbins - rbins).rename('FAGR')
        FAGRp = (rbins - rbins).rename('FAGRp')
        numS  = (rbins - rbins).rename('num_S2')
        numP  = (rbins - rbins).rename('num_FAGRp')
        numF  = (rbins - rbins).rename('num_FSLE')
        
        k2  =    K2.values[:-1]
        s2  =    S2.values[:-1]
        s2l =   S2L.values[:-1]
        s2t =   S2T.values[:-1]
        s3  =    S3.values[:-1]
        fslo= FSLEO.values[:-1]
        fsli= FSLEI.values[:-1]
        fag =  FAGR.values[:-1]
        fap = FAGRp.values[:-1]
        nvs =  numS.values[:-1]
        nvp =  numP.values[:-1]
        nvf =  numF.values[:-1]
        
        idxI = pairs.idxI.values
        idxJ = pairs.idxJ.values
        tlen = pairs.tlen.values

        Rearth = self.Rearth
        _histo = np.histogram
        rbinv  = rbins.values
        deltaT = self.dt
        dtype  = self.dtype
        
        for i in tqdm(range(N), ncols=80):
            size = tlen[i]
            
            #########   allocate variables   ########
            xx = np.zeros((2, size), dtype=dtype) + np.nan
            yy = np.zeros((2, size), dtype=dtype) + np.nan
            uu = np.zeros((2, size), dtype=dtype) + np.nan
            vv = np.zeros((2, size), dtype=dtype) + np.nan
            
            idxIS, idxIE = idxI[i] # the first particle
            idxJS, idxJE = idxJ[i] # the second particle
            
            #########      fill in data      ########
            xx[0, :] = xpos[idxIS:idxIE]
            xx[1, :] = xpos[idxJS:idxJE]
            yy[0, :] = ypos[idxIS:idxIE]
            yy[1, :] = ypos[idxJS:idxJE]
            uu[0, :] = uvel[idxIS:idxIE]
            uu[1, :] = uvel[idxJS:idxJE]
            vv[0, :] = vvel[idxIS:idxIE]
            vv[1, :] = vvel[idxJS:idxJE]
            
            #########   start calculations   ########
            if self.coord == 'latlon':
                xx  = np.deg2rad(xx)
                yy  = np.deg2rad(yy)
                
                rx  = (xx[0] - xx[1]) * np.cos((yy[0] + yy[1])/2.0) * Rearth
                ry  = (yy[0] - yy[1]) * Rearth
                r   = geodist(xx[0], xx[1], yy[0], yy[1]) * Rearth
                
                du  = uu[0] - uu[1]
                dv  = vv[0] - vv[1]
                dul = (rx * du + ry * dv) / r # longitudinal velocity
                dut = (rx * dv - ry * du) / r # transversal  velocity
                
            else:
                rx  = xx[0] - xx[1]
                ry  = yy[0] - yy[1]
                r   = np.hypot(rx, ry)

                du  = uu[0] - uu[1]
                dv  = vv[0] - vv[1]
                dul = (rx * du + ry * dv) / r # longitudinal velocity
                dut = (rx * dv - ry * du) / r # transversal  velocity
            
            ######### wrap r into DataArray #########
            r_or = xr.DataArray(r, dims='time',
                                coords={'time':np.arange(size) * deltaT})
            if interpT > 1:
                timeInt = np.linspace(0, r_or.time[-1], int((size-1)*interpT+1))
                r_da = r_or.interp(time=timeInt)
            else:
                r_da = r_or
            
            #########       for FSLE, FAGR     #########
            rd = r_or[r_or.argmin().values:]
            Td = xr.where(rd > rbins, 1, np.nan).idxmax('time')
            fsle = Td.diff('rbin')
            fsleO= (np.log(alpha) / fsle.where(fsle != 0))
            rd = r_da[r_da.argmin().values:]
            Td = xr.where(rd > rbins, 1, np.nan).idxmax('time')
            fsle = Td.diff('rbin')
            fsleI= (np.log(alpha) / fsle.where(fsle != 0))
            fagr = np.log(r_or).differentiate('time').values
            
            #########  accumulated within bins  #########
            tmp_K2 , _ = _histo(r, bins=rbinv, weights=(r_or**2).differentiate('time').values/2)
            tmp_S2 , _ = _histo(r, bins=rbinv, weights=du**2+dv**2)
            tmp_S2L, _ = _histo(r, bins=rbinv, weights=dul**2)
            tmp_S2T, _ = _histo(r, bins=rbinv, weights=dut**2)
            tmp_S3 , _ = _histo(r, bins=rbinv, weights=dul*(du**2+dv**2))
            tmp_FG , _ = _histo(r, bins=rbinv, weights=fagr)
            tmp_FGp, _ = _histo(r, bins=rbinv, weights=np.where(fagr>0, fagr, 0))
            tmp_noS, _ = _histo(r, bins=rbinv, weights=du-du+1)
            tmp_noP, _ = _histo(r, bins=rbinv, weights=np.where(fagr>0, 1, 0))
            tmp_noF    = np.where(np.isnan(fsleO), 0, 1)
            
            k2  += np.where(np.isnan(tmp_K2 ), 0, tmp_K2 )
            s2  += np.where(np.isnan(tmp_S2 ), 0, tmp_S2 )
            s2l += np.where(np.isnan(tmp_S2L), 0, tmp_S2L)
            s2t += np.where(np.isnan(tmp_S2T), 0, tmp_S2T)
            s3  += np.where(np.isnan(tmp_S3 ), 0, tmp_S3 )
            fslo+= np.where(np.isnan(fsleO  ), 0, fsleO  )
            fsli+= np.where(np.isnan(fsleI  ), 0, fsleI  )
            fag += np.where(np.isnan(tmp_FG ), 0, tmp_FG )
            fap += np.where(np.isnan(tmp_FGp), 0, tmp_FGp)
            nvs += tmp_noS
            nvp += tmp_noP
            nvf += tmp_noF
        
        K2    /= numS
        S2    /= numS
        S2L   /= numS
        S2T   /= numS
        S3    /= numS
        FAGR  /= numS
        FAGRp /= numP
        FSLEO /= numF
        FSLEI /= numF
        
        return K2, S2, S2L, S2T, S3, FSLEO, FSLEI, FAGR, FAGRp, numS, numP, numF

    
    def r_based_measures2(self,
        pairs: xr.Dataset,
        alpha: float,
        rbins: xr.DataArray,
        interpT: Optional[int] = 4
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """Calculate r-based measures using all available pairs
        
        r-based measures includes K2, S2, S2L, S2T, S3, FSLE, FAGR, FAGRp.
        No bootstrapping is done here, as the samples are quite large.
        
        Parameters
        ----------
        pairs: xarray.Dataset
            A given pairs.
        alpha: float
            ratio between neighbouring separation bin (for FSLE).
        rbins: xarray.DataArray
            A given separation bins.
        
        Returns
        -------
        K2: xarray.DataArray
            Relative diffusivity.
        S2: xarray.DataArray
            2nd-order velocity structure funciton.
        S2L: xarray.DataArray
            2nd-order longitudinal velocity structure funciton.
        S2T: xarray.DataArray
            2nd-order transversal velocity structure funciton.
        S3: xarray.DataArray
            3rd-order velocity structure funciton.
        FSLE: xarray.DataArray
            Finite-size Lyapunov exponent.
        FAGR: xarray.DataArray
            Finite-amplitude growth rate.
        FAGRp: xarray.DataArray
            Positive finite-amplitude growth rate (equivalent to FSLE).
        numS: xarray.DataArray
            Number of observations for K2, S2, S2L, S2T, S3, FAGR.
        numP: xarray.DataArray
            Number of observations for positive FAGR.
        numF: xarray.DataArray
            Number of observations for FSLE.
        """
        N = len(pairs['pair'])
        
        xpos = self.get_variable(pairs, self.xpos)
        ypos = self.get_variable(pairs, self.ypos)
        uvel = self.get_variable(pairs, self.uvel)
        vvel = self.get_variable(pairs, self.vvel)
        
        K2    = (rbins - rbins).rename('K2')
        S2    = (rbins - rbins).rename('S2')
        S2L   = (rbins - rbins).rename('S2L')
        S2T   = (rbins - rbins).rename('S2T')
        S3    = (rbins - rbins).rename('S3')
        FSLEO = (rbins - rbins).rename('FSLEO')
        FSLEI = (rbins - rbins).rename('FSLEI')
        FAGR  = (rbins - rbins).rename('FAGR')
        FAGRp = (rbins - rbins).rename('FAGRp')
        numS  = (rbins - rbins).rename('num_S2')
        numP  = (rbins - rbins).rename('num_FAGRp')
        numF  = (rbins - rbins).rename('num_FSLE')
        
        k2  =    K2.values[:-1]
        s2  =    S2.values[:-1]
        s2l =   S2L.values[:-1]
        s2t =   S2T.values[:-1]
        s3  =    S3.values[:-1]
        fslo= FSLEO.values[:-1]
        fsli= FSLEI.values[:-1]
        fag =  FAGR.values[:-1]
        fap = FAGRp.values[:-1]
        nvs =  numS.values[:-1]
        nvp =  numP.values[:-1]
        nvf =  numF.values[:-1]

        Rearth = self.Rearth
        _histo = np.histogram
        rbinv  = rbins.values
        deltaT = self.dt
        dtype  = self.dtype
        
        #########   start calculations   ########
        if self.coord == 'latlon':
            xx  = np.deg2rad(xpos)
            yy  = np.deg2rad(ypos)
            uu  = uvel
            vv  = vvel

            xx1 = xx.isel(particle=0)
            xx2 = xx.isel(particle=1)
            yy1 = yy.isel(particle=0)
            yy2 = yy.isel(particle=1)
            
            rx  = (xx1 - xx2) * np.cos((yy1 + yy2)/2.0) * Rearth
            ry  = (yy1 - yy2) * Rearth
            r   = geodist(xx1, xx2, yy1, yy2) * Rearth
            
            du  = uu.isel(particle=0) - uu.isel(particle=1)
            dv  = vv.isel(particle=0) - vv.isel(particle=1)
            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            
        else:
            xx  = xpos
            yy  = ypos
            uu  = uvel
            vv  = vvel
            
            rx  = xx.isel(particle=0) - xx.isel(particle=1)
            ry  = yy.isel(particle=0) - yy.isel(particle=1)
            r   = np.hypot(rx, ry)

            du  = uu.isel(particle=0) - uu.isel(particle=1)
            dv  = vv.isel(particle=0) - vv.isel(particle=1)
            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
        
        ######### interpolation for FSLE #########
        if interpT > 1:
            r_da = r.interp(rtime=np.linspace(0, r['rtime'][-1], int((len(r['rtime'])-1)*interpT+1)))
        else:
            r_da = r
        
        #########       for FSLE     #########
        for p in tqdm(range(N), ncols=80):
            rtmp = r_da.isel(pair=p)
            
            rd = rtmp[rtmp.argmin().values:]
            Td = xr.where(rd > rbins, 1, np.nan).idxmax('rtime')
            fsleO = Td.diff('rbin')
            fsleO = (np.log(alpha) / fsleO.where(fsleO != 0))
            fslo += np.where(np.isnan(fsleO), 0, fsleO)
            
            tmp_noF = xr.where(np.isnan(fsleO), 0, 1)
            nvf += tmp_noF
            
            rd = rtmp[rtmp.argmin().values:]
            Td = xr.where(rd > rbins, 1, np.nan).idxmax('rtime')
            fsleI = Td.diff('rbin')
            fsleI = (np.log(alpha) / fsleI.where(fsleI != 0))
            fsli += np.where(np.isnan(fsleI), 0, fsleI)
        
        fagr = np.log(r).differentiate('rtime').values
        
        #########  accumulated within bins  #########
        tmp_K2 , _ = _histo(r, bins=rbinv, weights=(r**2).differentiate('rtime').values/2)
        tmp_S2 , _ = _histo(r, bins=rbinv, weights=(du**2+dv**2).values)
        tmp_S2L, _ = _histo(r, bins=rbinv, weights=(dul**2).values)
        tmp_S2T, _ = _histo(r, bins=rbinv, weights=(dut**2).values)
        tmp_S3 , _ = _histo(r, bins=rbinv, weights=(dul*(du**2+dv**2)).values)
        tmp_FG , _ = _histo(r, bins=rbinv, weights=fagr)
        tmp_FGp, _ = _histo(r, bins=rbinv, weights=np.where(fagr>0, fagr, 0))
        tmp_noS, _ = _histo(r, bins=rbinv, weights=(du-du+1).values)
        tmp_noP, _ = _histo(r, bins=rbinv, weights=np.where(fagr>0, 1, 0))
        
        k2  += np.where(np.isnan(tmp_K2 ), 0, tmp_K2 )
        s2  += np.where(np.isnan(tmp_S2 ), 0, tmp_S2 )
        s2l += np.where(np.isnan(tmp_S2L), 0, tmp_S2L)
        s2t += np.where(np.isnan(tmp_S2T), 0, tmp_S2T)
        s3  += np.where(np.isnan(tmp_S3 ), 0, tmp_S3 )
        fag += np.where(np.isnan(tmp_FG ), 0, tmp_FG )
        fap += np.where(np.isnan(tmp_FGp), 0, tmp_FGp)
        nvs += tmp_noS
        nvp += tmp_noP
        
        K2    /= numS
        S2    /= numS
        S2L   /= numS
        S2T   /= numS
        S3    /= numS
        FAGR  /= numS
        FAGRp /= numP
        FSLEO /= numF
        FSLEI /= numF
        
        return K2, S2, S2L, S2T, S3, FSLEO, FSLEI, FAGR, FAGRp, numS, numP, numF
    
    
    def _get_all(self,
        ID: np.array,
        rowsize: np.array,
        xpos: np.array,
        ypos: np.array,
        times: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array,
               np.array, np.array, np.array, np.array]:
        ntraj = 0
        dtype = xpos.dtype
    
        for i in range(len(rowsize)):
            if rowsize[i] > 0:
                ntraj += 1
    
        if ntraj == len(ID):
            npair = ntraj * (ntraj - 1) // 2
        else:
            raise Exception(f'there are {len(ID)-ntraj} empty trajectories')
        
        r0   = [] # npair * 1, initial separation
        tlen = [] # npair * 1, record length
        stim = [] # npair * 1, initial time (datetime64 format)
        pID  = [] # npair * 2, two IDs for a pair
        xp   = [] # npair * 2, two x-component of initial positions
        yp   = [] # npair * 2, two y-component of initial positions
        idx1 = [] # npair * 2, [global_start, global_end]
        idx2 = [] # npair * 2, [global_start, global_end]
        
        idx = np.roll(rowsize.cumsum(), 1) # start index for each trajectory
        idx[0] = 0
        
        for i in range(len(ID)):
            for j in range(i+1, len(ID)):
                # global start indices for a pair
                idxI, idxJ = idx[i], idx[j]
                
                tsI = times[idxI:idxI+rowsize[i]]
                tsJ = times[idxJ:idxJ+rowsize[j]]
    
                # relative indices. End indices i2, j2 are exclusive
                # for slicing like [i1:i2]
                i1, i2, j1, j2 = get_overlap_indices(tsI, tsJ)
    
                if i1 != None:
                    pID.append([ID[i], ID[j]])
                    xp.append([xpos[idxI+i1], xpos[idxJ+j1]])
                    yp.append([ypos[idxI+i1], ypos[idxJ+j1]])
                    stim.append(times[idxI+i1])
                    tlen.append(i2-i1)

                    if self.coord == 'latlon':
                        xpos1, xpos2 = np.deg2rad(xp[-1])
                        ypos1, ypos2 = np.deg2rad(yp[-1])
                        
                        r0.append(geodist(xpos1, xpos2, ypos1, ypos2))
                    else:
                        r0.append(np.hypot(xpos1 - xpos2, ypos1 - ypos2))
                    
                    idx1.append([idxI+i1, idxI+i2]) # store global index
                    idx2.append([idxJ+j1, idxJ+j2]) # store global index

        if self.coord == 'latlon':
            tmp = self.Rearth
        else:
            tmp = 1
    
        return np.array(pID , dtype=np.int32), np.array(tlen, dtype=np.int32),\
               np.array(stim, dtype=times.dtype),\
               np.array(r0  , dtype=dtype) * tmp,\
               np.array(xp  , dtype=dtype)   , np.array(yp  , dtype=dtype   ),\
               np.array(idx1, dtype=np.int32), np.array(idx2, dtype=np.int32)
    
    
    def __repr__(self) -> str:
        """Print this class as a string"""
        if np.issubdtype(self.ds_traj[self.time].dtype, np.datetime64):
            suffix = ' (days)'
        else:
            suffix = ''
        
        return \
            f' RelativeDispersion class with:\n'\
            f'   xpos: {self.xpos} \n'\
            f'   ypos: {self.ypos} \n'\
            f'   uvel: {self.uvel} \n'\
            f'   vvel: {self.vvel} \n'\
            f'   time: {self.time} \n'\
            f'  coord: {self.coord}\n'\
            f'  delta: {self.dt:6.3f}{suffix}\n'\
            f'maxtlen: {self.maxtlen}\n'\



"""
Helper (private) methods are defined below
"""




