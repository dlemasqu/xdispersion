# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
import numba as nb
from tqdm import tqdm
from xhistogram.xarray import histogram
from analytics import TwoParticleStatistics
from utils import semilog_fit, geodist, get_overlap_indices


"""
Core classes are defined below
"""

class RelativeDispersion(TwoParticleStatistics):
    """
    This class is designed for performing relative dispersion analysis.
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

    
    """"""""""""""""""""""""""""""""""""""""""
    " Below are particle-related functions.  "
    """"""""""""""""""""""""""""""""""""""""""
    
    def filter_positions(self, ps, Tc, Ts, type='lp', order=1):
        """Butterworth filtering of particle positions in-place
        
        Butterworth filtering is used with order and type specified.
        
        Parameters
        ----------
        ps: list of xr.Dataset or list of list
            A list of particles or a list of pairs
        Tc: float or list of two floats
            Cutoff time scales.  One float for high- or low-pass filtering,
            and two for band-pass filtering
        Ts: float
            Sampling time.  Unit should be the same as Tc.
        type: str
            Type of filtering, should be ['lp', 'hp', 'bp'] for respectively
            low-pass, high-pass, and band-pass filtering.
        order: int
            Order of the butterworth filter.
        """
        b, a = signal.butter(order, 1.0/(Tc), type, fs=1.0/Ts)
        
        if isinstance(ps[0], xr.Dataset):
            pairs = False
        elif isinstance(ps[0][0], xr.Dataset):
            pairs=True
        else:
            raise Exception('unsupported ps, should be a list of particles or pairs')
        
        xpos, ypos = self.xpos, self.ypos
        
        if pairs:
            for p in ps:
                p1, p2 = p

                filteredx = signal.filtfilt(b, a, p1[xpos])
                filteredy = signal.filtfilt(b, a, p1[ypos])
                
                p1[xpos][:] = filteredx
                p1[ypos][:] = filteredy

                filteredx = signal.filtfilt(b, a, p2[xpos])
                filteredy = signal.filtfilt(b, a, p2[ypos])
                
                p2[xpos][:] = filteredx
                p2[ypos][:] = filteredy
        else:
            for p in ps:
                filteredx = signal.filtfilt(b, a, p[xpos])
                filteredy = signal.filtfilt(b, a, p[ypos])
                
                p[xpos][:] = filteredx
                p[ypos][:] = filteredy
    
    
    def calulate_velocity(self, pairs, scale):
        """Calculate velocity from positions.

        If particles already have velocity data, this will update the velocity
        in-place.  This is useful when particle position is filtered (e.g.,
        removing the inertial motion) and velocity needs an update.

        Parameters
        ----------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        scale: int
            Scaling the final velocity to a specific unit.
        """
        for p in tqdm(pairs, ncols=80):
            p1, p2 = p

            self._cal_vel(p1, scale)
            self._cal_vel(p2, scale)

    
    
    """""""""""""""""""""""""""""""""""""""
    " Below are pair-related functions.   "
    """""""""""""""""""""""""""""""""""""""

    def group_pairs(self, particles):
        """group all possible pairs from a set of particles.
        
        Parameters
        ----------
        particles: list of xr.Dataset
            A set of particles
        
        Returns
        -------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        """
        pairs = []
        
        for i in range(len(particles)-1):
            for j in range(i+1, len(particles)):
                pairs.append([particles[i], particles[j]])
        
        expected = len(particles) * (len(particles)-1) // 2
        actual = len(pairs)
        
        assert expected == actual
        
        print(f'there are {actual} pairs of particles')
        
        return pairs
    
    def find_pairs(self, pairs, rngs, chancePair=False, t_frac=1e-20, tlen=60, first=True):
        """find pairs if their initial separation is within the range.
        
        Parameters
        ----------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        rngs: list of 2 floats or list of 2-float list
            Separation bounds e.g., [r_lower, r_upper] or [[rl1, ru1], [rl2, ru2]].
        chancePair: bool
            Include chance pair or not.
        t_frac: float
            Time fraction of chance pair relative to the raw pair.  Keep pair
            if its fractional length exceeds this threshold.  For example, 0.9
            means chance pairs containing 90% of the time records of the raw
            pairs will be returned.
        
        Returns
        -------
        origins: list of original pairs (two xr.Datasets) of particles
            Selected original pairs of particles.
        chances: list of chance pairs (two xr.Datasets) of particles
            Selected chance pairs of particles.
        """
        if isinstance(rngs[0], list):
            glen = len(rngs)
        else:
            rngs = [rngs]
            glen = 1

        multiNones = [None] * glen

        origins = [[] for _ in range(glen)] # original pairs
        chances = [[] for _ in range(glen)] # chance pairs
        
        for pair in tqdm(pairs, ncols=80):
            aligned = self._align_by_time(pair)

            # selecting original pairs
            oconds = self._filter_pair(aligned, rngs)
            is_ori = False

            if oconds != None:
                if oconds != multiNones:
                    is_ori = True
                    
                    for ori, con in zip(origins, oconds):
                        if con is not None:
                            ori.append(con)
            
                # selecting chance pairs
                if chancePair:# and not is_ori:
                    cconds = self._filter_chance_pair2(aligned, rngs, t_frac, tlen, first=first)

                    if cconds != None and cconds != multiNones:
                        for cha, con in zip(chances, cconds):
                            if con is not None:
                                cha.append(con) # left pairs
        
        for ori, cha, rng in zip(origins, chances, rngs):
            print(f'original pairs: {len(ori)}, chance pairs: {len(cha)}'+
                  f', with r0 in [{rng[0]}, {rng[1]}]')
        
        if chancePair:
            if glen == 1:
                return origins[0], chances[0]
            else:
                return origins, chances
        else:
            if glen == 1:
                return origins[0]
            else:
                return origins
    
    def get_pairs_information(self, pairs):
        """get pairs information
        
        This extracts pair information from a given list of pairs.
    
        Parameters
        ----------
        pairs: list
            A list of pairs (two xr.Datasets) of particles
    
        Returns
        -------
        pinfo: xarray.Dataset
            Pair information as a xarray.Dataset.
        """
        pc = len(pairs)
        
        pair = np.arange(pc, dtype=np.int32)        # pair index, starts from 0 to the total number of pairs
        particle = np.array([0, 1], dtype=np.int32) # particle index in a single pair
        
        tlen = np.zeros([pc]) # length of each pair
        r0   = np.zeros([pc]) # initial separation of each pair
        stim = np.zeros([pc]) # start time of each pair
        
        gID   = np.zeros([pc, 2]) # two IDs of each pair
        xpos0 = np.zeros([pc, 2]) # initial lons of each pair
        ypos0 = np.zeros([pc, 2]) # initial lats of each pair
        
        for i, p in enumerate(pairs):
            dr1, dr2 = p
            
            gID[i, 0] = round(dr1.ID)
            gID[i, 1] = round(dr2.ID)

            tlen[i] = len(dr1[self.time])
            stim[i] = dr1[self.time].values[0]

            xpos0[i, 0] = dr1[self.xpos].values[0]
            ypos0[i, 0] = dr1[self.ypos].values[0]
            xpos0[i, 1] = dr2[self.xpos].values[0]
            ypos0[i, 1] = dr2[self.ypos].values[0]

            if self.coord == 'latlon':
                xposI, xposJ = np.deg2rad(xpos0[i])
                yposI, yposJ = np.deg2rad(ypos0[i])
                
                r0[i] = geodist(xposI, yposI, xposJ, yposJ)
            else:
                r0[i] = np.hypot(xposI)
        
        tlen  = xr.DataArray(tlen, name='tlen', dims='pair',
                             coords={'pair':pair})
        r0    = xr.DataArray(r0, name='r0', dims='pair',
                             coords={'pair':pair}) * self.Rearth
        stim  = xr.DataArray(stim, name='stim', dims='pair',
                             coords={'pair':pair})
        gID   = xr.DataArray(gID, name='gID', dims=['pair','particle'],
                             coords={'pair':pair, 'particle':particle})
        xpos0 = xr.DataArray(xpos0, name='xpos0', dims=['pair','particle'],
                             coords={'pair':pair, 'particle':particle})
        ypos0 = xr.DataArray(ypos0, name='ypos0', dims=['pair','particle'],
                             coords={'pair':pair, 'particle':particle})
    
        return xr.merge([tlen, stim, r0, gID, xpos0, ypos0])
    
    
    """"""""""""""""""""""""""""""""""""""""""
    " below are measure-related functions.   "
    """"""""""""""""""""""""""""""""""""""""""
    
    def separation_bins(self, r_lower, r_upper, alpha=1.2, method='cist', thre=None, incre=None):
        """Specify the separation bins
        
        Parameters
        ----------
        r_lower: float
            Lower bound of separation.
        r_upper: float
            Upper bound of separation.
        alpha: float
            Increment of neighbouring bins.
        
        Returns
        -------
        rbins: numpy.array
            Separation bins which is uniform in a log scale.
        """
        if thre == None:
            if method == 'fsle':
                n1 = np.floor(np.log(r_lower) / np.log(alpha))
                n2 = np.floor(np.log(r_upper) / np.log(alpha))
    
                rbins = alpha**np.arange(n1, n2)
                rbins = xr.DataArray(rbins, dims='rbin', coords={'rbin':rbins})
            else:
                num = np.log(r_upper/r_lower) / np.log(alpha)
                n   = np.arange(1, np.floor(num)+1)
                rbins = r_lower * alpha**n
                rbins = np.insert(rbins, 0, r_lower)
                rbins = xr.DataArray(rbins, dims='rbin', coords={'rbin':rbins})
        else:
            if method == 'fsle':
                n1 = np.floor(np.log(r_lower) / np.log(alpha))
                n2 = np.floor(np.log(thre) / np.log(alpha))
    
                rbins = np.hstack([alpha**np.arange(n1, n2), np.linspace(thre, r_upper, int((r_upper-thre)/incre))])
                rbins = xr.DataArray(rbins, dims='rbin', coords={'rbin':rbins})
            else:
                num = np.log(thre/r_lower) / np.log(alpha)
                n   = np.arange(1, np.floor(num)+1)
                rbins = r_lower * alpha**n
                rbins = np.hstack([np.insert(rbins, 0, r_lower), np.linspace(thre, r_upper, int((r_upper-thre)/incre))])
                rbins = xr.DataArray(rbins, dims='rbin', coords={'rbin':rbins})

        assert (rbins[1:].values / rbins[:-1].values != alpha).any()
        
        return rbins.rename('rbins')
    
    def stat_rv(self, pairs, reduction='mean'):
        """A statistical function of separation (r) and velocity (v).
        
        Parameters
        ----------
        pairs: list
            A list of pairs (two xr.Datasets) of particles
        power: int
            Power of the statistics.
        
        Returns
        -------
        rx: xarray.DataArray
            x-separation.
        ry: xarray.DataArray
            y-separation.
        rxy: xarray.DataArray
            cross x-y separation.
        r: xarray.DataArray
            separation.
        du: xarray.DataArray
            delta u.
        dv: xarray.DataArray
            delta v.
        dul: xarray.DataArray
            longitudinal velocity.
        dut: xarray.DataArray
            transversal velocity.
        vmi: xarray.DataArray
            velocity magnitude of particle i.
        vmj: xarray.DataArray
            velocity magnitude of particle j.
        uv: xarray.DataArray
            inner product of two particle's velocities.
        """
        return self._map_stat(self._stat_rv, pairs, reduction)
    
    def FSLE_fast(self, ds, rbins, interpT=1):
        """Finite-scale Lyapunov exponent
        
        Fast implementation for many pairs of the same time length
        
        Parameters
        ----------
        ds: xr.Dataset
            All particles.
        rbins: xr.DataArray
            Separation bins which is uniform in a log scale.
        interpT: bool or int
            Increase the temporal resolution or not.
        
        Returns
        -------
        FSLE: xarray.DataArray
            Finite-scale Lyapunov exponent.
        """
        fsle, nums = _FSLE_fast_cartesian(ds.xpos.values, ds.ypos.values, ds.time.values,
                                          rbins=rbins.values, interpT=interpT)
        
        fsle = xr.DataArray(fsle, dims='rbin', coords={'rbin': rbins[1:]}).rename('fsle')
        nums = xr.DataArray(nums, dims='rbin', coords={'rbin': rbins[1:]}).rename('nums')
        
        return fsle, nums
    
    def FSLE(self, pairs, rbins, interpT=1, allPairs=False, reduction='mean'):
        """Finite-scale Lyapunov exponent
        
        Parameters
        ----------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        rbins: xr.DataArray
            Separation bins which is uniform in a log scale.
        interp: int
            Interpolate time to better temporal resolution (little influence).
        allPairs: bool
            All possible pairs or selected pairs.
        reduction: str
            Reduction method of [mean, None].
        
        Returns
        -------
        FSLE: xarray.DataArray
            Finite-scale Lyapunov exponent.
        """
        fsle = (rbins - rbins).rename('fsle')
        nums = (rbins - rbins).rename('nums')

        fv = fsle.values[:-1]
        nv = nums.values[:-1]

        fslelst = []

        for pair in tqdm(pairs, ncols=80):
            if allPairs:
                aligned = self._align_by_time(pair)
            else:
                aligned = pair

            if aligned != None and len(aligned[0].time) > 1:
                dr1, dr2 = aligned

                if allPairs:
                    tt = np.arange(len(dr1.time)) * 15.0 / (60. * 24.)
                    dr1[self.time] = tt
                    dr2[self.time] = tt
                else:
                    tt = dr1[self.time]

                if type(interpT) is int:
                    interpT = np.linspace(tt[0], tt[-1],
                                        int((len(tt)-1)*interpT+1))

                tmp1, tmp2 = self._FSLE(aligned, rbins, interpT)

                if reduction == 'mean':
                    fv += np.where(np.isnan(tmp1), 0, tmp1)
                    nv += tmp2.values
                elif reduction == None:
                    fslelst.append(tmp1)
                    nv += tmp2.values
        
        if reduction == None:
            fsle = xr.concat(fslelst, dim='pair')
        elif reduction == 'mean':
            fsle = fsle / nums
        
        return fsle, nums.rename('num_fsle')
    
    def structure_functions(self, pairs, rbins, allPairs=True, reduction='mean'):
        """Structure functions
        
        Parameters
        ----------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        rbins: xr.DataArray
            Separation bins which is uniform in a log scale.
        allPairs: bool
            All possible pairs or selected pairs.
        reduction: str
            Reduction method of [mean, None].
        
        Returns
        -------
        S2: xarray.DataArray
            Second-order structure function.
        S2L: xarray.DataArray
            Second-order longitudinal structure function.
        S2T: xarray.DataArray
            Second-order transversal structure function.
        S3: xarray.DataArray
            Third-order structure function.
        """
        S2  = (rbins - rbins).rename('S2' )
        S2L = (rbins - rbins).rename('S2L')
        S2T = (rbins - rbins).rename('S2T')
        S3  = (rbins - rbins).rename('S3' )
        num = (rbins - rbins).rename('num')

        s2 = S2 .values[:-1]
        sl = S2L.values[:-1]
        st = S2T.values[:-1]
        s3 = S3 .values[:-1]
        nv = num.values[:-1]

        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []

        for pair in tqdm(pairs, ncols=80):
            if allPairs:
                aligned = self._align_by_time(pair)
            else:
                aligned = pair

            if aligned != None:
                dr1, dr2 = aligned

                if allPairs:
                    tt = np.arange(len(dr1.time)) * 15.0 / (60. * 24.)
                    dr1[self.time] = tt
                    dr2[self.time] = tt
                else:
                    tt = dr1[self.time]

                rx, ry, rxy, r, du, dv, dul, dut, vmi, vmj, uv = self._stat_rv([dr1, dr2])
                
                tmp1 = histogram(r.rename('rtmp'), bins=rbins.values, weights=du**2+dv**2      )
                tmp2 = histogram(r.rename('rtmp'), bins=rbins.values, weights=dul**2           )
                tmp3 = histogram(r.rename('rtmp'), bins=rbins.values, weights=dut**2           )
                tmp4 = histogram(r.rename('rtmp'), bins=rbins.values, weights=dul*(du**2+dv**2))
                tmp5 = histogram(r.rename('rtmp'), bins=rbins.values, weights=du-du+1          )
                
                if reduction == 'mean':
                    s2 += np.where(np.isnan(tmp1), 0, tmp1)
                    sl += np.where(np.isnan(tmp2), 0, tmp2)
                    st += np.where(np.isnan(tmp3), 0, tmp3)
                    s3 += np.where(np.isnan(tmp4), 0, tmp4)
                    nv += tmp5.values
                
                elif reduction == None:
                    lst1.append(tmp1)
                    lst2.append(tmp2)
                    lst3.append(tmp3)
                    lst4.append(tmp4)
                    nv += tmp5.values
        
        if reduction == None:
            S2  = xr.concat(lst1, dim='pair') / num
            S2L = xr.concat(lst2, dim='pair') / num
            S2T = xr.concat(lst3, dim='pair') / num
            S3  = xr.concat(lst4, dim='pair') / num
        elif reduction == 'mean':
            S2  = S2  / num
            S2L = S2L / num
            S2T = S2T / num
            S3  = S3  / num
        
        return S2.rename('S2'), S2L.rename('S2L'), S2T.rename('S2T'), S3.rename('S3'), num.rename('num_s2')

    
    def FAGR(self, pairs, rbins, allPairs=True, reduction='mean'):
        """Finite-amplitude growth rate
        
        Parameters
        ----------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        rbins: xr.DataArray
            Separation bins which is uniform in a log scale.
        allPairs: bool
            All possible pairs or selected pairs.
        reduction: str
            Reduction method of [mean, None].
        
        Returns
        -------
        FAGR: xarray.DataArray
            FAGR.
        FAGRp: xarray.DataArray
            Positive FAGR, which is similar to FSLE.
        """
        FAGR  = (rbins - rbins).rename('FAGR' )
        FAGRp = (rbins - rbins).rename('FAGRp')
        nums  = (rbins - rbins).rename('num'  )
        numsp = (rbins - rbins).rename('nump' )
        
        fagr  = FAGR .values[:-1]
        fagrp = FAGRp.values[:-1]
        nv    = nums .values[:-1]
        nvp   = numsp.values[:-1]

        lst1 = []
        lst2 = []

        for pair in tqdm(pairs, ncols=80):
            if allPairs:
                aligned = self._align_by_time(pair)
            else:
                aligned = pair

            if aligned != None:
                dr1, dr2 = aligned

                if allPairs:
                    tt = np.arange(len(dr1.time)) * 15.0 / (60. * 24.)
                    dr1[self.time] = tt
                    dr2[self.time] = tt
                else:
                    tt = dr1[self.time]
                
                if len(dr1[self.time]) > 2:
                    rx, ry, rxy, r, du, dv, dul, dut, vmi, vmj, uv = self._stat_rv([dr1, dr2])
                    
                    sFAGR = np.log(r).ffill('time').differentiate('time') # single-realization of FAGR
                    wei = xr.where(sFAGR>0, 1, 0)
                    tmp1 = histogram(r.rename('rtmp'), bins=rbins.values, weights=sFAGR)
                    tmp2 = histogram(r.rename('rtmp'), bins=rbins.values, weights=sFAGR*wei)
                    tmp3 = histogram(r.rename('rtmp'), bins=rbins.values, weights=r-r+1)
                    tmp4 = histogram(r.rename('rtmp'), bins=rbins.values, weights=wei)
                    
                    if reduction == 'mean':
                        fagr  += np.where(np.isnan(tmp1), 0, tmp1)
                        fagrp += np.where(np.isnan(tmp2), 0, tmp2)
                        nv    += tmp3.values
                        nvp   += tmp4.values
                    
                    elif reduction == None:
                        lst1.append(tmp1)
                        lst2.append(tmp2)
                        nv  += tmp3.values
                        nvp += tmp4.values
                    
        
        if reduction == None:
            FAGR  = xr.concat(lst1, dim='pair') / nums
            FAGRp = xr.concat(lst2, dim='pair') / numsp
        elif reduction == 'mean':
            FAGR  = FAGR  / nums
            FAGRp = FAGRp / numsp
        
        return FAGR.rename('FAGR'), FAGRp.rename('FAGRp'), nums.rename('num_fagr'), numsp.rename('num_fagrp')
    
    
    def relative_diffusivity(self, pairs, rbins, allPairs=True, reduction='mean'):
        """Relative diffusivity averaged at constant separation
        
        Parameters
        ----------
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        rbins: xr.DataArray
            Separation bins which is uniform in a log scale.
        allPairs: bool
            All possible pairs or selected pairs.
        reduction: str
            Reduction method of [mean, None].
        
        Returns
        -------
        FAGR: xarray.DataArray
            FAGR.
        FAGRp: xarray.DataArray
            Positive FAGR, which is similar to FSLE.
        """
        K2   = (rbins - rbins).rename('K2')
        nums = (rbins - rbins).rename('num')
        
        k2   = K2 .values[:-1]
        nv   = nums .values[:-1]
        
        lst1 = []

        for pair in tqdm(pairs, ncols=80):
            if allPairs:
                aligned = self._align_by_time(pair)
            else:
                aligned = pair

            if aligned != None:
                dr1, dr2 = aligned

                if allPairs:
                    tt = np.arange(len(dr1.time)) * 15.0 / (60. * 24.)
                    dr1[self.time] = tt
                    dr2[self.time] = tt
                else:
                    tt = dr1[self.time]
                
                if len(dr1[self.time]) > 2:
                    rx, ry, rxy, r, du, dv, dul, dut, vmi, vmj, uv = self._stat_rv([dr1, dr2])
                    
                    sk2  = (r**2).differentiate('time') # single-realization
                    tmp1 = histogram(r.rename('rtmp'), bins=rbins.values, weights=sk2 )
                    tmp2 = histogram(r.rename('rtmp'), bins=rbins.values, weights=r-r+1)
                    
                    if reduction == 'mean':
                        k2  += np.where(np.isnan(tmp1), 0, tmp1)
                        nv  += tmp2.values
                    
                    elif reduction == None:
                        lst1.append(tmp1)
                        nv += tmp2.values
                    
        
        if reduction == None:
            K2 = xr.concat(lst1, dim='pair') / nums
        elif reduction == 'mean':
            K2 = K2  / nums
        
        return K2.rename('K2'), nums.rename('num_K2')
    
    
    def PDF(self, r, rbins):
        """Probability density function of pair separations
        
        Parameters
        ----------
        r: xarray.DataArray
            Pair separations, typically as a function of ['pair', 'time'].
        rbins: numpy.array
            Separation bins which is uniform in a log scale.
        
        Returns
        -------
        PDF: xarray.DataArray
            Probability density function.
        """
        tmp = rbins.values if isinstance(rbins, xr.DataArray) else rbins
        PDF = histogram(r, bins=tmp, dim=['pair'], density=True).rename('PDF')

        return PDF.rename({r.name+'_bin':'r'})
    
    def CDF(self, PDF, bin_edges=None):
        """Cumulative density function of pair separations
        
        Parameters
        ----------
        PDF: xarray.DataArray
            Probability density function of pair separations.
        bin_edges: numpy.array
            1D array of bin edges (N+1 length).
        
        Returns
        -------
        CDF: xarray.DataArray
            Cumulative density function.
        """
        if bin_edges is None:
            values = PDF['r'].diff('r').values
            values = np.insert(values, 0, values[0])
            bin_width = xr.DataArray(values, dims='r', coords={'r':PDF['r'].values})
        else:
            bin_width = xr.DataArray(np.diff(bin_edges), dims='r',
                                     coords={'r':PDF['r'].values})
        
        return (PDF * bin_width).cumsum('r').rename('CDF')
    
    def CIST(self, CDF, lower, upper, maskout=None):
        """Cumulative inverse separation time

        This is a new diagnostic proposed by LaCasce and Meunier (2022, JFM).
        
        Parameters
        ----------
        CDF: xarray.DataArray
            Cumulative density function of pair separations.
        lower: float
            Lower bound of CDF.
        upper: float
            Upper bound of CDF.
        maskout: list of float
            A range of valid results e.g., [minvalue, maxvalue].
        
        Returns
        -------
        CIST: xarray.DataArray
            Cumulative inverse separation time (unit of inverse time of CDF).
        """
        CDFrng = CDF.where(np.logical_and(CDF>lower, CDF<upper))
        
        slope, inter, rms = xr.apply_ufunc(semilog_fit, CDFrng[self.time], CDFrng,
                                           dask='allowed',
                                           input_core_dims=[[self.time], [self.time]],
                                           output_core_dims=[[], [], []],
                                           vectorize=True)
        
        fitted = np.exp((0.5 - inter) / slope)
        diff = fitted.diff('r')
        cist  = 1.0 / diff

        if maskout:
            cist = cist.where(np.logical_and(cist>maskout[0], cist<maskout[1]))
        
        return cist.rename('cist')

    
    def bootstrap(self, vars, func, ensemble=1000, CI=0.95):
        """Calculate standard error and confidence interval

        Both standard error and CI are obtained through bootstrapping.

        Reference:
        https://www.dummies.com/article/academics-the-arts/science/biology/the-bootstrap-method-for-standard-errors-and-confidence-intervals-164614/
        https://www.stat.cmu.edu/~ryantibs/advmethods/notes/bootstrap.pdf
        https://www.schmidheiny.name/teaching/bootstrap2up.pdf
        
        Parameters
        ----------
        vars: list of xr.DataArray
            A set of given variables.
        func: function
            A function that transform a group of samples into a metric.
        ensemble: int
            How many times bootstrapping is done.
        CI: float
            Confidence interval.
        
        Returns
        -------
        err: xr.DataArray
            Standard error.
        CIL: xr.DataArray
            Lower bound of confidence intervals.
        CIU: xr.DataArray
            Upper bound of confidence intervals.
        """
        if not isinstance(vars, list):
            raise Exception('vars should be a list of xr.DataArray')
        
        size = len(vars[0]['pair'])
        
        tmp = []
        for i in range(ensemble):
            indices  = np.random.choice(range(size), size=size, replace=True)
            resample = [var.isel({'pair':indices}) for var in vars]
            metrics  = func(resample)
            tmp.append(metrics)
        
        re = xr.concat(tmp, dim='_ensem')
        re['_ensem'] = np.arange(ensemble)
        
        half = (1.0 - CI) / 2.0
        qntl = re.quantile([half, 1.0 - half], dim='_ensem', skipna=True)
        stde = re.std('_ensem')
        #emn  = re.mean('_ensem')
        #CIL  = (2.0 * emn - qntl.isel(quantile=1)).drop_vars('quantile')
        #CIU  = (2.0 * emn - qntl.isel(quantile=0)).drop_vars('quantile')
        CIL  = qntl.isel(quantile=0).drop_vars('quantile')
        CIU  = qntl.isel(quantile=1).drop_vars('quantile')
        
        return stde, CIL, CIU

    
    def principle_axis_components(self, rx2, ry2, rxy):
        """Calculate principle axis components
        
        Parameters
        ----------
        rx2: xr.DataArray
            Zonal component
        ry2: xr.DataArray
            Meridional component
        rxy: xr.DataArray
            Cross component
        
        Returns
        -------
        ra2: xr.DataArray
            Major component.
        rb2: xr.DataArray
            Minor component.
        rthe: xr.DataArray
            Angle between major and zonal components.
        """
        ra2  = (rx2 + ry2 + np.sqrt((rx2 - ry2)**2 + 4 * rxy**2)) / 2.0
        rb2  = rx2 + ry2 - ra2
        the = np.arctan2(ra2 - rx2, rxy)

        return ra2, rb2, the

    
    """"""""""""""""""""""""""""""""""""""""""
    " Below are the private helper methods.  "
    """"""""""""""""""""""""""""""""""""""""""
    
    def _map_stat(self, stat_func, pairs, reductions='mean', **kwargs):
        """apply a statistical function to all the pairs.
        
        Parameters
        ----------
        stat_func: function
            A function applies to a single pair.
        pairs: list of pairs (two xr.Datasets) of particles
            All possible pairs of particles.
        mean: bool
            Whether take an average over pair dimension and return the mean.
        returnnum: bool
            Return number of pairs or not.
        
        Returns
        -------
        re: list of xr.DataArray
            Metrics depending on stat_func.
        nums: xr.DataArray
            Number of valid pairs in deriving these statistics.
        """
        # try first pair to determine the length of the outputs
        if reductions == None:
            reductions = 'None'
        
        tmp = stat_func(pairs[0], **kwargs)

        if isinstance(tmp, (list, set, tuple)): # multiple returns
            N = len(tmp)

            if isinstance(reductions, str):
                reductions = [reductions] * N
            
            if len(reductions) != N:
                raise Exception('length of reductions not equal the no. of statistics')
            
            res = []
            
            for i in range(N):
                res.append([])
            
            for p in tqdm(pairs, ncols=80):
                tmp = stat_func(p, **kwargs)
                
                for i in range(N):
                    res[i].append(tmp[i])

            res = [self._reductOp_fast(var, red) for var, red in zip(res, reductions)]
            
        else: # single return
            if isinstance(reductions, list):
                if len(reductions) != 1:
                    raise Exception('length of reductions should be one')
                
                reductions = reductions[0]
            
            res = self._reductOp_fast([stat_func(p, **kwargs) for p in tqdm(pairs, ncols=80)],
                            reductions)
        
        return res
    
    def _reductOp(self, var, method):
        if method == 'mean':
            return xr.concat(var, dim='pair').mean('pair')
        elif method == 'sum':
            return xr.concat(var, dim='pair').sum('pair')
        elif method == 'None':
            return xr.concat(var, dim='pair')
        else:
            raise Exception(f'invalid method {method}')
    
    def _reductOp_fast(self, var, method):
        re = self._new_DataArray(var)
        data = re.values
        
        for i in range(len(re['pair'])):
            tmp = var[i].values
            data[i, :len(tmp)] = tmp
        
        if method == 'mean':
            return re.mean('pair')
        elif method == 'sum':
            return re.sum('pair')
        elif method == 'None':
            return re
        else:
            raise Exception(f'invalid method {method}')

    def _new_DataArray(self, var):
        npair = len(var)
        ntime = -1
        times = None

        for i in range(npair):
            length = len(var[i])
            
            if length > ntime:
                ntime = length
                times = var[i][self.time]

        if ntime != -1:
            data = np.zeros([npair, ntime], dtype=np.float32) * np.nan
            
            return xr.DataArray(data, dims=['pair', self.time],
                                coords={'pair':np.arange(npair), self.time:times})
        else:
            raise Exception('invalid size of array')

    def _stat_rv(self, pair):
        """A statistical function of separation (r) and velocity (v) of a single pair.
        
        Parameters
        ----------
        pair: list
            A pair (two xr.Datasets) of particles
        
        Returns
        -------
        rx: float
            x-separation.
        ry: float
            y-separation.
        rxy: float
            cross separation.
        r: float
            total separation.
        du: float
            delta u.
        dv: float
            delta v.
        dul: float
            longitudinal velocity.
        dut: float
            transversal velocity.
        vmi: float
            velocity magnitude of particle i.
        vmj: float
            velocity magnitude of particle j.
        uv: float
            inner product of two particle's velocities.
        """
        pi, pj = pair
        
        Rearth     = self.Rearth
        xpos, ypos = self.xpos, self.ypos
        uvel, vvel = self.uvel, self.vvel
        ui, uj, vi, vj = pi[uvel], pj[uvel], pi[vvel], pj[vvel]
        
        du  = ui - uj
        dv  = vi - vj
        vmi = np.hypot(ui, vi)
        vmj = np.hypot(uj, vj)
        uv  = ui * uj + vi * vj
        
        if self.coord == 'latlon':
            xi = np.deg2rad(pi[xpos])
            xj = np.deg2rad(pj[xpos])
            yi = np.deg2rad(pi[ypos])
            yj = np.deg2rad(pj[ypos])
            
            rx = ((xi - xj) * np.cos((yi + yj)/2.0) * Rearth)
            ry = ((yi - yj) * Rearth)
            rxy= rx * ry
            r  = np.hypot(rx, ry)
            
            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            r   = (geodist(xi, xj, yi, yj) * Rearth)
            
        else:
            dx, dy = pi[xpos] - pj[xpos], pi[ypos] - pj[ypos]
            
            rx = dx
            ry = dy
            rxy = dx * dy
            r   = np.hypot(rx, ry)

            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            
        return rx, ry, rxy, r, du, dv, dul, dut, vmi, vmj, uv
    
    def _FSLE(self, pair, rbins=None, interpT=False):
        """Finite-scale Lyapunov exponent
        
        Parameters
        ----------
        pair: list
            A pairs (two xr.Datasets) of particles.
        rbins: xarray.DataArray
            Separation bins which is uniform in a log scale
        
        Returns
        -------
        FSLE: float
            Finite-scale Lyapunov exponent.
        """
        drfi, drfj = pair
        xpos, ypos = self.xpos, self.ypos
        alpha = rbins.values
        alpha = alpha[-1] / alpha[-2] # ratio of neighbouring bins
        
        if self.coord == 'latlon':
            xi = np.deg2rad(drfi[xpos])
            xj = np.deg2rad(drfj[xpos])
            yi = np.deg2rad(drfi[ypos])
            yj = np.deg2rad(drfj[ypos])
            
            rp = (geodist(xi, xj, yi, yj) * self.Rearth)
            
        else:
            xi = drfi[xpos]
            xj = drfj[xpos]
            yi = drfi[ypos]
            yj = drfj[ypos]
            
            rp = np.hypot(xi - xj, yi - yj)
        
        if interpT is None or interpT is False:
            r_interp = rp
        else:
            r_interp = rp.interp({self.time:interpT})
        
        rd = r_interp[r_interp.argmin().values:]
        Td = xr.where(rd > rbins, 1, np.nan).idxmax(self.time)
        FSLE = Td.diff(rbins.dims[0])
        FSLE = (np.log(alpha) / FSLE.where(FSLE != 0)).rename('FSLE')
        nums = xr.where(np.isnan(FSLE), 0, 1)
        
        return FSLE, nums
    
    def _filter_pair(self, aligned, rngs):
        """select pair if its initial separation (r0) is within given range.
        
        Select pair if
            1) r0 in rng=[r_lower, r_upper]
            2) aligned records > 0.
        
        Parameters
        ----------
        aligned: list
            A pair (two xr.Datasets) of particles aligned by time (same t-length).
        rngs: list of 2 floats or list of 2-float list
            Separation bounds e.g., [r_lower, r_upper] or [[rl1, ru1], [rl2, ru2]].
        
        Returns
        -------
        aligned
            Aligned pair if valid, otherwise None, in each ranges.
        """
        if aligned is None:
            return None
        
        r0 = self._R_initial(aligned)

        conds = [aligned if rng[0] <= r0 <= rng[1] else None for rng in rngs]
        
        return conds
    
    def _filter_chance_pair(self, aligned, rngs, t_frac=1.0, tlen=60, first=True):
        """select pair if its separation at some time is within the given range.
        
        Select pair if
            1) lower <= r(n) <= upper, and
            2) aligned records > 0.
        
        Parameters
        ----------
        aligned: list
            A pair (two xr.Datasets) of particles aligned by time (same t-length).
        rngs: list of 2 floats or list of 2-float list
            Separation bounds e.g., [r_lower, r_upper] or [[rl1, ru1], [rl2, ru2]].
        t_frac: float
            Time fraction of chance pair relative to the raw pair.
        
        Returns
        -------
        aligned list
            Aligned pair if valid, otherwise None.
        """
        if aligned is None:
            return None
        
        Rearth     = self.Rearth
        drfi, drfj = aligned
        xpos, ypos = self.xpos, self.ypos
        
        if self.coord == 'latlon':
            xi = np.deg2rad(drfi[xpos])
            xj = np.deg2rad(drfj[xpos])
            yi = np.deg2rad(drfi[ypos])
            yj = np.deg2rad(drfj[ypos])
            rp = geodist(xi, xj, yi, yj) * Rearth
            
        else:
            xi = drfi[xpos]
            xj = drfj[xpos]
            yi = drfi[ypos]
            yj = drfj[ypos]
            rp = np.hypot(xi - xj, yi - yj)
        
        conds = []
        
        for rng in rngs:
            minIdx  = rp.argmin(self.time).values
            chanced = np.logical_and(rng[0] <= rp, rp <= rng[1])
            
            if chanced.any():
                leng = len(drfi[self.time])
                idx  = -1
                
                if first:
                    idx = chanced.values.nonzero()[0][0]
                else:
                    for index in chanced.values.nonzero()[0]:
                        if index >= minIdx:
                            idx = index
                            break

                if idx == -1:
                    conds.append(None)
                else:
                    frac = 1.0 - float(idx) / leng
                    tsize = leng - idx
    
                    if (t_frac <= frac < 1) and (tsize >= tlen):
                        truncatedi = drfi.isel({self.time: slice(idx, -1)})
                        truncatedj = drfj.isel({self.time: slice(idx, -1)})
                        
                        conds.append([truncatedi, truncatedj])
                    else:
                        conds.append(None)
                
            else:
                conds.append(None)
        
        return conds
    
    def _filter_chance_pair2(self, aligned, rngs, t_frac=1.0, tlen=60, first=True):
        """select pair if its separation at some time is within the given range.
        
        Select pair if
            1) lower <= r(n) <= upper, and
            2) aligned records > 0.
        
        Parameters
        ----------
        aligned: list
            A pair (two xr.Datasets) of particles aligned by time (same t-length).
        rngs: list of 2 floats or list of 2-float list
            Separation bounds e.g., [r_lower, r_upper] or [[rl1, ru1], [rl2, ru2]].
        t_frac: float
            Time fraction of chance pair relative to the raw pair.
        
        Returns
        -------
        aligned list
            Aligned pair if valid, otherwise None.
        """
        if aligned is None:
            return None
        
        Rearth     = self.Rearth
        drfi, drfj = aligned
        xpos, ypos = self.xpos, self.ypos
        
        if self.coord == 'latlon':
            xi = np.deg2rad(drfi[xpos])
            xj = np.deg2rad(drfj[xpos])
            yi = np.deg2rad(drfi[ypos])
            yj = np.deg2rad(drfj[ypos])
            rp = geodist(xi, xj, yi, yj) * Rearth
            
        else:
            xi = drfi[xpos]
            xj = drfj[xpos]
            yi = drfi[ypos]
            yj = drfj[ypos]
            rp = np.hypot(xi - xj, yi - yj)
        
        conds = []
        
        for rng in rngs:
            minIdx  = rp.argmin(self.time).values
            chanced = (rng[0] <= rp.values[minIdx]) and (rp.values[minIdx] <= rng[1])
            #print(drfi.ID, drfj.ID, rp.values[minIdx], minIdx)
            
            #if rng[0]==0.08 and drfi.ID==286 and drfj.ID==307:
            #    print(minIdx, chanced, len(drfi[self.time]) - minIdx, 1.0 - float(minIdx) / len(drfi[self.time]))
            
            if chanced and len(rp[self.time]) >= 4 * 24 * 10: # > 10 day
                tsize = len(drfi[self.time]) - minIdx
                frac  = 1.0 - float(minIdx) / len(drfi[self.time])
                
                if (t_frac <= frac <= 1):# and (tsize >= tlen):
                    truncatedi = drfi.isel({self.time: slice(minIdx, -1)})
                    truncatedj = drfj.isel({self.time: slice(minIdx, -1)})
                    #print(truncatedi.ID, truncatedj.ID, len(truncatedi[self.time]))

                    if len(truncatedi[self.time]) != 0:
                        conds.append([truncatedi, truncatedj])
                    else:
                        conds.append(None)
                else:
                    conds.append(None)
                
            else:
                conds.append(None)
        
        return conds
    
    def _R_initial(self, pair):
        """Calculate initial separation.
        
        Parameters
        ----------
        pair: list
            A pair (two xr.Datasets) of particles.
        
        Returns
        -------
        d_init: float
            Initial separation of this pair particles.
        """
        xpos, ypos = self.xpos, self.ypos
        drfi, drfj = pair
        
        if self.coord == 'latlon':
            xi = drfi[xpos].values[0]
            yi = drfi[ypos].values[0]
            xj = drfj[xpos].values[0]
            yj = drfj[ypos].values[0]
            
            xi, yi, xj, yj = np.deg2rad([xi, yi, xj, yj])
            
            d_init = geodist(xi, xj, yi, yj) * self.Rearth
            
        else:
            xi = drfi[xpos].values[0]
            yi = drfi[ypos].values[0]
            xj = drfj[xpos].values[0]
            yj = drfj[ypos].values[0]
            
            d_init = np.hypot(xi - xj, yi - yj)
            
        return d_init
    
    def _cal_vel(self, particle, scale=1.0):
        """Calculate velocity using finite difference method.
        
        Central finite difference scheme is used in the interior and
        forward/backward finite difference scheme is used at the end points.

        Parameters
        ----------
        particle: xr.Dataset
            A single particle.
        scale: float
            Scaling the final velocity to a specific unit.
        """
        xpos  = self.xpos
        ypos  = self.ypos
        time  = self.time
        deg2m = self.deg2m

        xs = particle[self.xpos]
        ys = particle[self.ypos]

        uvel = xs.differentiate(time)
        vvel = ys.differentiate(time)
        
        if self.coord == 'latlon':
            uvel *= deg2m * np.cos(np.deg2rad(ys)) * scale
            vvel *= deg2m * scale

        particle[self.uvel][:] = uvel.values
        particle[self.vvel][:] = vvel.values
    
    def _align_by_time(self, pair):
        """Align a pair of particle by their time series.
        
        Parameters
        ----------
        pair: list
            A pair (two xr.Datasets) of particles.
        
        Returns
        -------
        pair: list
            Aligned pair (two particles with the same time span).
        """
        aligned = xr.align(pair[0], pair[1])
        
        if len(aligned[0][self.time]) < 1:
            return None
        else:
            return aligned
    
    def _remove_none(self, ls):
        """Remove None in a given list.
        
        Parameters
        ----------
        ls: list
            A list of pairs, may including None.
        
        Returns
        -------
        pair: list
            A resulted list without None.
        """
        return list(filter(None, ls))

    def __repr__(self):
        """Print this class as a string"""
        return \
            ' relative dispersion class with:\n'\
            '   xpos: ' + self.xpos  + '\n'\
            '   ypos: ' + self.ypos  + '\n'\
            '   uvel: ' + self.uvel  + '\n'\
            '   vvel: ' + self.vvel  + '\n'\
            '   time: ' + self.time  + '\n'\
            '  coord: ' + self.coord + '\n'\



"""
Helper (private) methods are defined below
"""

@nb.jit(nopython=True, cache=False)
def _FSLE_fast_cartesian(xpos, ypos, time, rbins, interpT=1):
    """Calculate FSLE using numba for efficiency

    This is for Cartesian coordinates.  Assumed that all trajectories have the same time range.
    
    Parameters
    ----------
    xpos: numpy.ndarray
        x-position (in meter).
    ypos: numpy.ndarray
        y-position (in meter).
    time: numpy.ndarray
        Relative time axis.
    rbins: numpy.ndarray
        Separation scales against which FSLE is defined.
    
    Returns
    -------
    FSLE: numpy.ndarray
        FSLE as a function of Rscales.
    """
    J, I = xpos.shape # assume to be [ID, time]
    
    tmp = np.zeros(len(rbins)-1)
    cnt = np.zeros(len(rbins)-1)
    ts  = np.zeros(rbins.shape)
    tim = np.interp(np.linspace(0, len(time)-1, (len(time)-1)*interpT+1), np.arange(len(time)), time)
    alpha = rbins[-1] / rbins[-2]
    
    for j in range(J-1):
        p1x = xpos[j]
        p1y = ypos[j]
        
        for i in range(j+1, J):
            p2x = xpos[i]
            p2y = ypos[i]
            
            sep = np.hypot(np.abs(p1x - p2x), np.abs(p1y - p2y))
            sep = np.interp(np.linspace(0, len(sep)-1, (len(sep)-1)*interpT+1), np.arange(len(sep)), sep)
            sep = sep[sep.argmin():]
            
            if len(sep) > 2:
                ts.fill(np.nan)
                idx = -1
                
                for r, Rs in enumerate(rbins):
                    for s in range(len(sep)):
                        if sep[s] > Rs:
                            idx = s
                            break
                    
                    if idx != -1:
                        ts[r] = tim[idx]
                # print(f'tmp.shape: {tmp.shape}')
                # print(f'ts.shape: {ts.shape}')
                FSLE = np.diff(ts)
                # print(f'FSLE.shape: {FSLE.shape}')
                FSLE = (np.log(alpha) / np.where(FSLE != 0, FSLE, np.nan))
                
                tmp += np.where(np.isnan(FSLE), 0, FSLE)
                cnt += np.where(np.isnan(FSLE), 0, 1)
    
    return tmp / cnt, cnt




