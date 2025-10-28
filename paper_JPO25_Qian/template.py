# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from utils import mean_at_rbin


"""
A template for calculating all the available
measures given a set of pair particles, and
return all as a xarray.Dataset that can be
easily output to a file.
"""

def cal_all_measures(rd, pairs, rbins):
    """Calculate all available measures.  Users can add their own measures.
    
    Parameters
    ----------
    rd: RelativeDispersion instance
        Relative dispersion application
    pairs: list of two xr.Datasets
        Pair particles
    rbins: xr.DataArray
        A specified separation bins for r-based measures
    
    Returns
    -------
    dset: xr.Dataset
        All measures in a single xr.Dataset.
    """
    #----------- functions below are used for bootstrapping ---------------
    def func_mean(samples):
        return samples[0].mean('pair')
    
    def func_ani(samples):
        rx2, ry2, rxy = samples
        rx2m = rx2.mean('pair')
        ry2m = ry2.mean('pair')
        rxym = rxy.mean('pair')
        ra2m, rb2m, rthem = rd.principle_axis_components(rx2m, ry2m, rxym)
        return ra2m / rb2m
    
    def func_lvc(samples):
        uv, vm1, vm2 = samples
        uvm  =  uv.mean('pair')
        vm1m = vm1.mean('pair')
        vm2m = vm2.mean('pair')
        return 2.0 * uvm / (vm1m + vm2m)
    
    def func_Ku(samples):
        r4, r2 = samples
        r4m = r4.mean('pair')
        r2m = r2.mean('pair')
        return r4m / (r2m ** 2.0)
    
    def func_S2t(samples):
        du2, dv2 = samples
        du2m = du2.mean('pair')
        dv2m = dv2.mean('pair')
        return du2m + dv2m
    
    def func_S2r(samples):
        du2, dv2, r = samples
        S2r = mean_at_rbin(du2+dv2, r, rbins.values)
        return S2r
    
    def func_K2r(samples):
        r2, r = samples
        K2r = mean_at_rbin(r2.differentiate('time')/2, r, rbins.values)
        return K2r
    
    def func_K2t(samples):
        r2m = samples[0].mean('pair')
        return r2m.differentiate('time') / 2.0
    
    def func_FAGRp(samples):
        sFAGR, r = samples
        FAGR = mean_at_rbin(sFAGR, r, rbins.values, cond=sFAGR>0)
        return FAGR
    
    def func_FAGR(samples):
        sFAGR, r = samples
        FAGR = mean_at_rbin(sFAGR, r, rbins.values)
        return FAGR
    
    def func_cist(samples):
        PDFm = rd.PDF(samples[0], rbins)
        CDFm = rd.CDF(PDFm, rbins)
        return rd.CIST(CDFm, 0.1, 0.9, maskout=[1e-6, 1e2])
    
    #----------------------- calculate separation, velocity, FSLE --------------------#
    rx, ry, rxy, r, du, dv, dul, dut, vmi, vmj, uv = rd.stat_rv(pairs, reduction=None)
    FSLE, nums2 = rd.FSLE(pairs, rbins, interpT=False, allPairs=False, reduction=None)
    
    #----------------------- average at constant separation rbins --------------------#
    trms = mean_at_rbin(r.time           , r, rbins.values)
    S2r  = mean_at_rbin((du**2)+(dv**2)  , r, rbins.values)
    S2ll = mean_at_rbin((dul**2)         , r, rbins.values)
    S2tr = mean_at_rbin((dut**2)         , r, rbins.values)
    S3r  = mean_at_rbin(dul*(du**2+dv**2), r, rbins.values)
    
    # ffill() to ensure nans will not propagate to contaminate the results
    K2r   = mean_at_rbin((r**2).ffill('time').differentiate('time') / 2.0, r, rbins.values)
    sFAGR = np.log(r).ffill('time').differentiate('time') # single-realization of FAGR
    FAGR  = mean_at_rbin(sFAGR, r, rbins.values)
    FAGRp = mean_at_rbin(sFAGR, r, rbins.values, cond=(sFAGR>0))
    
    #---------------------------------- boostrapping -------------------------------#
    stdeFS , CILFS , CIUFS  = rd.bootstrap([FSLE]              , func=func_mean)
    stderx , CILrx , CIUrx  = rd.bootstrap([rx**2]             , func=func_mean)
    stdery , CILry , CIUry  = rd.bootstrap([ry**2]             , func=func_mean)
    stder  , CILr  , CIUr   = rd.bootstrap([r**2]              , func=func_mean)
    stdeK2r, CILK2r, CIUK2r = rd.bootstrap([r**2, r]           , func=func_K2r )
    stdeK2t, CILK2t, CIUK2t = rd.bootstrap([r**2]              , func=func_K2t )
    stdeKu , CILKu , CIUKu  = rd.bootstrap([r**4 , r**2]       , func=func_Ku  )
    stdeS2r, CILS2r, CIUS2r = rd.bootstrap([du**2, dv**2, r]   , func=func_S2r )
    stdeS2t, CILS2t, CIUS2t = rd.bootstrap([du**2, dv**2]      , func=func_S2t )
    stdean , CILan , CIUan  = rd.bootstrap([rx**2, ry**2, rxy] , func=func_ani )
    stdelv , CILlv , CIUlv  = rd.bootstrap([uv, vmi**2, vmj**2], func=func_lvc )
    stdeFG , CILFG , CIUFG  = rd.bootstrap([sFAGR, r]          , func=func_FAGR)
    stdeFGp, CILFGp, CIUFGp = rd.bootstrap([sFAGR, r]          , func=func_FAGRp)
    stdeci , CILci , CIUci  = rd.bootstrap([r.rename('r')]     , func=func_cist)
    
    #-------------------------- get PDF and number of pairs -----------------------#
    PDF = rd.PDF(r.rename('r'), rbins)
    nums = xr.where(~np.isnan(r), 1, 0).sum('pair')
    
    #---------------------------- average at constant time ------------------------#
    rx2 = (rx**2).mean('pair')
    ry2 = (ry**2).mean('pair')
    rxy =   (rxy).mean('pair')
    r2  =  (r**2).mean('pair')
    r4  =  (r**4).mean('pair')
    du2 = (du**2).mean('pair')
    dv2 = (dv**2).mean('pair')
    vmi =(vmi**2).mean('pair')
    vmj =(vmj**2).mean('pair')
    uv  =  uv.mean('pair')
    FSLE = FSLE.mean('pair')
    
    #------------------------------- various measures ----------------------------#
    ra2, rb2, rthe = rd.principle_axis_components(rx2, ry2, rxy)
    r2N = rd.numerical_r2(PDF, scaled2PiR=True)
    lvc = 2.0 *uv /(vmi + vmj)
    initm = (rx[:, 0] * du + ry[:, 0] * dv).mean('pair')
    initn = (rx[:, 0] * du + ry[:, 0] * dv).mean('pair') / (r2[0] * (du**2 + dv**2).mean('pair'))
    initn2 = (rx[:, 0] * du + ry[:, 0] * dv).mean('pair') / (r2[0] + (du**2 + dv**2).mean('pair'))
    anixy = rx2 / ry2
    aniab = ra2 / rb2
    Ku  = r4 / (r2 ** 2.0)
    KuN = rd.numerical_Kurtosis(PDF, scaled2PiR=True)
    S2t = (du2 + dv2)
    K2t = r2.differentiate('time') / 2.0
    CVE = K2t / r2     # should be similar to FAGR
    K2tN= rd.numerical_K2(PDF, scaled2PiR=True)
    K2tN= mean_at_rbin(K2tN, K2tN.r, rbins.values)
    pFSLE = K2t / r2
    CDF = rd.CDF(PDF, rbins)
    cist1 = rd.CIST(CDF, 0.10, 0.90, maskout=[1e-6, 1e2])
    cist2 = rd.CIST(CDF, 0.15, 0.85, maskout=[1e-6, 1e2])
    cist3 = rd.CIST(CDF, 0.20, 0.80, maskout=[1e-6, 1e2])
    cistN = rd.numerical_CIST(PDF, 0.1, 0.9, maskout=[1e-6, 1e2], scaled2PiR=True)
    
    #------------------------------- output list ----------------------------#
    vs = [rx2, ry2, rxy, r2, r2N, r4, ra2, rb2, rthe,
          trms, initm, initn, initn2, FAGR, FAGRp,
          du2, dv2, vmi, vmj, uv, lvc, anixy, aniab, Ku, KuN,
          S2r, S2ll, S2tr, S2t, S3r, K2r, K2t, K2tN, pFSLE,
          stderx, stdery, stder, CILrx, CILry, CILr, CIUrx, CIUry, CIUr,
          stdeK2r, CILK2r, CIUK2r, stdeK2t, CILK2t, CIUK2t,
          stdeKu, CILKu, CIUKu, stdeS2r, CILS2r, CIUS2r, stdeS2t, CILS2t, CIUS2t, stdean, CILan, CIUan,
          stdelv, CILlv, CIUlv, stdeFG, CILFG, CIUFG, stdeFGp, CILFGp, CIUFGp, stdeci, CILci, CIUci,
          PDF, CDF, cist1, cist2, cist3, cistN, FSLE, nums, nums2, stdeFS, CILFS, CIUFS]
    
    #------------------------------- output names ----------------------------#
    names = ['rx2', 'ry2', 'rxy', 'r2', 'r2N', 'r4', 'ra2', 'rb2', 'rthe',
             'trms', 'initm', 'initn', 'initn2', 'FAGR', 'FAGRp',
             'du2', 'dv2', 'vm1', 'vm2', 'uv', 'lvc', 'anixy', 'aniab', 'Ku', 'KuN',
             'S2r', 'S2ll', 'S2tr', 'S2t', 'S3r', 'K2r', 'K2t', 'K2tN', 'pFSLE',
             'stderx', 'stdery', 'stdre', 'CILrx', 'CILry', 'CILr', 'CIUrx', 'CIUry', 'CIUr',
             'stdeK2r', 'CILK2r', 'CIUK2r', 'stdeK2t', 'CILK2t', 'CIUK2t',
             'stdeKu', 'CILKu', 'CIUKu', 'stdeS2r', 'CILS2r', 'CIUS2r', 'stdeS2t', 'CILS2t', 'CIUS2t', 'stdean', 'CILan', 'CIUan',
             'stdelv', 'CILlv', 'CIUlv', 'stdeFG', 'CILFG', 'CIUFG', 'stdeFGp', 'CILFGp', 'CIUFGp', 'stdeci', 'CILci', 'CIUci',
             'PDF', 'CDF', 'cist1', 'cist2', 'cist3', 'cistN', 'FSLE', 'nums', 'nums2', 'stdeFSLE', 'CILFSLE', 'CIUFSLE']
    
    #------------------------------- output comments ----------------------------#
    comments = ['x-component of relative dispersion',
                'y-component of relative dispersion',
                'cross-component of relative dispersion',
                'total relative dispersion',
                'numerical total relative dispersion',
                'total 4th-order separation',
                'major-component of relative dispersion',
                'minor-component of relative dispersion',
                'angle of the major axis',
                'mean time at constant r',
                'initial memory term',
                'normalized initial memory term *',
                'normalized initial memory term +',
                'finite-amplitude growth rate',
                'finite-amplitude growth rate (positive values)',
                'squared u-velocity difference',
                'squared v-velocity difference',
                'squared speed of the first particle',
                'squared speed of the second particle',
                'velocity projection of the first particle onto the second one',
                'Lagrangian velocity correlation',
                'anisotropy using x/y components <rx2>/<ry2>',
                'anisotropy using major/minor components <ra2>/<rb2>',
                'kurtosis', 'numerical kurtosis',
                '2nd-order structure function-r',
                '2nd-order longitudinal structure function',
                '2nd-order transversal structure function',
                '2nd-order structure function-t',
                '3rd-order structure function-r',
                'relative diffusivity-r',
                'relative diffusivity-t',
                'numerical relative diffusivity-t',
                'proxy finite-size Lyapunov exponent',
                'standard error for rx2',
                'standard error for ry2',
                'standard error for r2',
                'lower-bound of confidence interval for rx2',
                'lower-bound of confidence interval for ry2',
                'lower-bound of confidence interval for r2',
                'upper-bound of confidence interval for rx2',
                'upper-bound of confidence interval for ry2',
                'upper-bound of confidence interval for r2',
                'standard error for K2r',
                'lower-bound of confidence interval for K2r',
                'upper-bound of confidence interval for K2r',
                'standard error for K2t',
                'lower-bound of confidence interval for K2t',
                'upper-bound of confidence interval for K2t',
                'standard error for Ku',
                'lower-bound of confidence interval for Ku',
                'upper-bound of confidence interval for Ku',
                'standard error for S2r',
                'lower-bound of confidence interval for S2r',
                'upper-bound of confidence interval for S2r',
                'standard error for S2t',
                'lower-bound of confidence interval for S2t',
                'upper-bound of confidence interval for S2t',
                'standard error for anisotropy',
                'lower-bound of confidence interval for anisotropy',
                'upper-bound of confidence interval for anisotropy',
                'standard error for lvc',
                'lower-bound of confidence interval for lvc',
                'upper-bound of confidence interval for lvc',
                'standard error for FAGR',
                'lower-bound of confidence interval for FAGR',
                'upper-bound of confidence interval for FAGR',
                'standard error for FAGRp',
                'lower-bound of confidence interval for FAGRp',
                'upper-bound of confidence interval for FAGRp',
                'standard error for cist',
                'lower-bound of confidence interval for cist',
                'upper-bound of confidence interval for cist',
                'probability density function of r',
                'cumulative density function of r',
                'cumulative inverse separation time 0.1-0.9',
                'cumulative inverse separation time 0.15-0.85',
                'cumulative inverse separation time 0.2-0.8',
                'numerical CIST 0.1-0.9',
                'finite size lyapunov exponent',
                'number of pairs',
                'number of pairs in FSLE',
                'standard error of FSLE',
                'lower-bound of confidence interval for FSLE',
                'upper-bound of confidence interval for FSLE']

    if len(vs) != len(names) or len(vs) != len(comments):
        raise Exception(f'invalid lengths: {len(vs)}, {len(names)}, {len(comments)}')
    
    tmp = []
    for v, n, c in zip(vs, names, comments):
        v = v.rename(n)
        v.attrs['comment'] = c
        tmp.append(v)

    return xr.merge(tmp)




"""
Below are functions used for bootstrapping
"""
