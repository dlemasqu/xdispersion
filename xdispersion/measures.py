# -*- coding: utf-8 -*-
"""
Created on 2025.02.26

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from typing import Optional, Tuple, Literal, Union, List
from xhistogram.xarray import histogram
from .utils import semilog_fit, mean_at_rbin, gen_rbins, bootstrap


default_rbins = gen_rbins(0.01, 1000, alpha=1.2)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Below are functions for separation/velocity measures "
"                                                      "
"   Suffix '_t' means averaged at constant time, and   "
"   suffix '_r' means averaged at constant rbin.       "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def rel_disp(
    r: xr.DataArray,
    order: Optional[int] = 2,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate moments of relative separation

    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    order: int
        Order of moment for dispersion
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    rN: xarray.DataArray
        Nth-moment of separation.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    rN = r ** order
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            return v.mean('pair')
        elif mean_at == 'const-r':
            return mean_at_rbin(v, r, rbins)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(rN, r).rename(f'r{order}_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [rN, r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(rN, r).rename(f'r{order}_{mean_at[-1]}'),\
               lb.rename(f'LBr{order}_{mean_at[-1]}'),\
               ub.rename(f'UBr{order}_{mean_at[-1]}')


def vel_struct_func(
    du: xr.DataArray,
    r: xr.DataArray,
    order: Optional[int] = 2,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-r',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate velocity structure function

    Given du, one gets zonal component S2x;
    Given dv, one gets meridional component S2y;
    Given hypot(du, dv), one gets total S2;
    Given dul, one gets longitudinal component S2ll;
    Given dut, one gets transversal component S2tr;
    Given dul*hypot(du, dv)**2, with order=1, one gets S3;
    
    Parameters
    ----------
    du: xarray.DataArray
        relative velocity.
    r: xarray.DataArray
        Relative separation.
    order: int
        Order of moment for dispersion
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    SN: xr.DataArray
        Nth-order velocity structure funciton.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    SN = du ** order
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            return v.mean('pair')
        elif mean_at == 'const-r':
            return mean_at_rbin(v, r, rbins)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(SN, r).rename(f'S{order}_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [SN, r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(SN, r).rename(f'S{order}_{mean_at[-1]}'),\
               lb.rename(f'LBS{order}_{mean_at[-1]}'),\
               ub.rename(f'UBS{order}_{mean_at[-1]}')


def rel_diff(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate relative diffusivity
    
    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    K2: xr.DataArray
        Relative diffusivity.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # ffill() to ensure nans will not propagate to
    # contaminate the results via finite differencing
    K2 = (r**2.0).ffill('rtime').differentiate('rtime') / 2.0
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            return v.mean('pair')
        elif mean_at == 'const-r':
            return mean_at_rbin(v, r, rbins)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(K2, r).rename(f'K2_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [K2, r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(K2, r).rename(f'K2_{mean_at[-1]}'),\
               lb.rename(f'LBK2_{mean_at[-1]}'),\
               ub.rename(f'UBK2_{mean_at[-1]}')


def famp_growth_rate(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    positive: Optional[bool] = True,
    mean_at: Literal['const-t', 'const-r'] = 'const-r',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate finite-amplitude growth rate
    
    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    positive: boolean
        Average positive or all FAGR. Positive FAGR is close to FSLE.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    FAGR: xr.DataArray
        Finite-amplitude growth rate.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # ffill() to ensure nans will not propagate to
    # contaminate the results via finite differencing
    sFAGR = np.log(r).ffill('rtime').differentiate('rtime')
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            if positive:
                return v.where(v>0).mean('pair')
            else:
                return v.mean('pair')
        elif mean_at == 'const-r':
            if positive:
                return mean_at_rbin(v, r, rbins, cond=v>0)
            else:
                return mean_at_rbin(v, r, rbins)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
    
    p = 'p' if positive else ''
    
    if ensemble <= 0:
        return how_to_mean(sFAGR, r).rename(f'FAGR{p}_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [sFAGR, r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(sFAGR, r).rename(f'FAGR{p}_{mean_at[-1]}'),\
               lb.rename(f'LBFAGR{p}_{mean_at[-1]}'),\
               ub.rename(f'UBFAGR{p}_{mean_at[-1]}')


def init_memory(
    rx: xr.DataArray,
    ry: xr.DataArray,
    du: xr.DataArray,
    dv: xr.DataArray,
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate initial memory
    
    Parameters
    ----------
    rx: xarray.DataArray
        Zonal component of separation.
    ry: xarray.DataArray
        meridional component of separation.
    du: xarray.DataArray
        Zonal component of relative velocity.
    dv: xarray.DataArray
        Meridional component of relative velocity.
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    initm: xr.DataArray
        initial memory.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    initm = rx.isel(rtime=0) * du + ry.isel(rtime=0) * dv
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            return v.mean('pair')
        elif mean_at == 'const-r':
            return mean_at_rbin(v, r, rbins)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(initm, r).rename(f'initm_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [initm, r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(initm, r).rename(f'initm_{mean_at[-1]}'),\
               lb.rename(f'LBinitm_{mean_at[-1]}'),\
               ub.rename(f'UBinitm_{mean_at[-1]}')

def anisotropy(
    rx: xr.DataArray,
    ry: xr.DataArray,
    rxy: xr.DataArray,
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate anisotropy
    
    Parameters
    ----------
    rx: xr.DataArray
        Zonal component of dispersion
    ry: xr.DataArray
        Meridional component of dispersion
    rxy: xr.DataArray
        Cross component of dispersion
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    aniso: xr.DataArray
        Anisotropy.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    
    Returns
    -------
    ani: xr.DataArray
        Anisotropy.
    """
    # define how to take average
    def how_to_mean(rx, ry, rxy, r):
        if mean_at == 'const-t':
            rx2m = (rx**2.0).mean('pair')
            ry2m = (ry**2.0).mean('pair')
            rxym = rxy.mean('pair')
            ra2m, rb2m, _ = principle_axis_components(rx2m, ry2m, rxym)
            return np.sqrt(ra2m / rb2m)
        elif mean_at == 'const-r':
            rx2m = mean_at_rbin(rx**2.0, r, rbins)
            ry2m = mean_at_rbin(ry**2.0, r, rbins)
            rxym = mean_at_rbin(rxy, r, rbins)
            ra2m, rb2m, _ = principle_axis_components(rx2m, ry2m, rxym)
            return np.sqrt(ra2m / rb2m)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(rx, ry, rxy, r).rename(f'aniso_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [rx, ry, rxy, r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(rx, ry, rxy, r).rename(f'aniso_{mean_at[-1]}'),\
               lb.rename(f'LBaniso_{mean_at[-1]}'),\
               ub.rename(f'UBaniso_{mean_at[-1]}')


def lagr_vel_corr(
    uv: xr.DataArray,
    vs1: xr.DataArray,
    vs2: xr.DataArray,
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate Lagrangian velocity correlation
    
    Parameters
    ----------
    uv: xr.DataArray
        Cross-variation of velocity
    vs1: xr.DataArray
        Velocity magnitude of first particle
    vs2: xr.DataArray
        Velocity magnitude of second particle
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    lvc: xr.DataArray
        Lagrangian velocity correlation, within [-1, 1].
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # define how to take average
    def how_to_mean(uv, v1, v2, r):
        if mean_at == 'const-t':
            uvm = uv.mean('pair')
            v1m = (v1**2.0).mean('pair')
            v2m = (v2**2.0).mean('pair')
            return (2.0 * uvm) / (v1m + v2m)
        elif mean_at == 'const-r':
            uvm = mean_at_rbin(uv     , r, rbins)
            v1m = mean_at_rbin(v1**2.0, r, rbins)
            v2m = mean_at_rbin(v2**2.0, r, rbins)
            return (2.0 * uvm) / (v1m + v2m)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(uv, vs1, vs2, r).rename(f'lvc_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [uv, vs1, vs2, r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(uv, vs1, vs2, r).rename(f'lvc_{mean_at[-1]}'),\
               lb.rename(f'LBlvc_{mean_at[-1]}'),\
               ub.rename(f'UBlvc_{mean_at[-1]}')


def kurtosis(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate Kurtosis
    
    Parameters
    ----------
    r: xr.DataArray
        Relative separation r
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    Ku: xr.DataArray
        Kurtosis.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # define how to take average
    def how_to_mean(r):
        if mean_at == 'const-t':
            r4m = (r**4).mean('pair')
            r2m = (r**2).mean('pair')
            return r4m / r2m ** 2
        elif mean_at == 'const-r':
            r4m = mean_at_rbin(r**4, r, rbins)
            r2m = mean_at_rbin(r**2, r, rbins)
            return r4m / r2m ** 2
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(r).rename(f'Ku_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(r).rename(f'Ku_{mean_at[-1]}'),\
               lb.rename(f'LBKu_{mean_at[-1]}'),\
               ub.rename(f'UBKu_{mean_at[-1]}')


def cen_vul_exp(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate Cencini-Vulpiani exponent or proxy FSLE

    This should be similar to FAGR.
    
    Parameters
    ----------
    r: xr.DataArray
        Relative separation r
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    CVE: xr.DataArray
        Cencini-Vulpiani exponent.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # define how to take average
    def how_to_mean(r):
        r2 = (r ** 2.0)
        K2 = r2.differentiate('rtime') / 2.0
        
        if mean_at == 'const-t':
            return K2.mean('pair') / r2.mean('pair')
        elif mean_at == 'const-r':
            return mean_at_rbin(K2, r, rbins) / rbins**2.0
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(r).rename(f'K2_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(r).rename(f'K2_{mean_at[-1]}'),\
               lb.rename(f'LBK2_{mean_at[-1]}'),\
               ub.rename(f'UBK2_{mean_at[-1]}')


def fsize_lyap_exp(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    interpT: Optional[int] = 1,
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate finite-size Lyapunov exponent
    
    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    interpT: int (> 0)
        Interpolate along time dimension so that FSLE can be resolved
        at the smallest scales.  1 means no interpolation.
    
    Returns
    -------
    FSLE: float
        Finite-Scale Lyapunov Exponent.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    def get_Td(r_single, rbins):
        if interpT > 1:
            rtime = r_single.rtime
            timeInt = np.linspace(rtime[0], rtime[-1], int((len(rtime)-1)*interpT+1))
            rinterp = r_single.interp(rtime=timeInt)
        elif interpT == 1:
            rinterp = r_single
        else:
            raise Exception(f'invalid interpT {interpT}, should be larger than 0')
        
        rd = rinterp[rinterp.argmin().values:]
        return xr.where(rd > rbins, 1, np.nan).idxmax('rtime')
    
    alpha = rbins.values[-1] / rbins.values[-2] # ratio of neighbouring bins

    # loop over each pair to get Td
    Td = []
    for i in range(len(r['pair'])):
        Td.append(get_Td(r.isel(pair=i), rbins))

    Td = xr.concat(Td, dim='pair')
    
    FSLE = Td.diff('rbin')
    FSLE = (np.log(alpha) / FSLE.where(FSLE != 0))
    
    # define how to take average
    def how_to_mean(v):
        if mean_at == 'const-r':
            return v.mean('pair')
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be only const-r')
        
    if ensemble <= 0:
        return how_to_mean(FSLE).rename(f'FSLE_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [FSLE], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(FSLE).rename(f'FSLE_{mean_at[-1]}'),\
               lb.rename(f'LBFSLE_{mean_at[-1]}'),\
               ub.rename(f'UBFSLE_{mean_at[-1]}')


def cumul_inv_sep_time(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    lower: Optional[float] = 0.1,
    upper: Optional[float] = 0.9,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    maskout: List[float] = [1e-8, 5e4],
    interpT: Optional[int] = 1,
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Definition of cumulative inverse separation time

    This is a new diagnostic proposed by LaCasce and Meunier (2022, JFM).
    
    Parameters
    ----------
    r: xarray.DataArray
        Pair separations, typically as a function of ['pair', 'time'].
    rbins: numpy.array
        Separation bins which is uniform in a logarithm scale.
    lower: float
        Lower bound of CDF.
    upper: float
        Upper bound of CDF.
    maskout: list of float
        A range of valid results e.g., [minvalue, maxvalue].
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    CIST: xarray.DataArray
        Cumulative inverse separation time (unit of inverse time of CDF).
    """
    # define how to take average
    def how_to_mean(v):
        if mean_at == 'const-r':
            CDF = cumul_dens_func(prob_dens_func(v, rbins), rbins)
            CDFrng = CDF.where(np.logical_and(CDF>lower, CDF<upper))
            
            slope, inter, rms = xr.apply_ufunc(semilog_fit, CDFrng['rtime'], CDFrng,
                                               dask='allowed',
                                               input_core_dims=[['rtime'], ['rtime']],
                                               output_core_dims=[[], [], []],
                                               vectorize=True)
            
            fitted = np.exp((0.5 - inter) / slope)
            diff = fitted.diff('rbin')
            CIST  = 1.0 / diff
            
            if maskout:
                CIST = CIST.where(np.logical_and(CIST>maskout[0], CIST<maskout[1]))
            
            return CIST
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be only const-r')
        
    if ensemble <= 0:
        return how_to_mean(r).rename(f'CIST_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [r], {},
                           ensemble=ensemble, CI=CI)
        
        return how_to_mean(r).rename(f'CIST_{mean_at[-1]}'),\
               lb.rename(f'LBFSLE_{mean_at[-1]}'),\
               ub.rename(f'UBFSLE_{mean_at[-1]}')


def prob_dens_func(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
) -> xr.DataArray:
    """Definition of probability density function of pair separation r
    
    Parameters
    ----------
    r: xarray.DataArray
        Pair separations, typically as a function of ['pair', 'time'].
    rbins: numpy.array
        Separation bins which is uniform in a logarithm scale.
    
    Returns
    -------
    PDF: xarray.DataArray
        Probability density function.
    """
    tmp = rbins.values if isinstance(rbins, xr.DataArray) else rbins
    PDF = histogram(r.rename('r'), bins=tmp, dim=['pair'], density=True).rename('PDF')
    PDF['r_bin'] = tmp[1:]
    
    return PDF.rename({'r_bin':'rbin'})


def cumul_dens_func(
    PDF: xr.DataArray,
    bin_edges: Union[xr.DataArray, np.array] = None,
) -> xr.DataArray:
    """Definition of cumulative density function of pair separation r
    
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
        values = PDF['rbin'].diff('rbin').values
        values = np.insert(values, 0, values[0])
        bin_width = xr.DataArray(values, dims='rbin', coords={'rbin':PDF['rbin'].values})
    else:
        bin_width = xr.DataArray(np.diff(bin_edges), dims='rbin',
                                 coords={'rbin':PDF['rbin'].values})
    
    return (PDF * bin_width).cumsum('rbin').rename('CDF')


"""
Helper methods are defined below
"""
def principle_axis_components(
    rx2m: xr.DataArray,
    ry2m: xr.DataArray,
    rxym: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Calculate principle axis components
    
    Parameters
    ----------
    rx2m: xr.DataArray
        Zonal component of dispersion <rx^2>
    ry2m: xr.DataArray
        Meridional component of dispersion <ry^2>
    rxym: xr.DataArray
        Cross component of dispersion <rxy>
    
    Returns
    -------
    ra: xr.DataArray
        Major component of separation.
    rb: xr.DataArray
        Minor component of separation.
    ang: xr.DataArray
        Angle between major and zonal components.
    """
    ra2 = (rx2m + ry2m + np.sqrt((rx2m - ry2m)**2 + 4 * rxym**2)) / 2.0
    rb2 = rx2m + ry2m - ra2
    ang = np.arctan2(ra2 - rx2m, rxym)
    
    return ra2, rb2, ang


def rotational_divergent_components(
    S2ll: xr.DataArray,
    S2tr: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Calculate rotational and divergent components of velocity structure function
    
    Parameters
    ----------
    S2ll: xr.DataArray
        Longitudinal component of 2nd-order velocity structure function.
    S2tr: xr.DataArray
        Transversal component of 2nd-order velocity structure function.
    
    Returns
    -------
    S2rr: xr.DataArray
        Rotational component of 2nd-order velocity structure function.
    S2dd: xr.DataArray
        Divergent component of 2nd-order velocity structure function.
    """
    rr = S2ll.rbin
    
    S2rr = S2tr + ((S2tr - S2ll)/rr*rr.diff('rbin')).cumsum('rbin')
    S2dd = S2ll - ((S2tr - S2ll)/rr*rr.diff('rbin')).cumsum('rbin')
    
    return S2rr.rename('S2rr'), S2dd.rename('S2dd')

