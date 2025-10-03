# -*- coding: utf-8 -*-
"""
Created on Thur Nov  20 17:24:46 2024

@author: Yu-Kun Qian
"""
import numpy as np
import mpmath as mpm
import xarray as xr
from typing import Optional, List, Dict, Tuple
from scipy.special import i0, ive, gammaincinv, gamma


"""
This contains the analytical predictions of several measures
of relative dispersion in different turbulent regimes:

K^-3  : -3 or nonlocal or Lundgren regime
K^-2  : -2 regime (similar to GM internal wave spectrum???)
K^-5/3: -5/3 or local or Richardson regime
decorr: diffusive or asymptotic or Rayleigh regime

The analytical predictions are separated into full ones, and
asymptotic ones which are simpler than the full ones.

All the predictions are derived from the full expressions of
p.d.f. of pair separation under different regimes.  Since the
full solutions of some measures are too complex to write out,
one may get the full predictions by numerical integration of
the full expressions of the p.d.f.
"""

def ana_r2(
    r: xr.DataArray,
    params: Dict[str, str],
    regime: str
) -> xr.DataArray:
    """Calculate analytic relative dispersion
    
    Four regimes are available:
      * -3   | nonlocal | Lundgren   regime -> r2(t) ~ exp(t)
      * -2   | local    | Richardson regime -> r2(t) ~ t^(4)
      * -5/3 | local    | GM         regime -> r2(t) ~ t^(3)
      * diffusive       | rayleigh   regime -> r2(t) ~ t^(1)

    Note that the asymptotic prediction uses "-a" suffix
    after regime str e.g., ['Lundgren-a'].
    
    Parameters
    ----------
    r: xr.DataArray
        A given separation.
    params: dict
        Parameters used in the expression of each regime.
        Should be ['T', 'beta', 'lambda', 'k2']
    regime: str
        Regime name shown above e.g., ['nonlocal', '-5/3', ...]
    
    Returns
    -------
    r2: xr.DataArray
        Analytical prediction of relative dispersion.
    """
    ts = r['time'] if 'time' in r.coords else r['rtime']

    if 'r0' in params:
        r0 = params['r0']
    else:
        r0 = r[0].values
    
    if regime.lower() in ['lundgren', 'nonlocal', '-3']:
        TL = params['T']
        r2 = r0**2.0 * np.exp(8.0 * ts / TL)
        
    elif regime.lower() in ['-2']:
        def func(tmp, coeff): # expression
            return  coeff * mpm.hyp1f1(8, 4, tmp) * mpm.exp(-tmp)
        
        lmbd = params['lambda']
        tmp = (4.0 * r0**(1.0/2.0)) / (lmbd * ts)
        coeff= gamma(8) / gamma(4) * (lmbd * ts / 4.0)**4.0
        newF = np.frompyfunc(func, 2, 1)
        r2   = newF(tmp, coeff).astype(r.dtype)
        
    elif regime.lower() in ['-2-a']:
        r2 = 3.28125 * (params['lambda'] * ts) ** 4.0
        
    elif regime.lower() in ['richardson', 'local', '-5/3']:
        def func(tmp, coeff): # expression
            return  coeff * mpm.hyp1f1(6, 3, tmp) * mpm.exp(-tmp)
        
        beta = params['beta']
        tmp  = (9.0 * r0**(2.0/3.0)) / (4.0 * beta * ts)
        coeff= gamma(6) * (4.0 * beta * ts / 9.0)**3.0 / 2.0
        newF = np.frompyfunc(func, 2, 1)
        r2   = newF(tmp, coeff).astype(r.dtype)
        
    elif regime.lower() in ['richardson-a', 'local-a', '-5/3-a']:
        r2 = 5.2675 * (params['beta'] * ts) **3.0
        
    elif regime.lower() in ['rayleigh', 'diffusive']:
        r2 = 4.0 * params['k2'] * ts + r0**2.0
        
    elif regime.lower() in ['rayleigh-a', 'diffusive-a']:
        r2 = 4.0 * params['k2'] * ts
        
    else:
        raise Exception(f'invalid regime {regime}, '+\
                        'should be [Lundgren, Richardson, Rayleigh], '+\
                        'or [local, nonlocal, -3, -2, -5/3]')

    r2[0] = r0 ** 2.0
    
    return r2.rename('r2_' + regime[:2])


def ana_Ku(
    r: xr.DataArray,
    params: Dict[str, str],
    regime: str
) -> xr.DataArray:
    """Calculate analytic kurtosis
    
    Four regimes are available:
      * -3   | nonlocal | Lundgren   regime -> Ku(t) ~ exp(t)
      * -2   | local    | Richardson regime -> Ku(t) ~ 9.4286
      * -5/3 | local    | GM         regime -> Ku(t) ~ 5.6
      * diffusive       | rayleigh   regime -> Ku(t) ~ 2
    
    Parameters
    ----------
    r: xr.DataArray
        A given separation.
    params: dict
        Parameters used in the expression of each regime.
        Should be ['T', 'beta', 'lambda', 'k2']
    regime: str
        Regime name shown above e.g., ['nonlocal', '-5/3', ...]
    
    Returns
    -------
    Ku: xr.DataArray
        Analytic prediction of Kurtosis.
    """
    ts = r['time'] if 'time' in r.coords else r['rtime']
    
    if regime.lower() in ['lundgren', 'nonlocal', '-3']:
        Ku = np.exp(8.0 * ts / params['T'])
        
    elif regime.lower() in ['-2']:
        def func(tmp): # expression, use mpm instead of scipy to avoid overflow
            coeff = gamma(4) * gamma(12) / gamma(8) **2.0
            return  coeff * mpm.hyp1f1(12, 4, tmp) / mpm.hyp1f1(8, 4, tmp) **2.0 * mpm.exp(tmp)
        
        r0   = params['r0']
        lmbd = params['lambda']
        tmp  = (4.0 * r0**(1.0/2.0)) / (lmbd * ts)
        newF = np.frompyfunc(func, 1, 1)
        Ku   = newF(tmp).astype(r.dtype)
        
    elif regime.lower() in ['-2-a']:
        Ku = ts - ts + 9.4286

    elif regime.lower() in ['richardson', 'local', '-5/3']:
        def func(tmp): # expression, use mpm instead of scipy to avoid overflow
            coeff = 2.0 * gamma(9) / gamma(6) **2.0
            return  coeff * mpm.hyp1f1(9, 3, tmp) / mpm.hyp1f1(6, 3, tmp) **2.0 * mpm.exp(tmp)
        
        r0   = params['r0']
        beta = params['beta']
        tmp  = (9.0 * r0**(2.0/3.0)) / (4.0 * beta * ts)
        newF = np.frompyfunc(func, 1, 1)
        Ku   = newF(tmp).astype(r.dtype)
        
    elif regime.lower() in ['richardson-a', 'local-a', '-5/3-a']:
        Ku = ts - ts + 5.6

    elif regime.lower() in ['rayleigh', 'diffusive']:
        r0 = params['r0']
        k2 = params['k2']

        tmp = r0 ** 2 / (64 * k2 * ts)
        Ku  = (tmp**2/16 + tmp + 1/32) / (2*tmp + 1/8)**2
        
    elif regime.lower() in ['rayleigh-a', 'diffusive-a']:
        Ku = ts - ts + 2.0

    else:
        raise Exception(f'invalid regime {regime}, '+\
                        'should be [Lundgren, Richardson, Rayleigh], '+\
                        'or [local, nonlocal, -3, -2, -5/3]')
    
    return Ku.rename('Ku_' + regime[:2])


def ana_K2(
    r: xr.DataArray,
    params: Dict[str, str],
    regime: str
) -> xr.DataArray:
    """Calculate analytic relative diffusivity
    
    Four regimes are available:
      * -3   | nonlocal | Lundgren   regime -> K2(r) ~ T^-1   r^(2)
      * -2   | local    | Richardson regime -> K2(r) ~ beta   r^(4/3)
      * -5/3 | local    | GM         regime -> K2(r) ~ lambda r^(3/2)
      * diffusive       | rayleigh   regime -> K2(r) ~ r^(0)
    
    Parameters
    ----------
    r: xr.DataArray
        A given separation.
    params: dict
        Parameters used in the expression of each regime.
        Should be ['T', 'beta', 'lambda', 'k2']
    regime: str
        Regime name shown above e.g., ['nonlocal', '-5/3', ...]
    
    Returns
    -------
    K2: xr.DataArray
        Analytic prediction of relative diffusivity.
    """
    if regime.lower() in ['lundgren', 'nonlocal', '-3']:
        K2 = 4.0 / params['T'] * r**2.0
    elif regime.lower() in ['-2']:
        K2 = 2.69178 * params['lambda'] * r **(3.0/2.0)
    elif regime.lower() in ['richardson', 'local', '-5/3']:
        K2 = 2.6099 * params['beta'] * r **(4.0/3.0)
    elif regime.lower() in ['rayleigh', 'diffusive']:
        K2 = r - r + 2.0 * params['k2']
    else:
        raise Exception(f'invalid regime {regime}, '+\
                        'should be [Lundgren, Richardson, Rayleigh], '+\
                        'or [local, nonlocal, -3, -2, -5/3]')
    
    return K2.rename('K2_' + regime[:2])


def ana_S2(
    r: xr.DataArray,
    params: Dict[str, str],
    regime: str
) -> xr.DataArray:
    """Calculate scaling of 2nd-order velocity structure function
    
    Four regimes are available:
      * -3   | nonlocal | Lundgren   regime -> S2(r) ~ r^(2)
      * -2   | local    | Richardson regime -> S2(r) ~ r^(1)
      * -5/3 | local    | GM         regime -> S2(r) ~ r^(2/3)
      * diffusive       | rayleigh   regime -> S2(r) ~ r^(0)

    Note that this is scaling only.  Params can be arbitary here.
    
    Parameters
    ----------
    r: xr.DataArray
        A given separation.
    params: dict
        Parameters used in the expression of each regime.
        Should be ['T', 'beta', 'lambda', 'k2']
    regime: str
        Regime name shown above e.g., ['nonlocal', '-5/3', ...]
    
    Returns
    -------
    S2: xr.DataArray
        Scaling of 2nd-order velocity structure function.
    """
    if regime.lower() in ['lundgren', 'nonlocal', '-3']:
        S2 = r **2.0 / params['T']
    elif regime.lower() in ['-2']:
        S2 = params['lambda'] * r **1.0
    elif regime.lower() in ['richardson', 'local', '-5/3']:
        S2 = params['beta'] * r **(2.0/3.0)
    elif regime.lower() in ['rayleigh', 'diffusive']:
        S2 = r - r + params['k2']
    else:
        raise Exception(f'invalid regime {regime}, '+\
                        'should be [Lundgren, Richardson, Rayleigh], '+\
                        'or [local, nonlocal, -3, -2, -5/3]')
    
    return S2.rename('S2_' + regime[:2])


def ana_S3(
    r: xr.DataArray,
    params: Dict[str, str],
    regime: str
) -> xr.DataArray:
    """Calculate analytic 3rd-order velocity structure function
    
    Four regimes are available:
      * -3   | nonlocal | Lundgren   regime -> S2(r) ~ r^(3)
      * -2   | local    | Richardson regime -> S2(r) ~ r^(3/2)
      * -5/3 | local    | GM         regime -> S2(r) ~ r^(1)
      * diffusive       | rayleigh   regime -> S2(r) ~ r^(0) ???

    Note that this is scaling only.  Params can be arbitary here.
    
    Parameters
    ----------
    r: xr.DataArray
        A given separation.
    params: dict
        Parameters used in the expression of each regime.
        Should be ['T', 'beta', 'lambda', 'k2']
    regime: str
        Regime name shown above e.g., ['nonlocal', '-5/3', ...]
    
    Returns
    -------
    S3: xr.DataArray
        Scaling of 3rd-order structure function.
    """
    if regime.lower() in ['lundgren', 'nonlocal', '-3']:
        S3 = r **3.0 / params['T']
    elif regime.lower() in ['-2']:
        S3 = params['lambda'] * r **1.5
    elif regime.lower() in ['richardson', 'local', '-5/3']:
        S3 = params['beta'] * r **1.0
    elif regime.lower() in ['rayleigh', 'diffusive']:
        S3 = r - r + params['k2']
    else:
        raise Exception(f'invalid regime {regime}, '+\
                        'should be [Lundgren, Richardson, Rayleigh], '+\
                        'or [local, nonlocal, -3, -2, -5/3]')
    
    return S3.rename('S3_' + regime[:2])


def ana_PDF(
    r: xr.DataArray,
    t: xr.DataArray,
    params: Dict[str, str],
    regime: str
) -> xr.DataArray:
    """Calculate analytic PDF of pair separation
    
    Five regimes are available:
      * -3   | nonlocal | Lundgren   regime
      * -2   | local    | Richardson regime
      * -5/3 | local    | GM         regime
      * diffusive       | rayleigh   regime
      * generalized regime, covering (but not limited to)
        the first three regimes
    
    Parameters
    ----------
    r: xr.DataArray
        A given separation
    t: xr.DataArray
        A given time
    params: dict
        Parameters used in the expression of each regime.
        Should be ['T', 'beta', 'lambda', 'k2', 'generalized'].
        Additional ones are ['r0', 'slope'], means inital
        separation and spectral slope (e.g., -3, -2, -5/3).
    regime: str
        Regime name shown above e.g., ['nonlocal', '-5/3', ...]
    
    Returns
    -------
    PDF: xr.DataArray
        Analytic predictions of PDF.
    """
    ts = t
    r0 = params['r0']
    
    if regime.lower() in ['lundgren', 'nonlocal', '-3']:
        TL  = params['T']
        t_T = ts / TL
        PDF = 1.0 / (4.0 * np.pi**1.5 * t_T**0.5 * r0**2) * np.exp(
            - (np.log(r/r0) + 2.0 * t_T)**2.0 / (4.0 * t_T))
        
    elif regime.lower() in ['-2']:
        lda = params['lambda']
        R34 = 3.0 / 4.0
        R14 = 1.0 / 4.0
        tmp = lda * ts
        # note: ive(a) = iv(a) * exp(-a), use ive to avoid overflow
        PDF = 1.0 / (np.pi * tmp * (r0 * r)**R34)\
            * ive(3, 8.0 * (r0 * r)**R14 / tmp)\
            * np.exp(-(4.0 * (r0**R14 - r**R14)**2) / tmp)
        
    elif regime.lower() in ['-2-a']:
        lda = params['lambda']
        tmp = lda * ts
        PDF = 1.0 / (4.0 * np.pi * 6.0 * (tmp / 4.0)**4.0)\
            * np.exp(-(4.0 * r**0.5) / tmp)
        
    elif regime.lower() in ['richardson', 'local', '-5/3']:
        beta = params['beta']
        R23 = 2.0 / 3.0
        R13 = 1.0 / 3.0
        tmp = 4.0 * beta * ts
        # note: ive(a) = iv(a) * exp(-a), use ive to avoid overflow
        PDF = 3.0 / (np.pi * tmp * (r0 * r)**R23)\
            * ive(2, 9.0 * (r0 * r)**R13 / tmp * 2.0)\
            * np.exp(-(9.0 * (r0**R13 - r**R13)**2.0) / tmp)
        
    elif regime.lower() in ['richardson-a', 'local-a', '-5/3-a']:
        beta = params['beta']
        tmp  = beta * ts
        PDF  = 1.5**5.0 / (4.0 * np.pi * tmp**3.0)\
             * np.exp(-(9.0 * r**(2.0/3.0)) / (4.0 * tmp))
        
    elif regime.lower() in ['generalized']:
        lda = params['lambda']
        slp = params['slope']
        aa  = slp + 1.0
        bb  = 3.0 - slp
        tmp = lda * ts
        # note: ive(a) = iv(a) * exp(-a), use ive to avoid overflow
        PDF = (4.0/bb) / (4.0 * np.pi * tmp * (r0 * r)**(aa/4.0))\
            * ive(aa/bb, (8.0 / bb**2.0) * (r0 * r)**(bb/4.0) / tmp)\
            * np.exp(-((4.0/bb**2.0) * (r0**(bb/4.0) - r**(bb/4.0))**2) / tmp)
        
    elif regime.lower() in ['generalized-a']:
        lda = params['lambda']
        slp = params['slope']
        bb  = 3.0 - slp
        tmp = lda * ts
        PDF = bb / (4.0 * np.pi * gamma(4.0/bb) * (bb**2.0 * tmp / 4.0)**(4.0/bb))\
            * np.exp(-(4.0/bb**2.0) * r**(bb/2) / tmp)
            
    elif regime.lower() in ['rayleigh', 'diffusive']:
        k2  = params['k2']
        tmp = k2 * ts
        PDF = 1.0 / (4.0 * np.pi * tmp) * np.exp(
                - (r0**2.0 + r**2.0) / (4.0 * tmp)
              ) * i0(r0 * r / (2.0 * tmp))
            
    elif regime.lower() in ['rayleigh-a', 'diffusive-a']:
        k2  = params['k2']
        tmp = 4.0 * k2 * ts
        PDF = 1.0 / (np.pi * tmp) * np.exp(- r**2.0 / tmp)
        
    else:
        raise Exception(f'invalid regime {regime}, '+\
                        'should be [Lundgren, Richardson, Rayleigh], '+\
                        'or [local, nonlocal, -3, -2, -5/3]')
    
    return PDF.rename('PDF_' + regime[:2])


def ana_CIST(
    r: xr.DataArray,
    alpha: float,
    params: Dict[str, str],
    regime: str
) -> xr.DataArray:
    """Calculate analytic CIST of pair separation
    
    Four regimes are available:
      * -3   | nonlocal | Lundgren   regime -> CIST(r) ~ r^(0)
      * -2   | local    | Richardson regime -> CIST(r) ~ r^(-1/2)
      * -5/3 | local    | GM         regime -> CIST(r) ~ r^(-2/3)
      * diffusive       | rayleigh   regime -> CIST(r) ~ r^(-2)
    
    Parameters
    ----------
    r: xr.DataArray
        A given separation.
    params: dict
        Parameters used in the expression of each regime.
        Should be ['T', 'beta', 'lambda', 'k2']
    regime: str
        Regime name shown above e.g., ['nonlocal', '-5/3', ...]
    
    Returns
    -------
    CIST: xr.DataArray
        Analytic prediction of CIST.
    """
    al = alpha
    
    if al <= 1:
        raise Exception(f'alpha ({al}) should be larger than 1')
    
    if regime.lower() in ['lundgren', 'nonlocal', '-3']:
        TL = params['T']
        CIST = (r - r) + 2.0 / TL / np.log(al)
        
    elif regime.lower() in ['-2-a']:
        lda  = params['lambda']
        CIST = lda * gammaincinv(4, 0.5) / (4.0 * (al**0.5 - 1.0)) * r**-0.5
        
    elif regime.lower() in ['richardson-a', 'local-a', '-5/3-a']:
        beta = params['beta']
        R23  = 2.0 / 3.0
        CIST = 4.0 * beta * gammaincinv(3, 0.5) / (9.0 * (al**R23 - 1.0)) * r**-R23
        
    elif regime.lower() in ['rayleigh-a', 'diffusive-a']:
        k2 = params['k2']
        CIST = 4.0 * k2 * np.log(2.0) / (al**2.0 - 1.0) * r**-2.0
        
    else:
        raise Exception(f'invalid regime {regime}, '+\
                        'should be [Lundgren, Richardson-a, Rayleigh-a], '+\
                        'or [local-a, nonlocal, -3, -2, -5/3], '+\
                        'or there are no available regime')
    
    return CIST.rename('CIST_' + regime[:2])


"""
Since the full solutions of some measures are too complex to write out,
one may get the full predictions by numerical integration of the full
expressions of the p.d.f., using the methods below.
"""

def num_r2(
    PDF: xr.DataArray,
    scaled2PiR: bool = False
) -> xr.DataArray:
    """Calculate relative dispersion numerically from PDF
    
    Parameters
    ----------
    PDF: xr.DataArray
        A given PDF for a specific regime
    scaled2PiR: boolean
        Whether the PDF has been multiplied by 2*Pi*r
    
    Returns
    -------
    r2: xr.DataArray
        analytic prediction of relative dispersion.
    """
    values = PDF['rbin'].diff('rbin').values
    values = np.insert(values, 0, values[0])
    bin_width = xr.DataArray(values, dims='rbin', coords={'rbin':PDF['rbin'].values})
    
    if scaled2PiR:
        r2 = (PDF * bin_width * PDF.rbin**2).sum('rbin')
    else:
        r2 = (2 * np.pi * PDF * bin_width * PDF.rbin**3).sum('rbin')
    
    return r2.rename('r2')

def num_Ku(
    PDF: xr.DataArray,
    scaled2PiR: bool = False
) -> xr.DataArray:
    """Calculate kurtosis numerically from PDF
    
    Parameters
    ----------
    PDF: xr.DataArray
        A given PDF for a specific regime
    scaled2PiR: boolean
        Whether the PDF has been multiplied by 2*Pi*r
    
    Returns
    -------
    Ku: xr.DataArray
        analytic prediction of Kurtosis.
    """
    values = PDF['rbin'].diff('rbin').values
    values = np.insert(values, 0, values[0])
    bin_width = xr.DataArray(values, dims='rbin', coords={'rbin':PDF['rbin'].values})
    
    if scaled2PiR:
        r4 = (PDF * bin_width * PDF.rbin**4).sum('rbin')
        r2 = (PDF * bin_width * PDF.rbin**2).sum('rbin')
    else:
        r4 = (2 * np.pi * PDF * bin_width * PDF.rbin**5).sum('rbin')
        r2 = (2 * np.pi * PDF * bin_width * PDF.rbin**3).sum('rbin')
    
    Ku = r4 / (r2)**2
    
    return Ku.rename('Ku')

def num_K2(
    PDF: xr.DataArray,
    scaled2PiR: bool = False
) -> xr.DataArray:
    """Calculate diffusivity numerically from PDF
    
    Parameters
    ----------
    PDF: xr.DataArray
        A given PDF for a specific regime
    scaled2PiR: boolean
        Whether the PDF has been multiplied by 2*Pi*r
    
    Returns
    -------
    Ku: xr.DataArray
        analytic prediction of relative diffusivity.
    """
    values = PDF['rbin'].diff('rbin').values
    values = np.insert(values, 0, values[0])
    bin_width = xr.DataArray(values, dims='rbin', coords={'rbin':PDF['rbin'].values})
    
    if scaled2PiR:
        r2 = (PDF * bin_width * PDF.rbin**2).sum('rbin')
        K2 = r2.differentiate('time') / 2.0
        
        K2 = K2.rename({'time':'rbin'})
        K2['rbin'] = np.sqrt(r2.values)
    else:
        r2 = (2 * np.pi * PDF * bin_width * PDF.rbin**3).sum('rbin')
        K2 = r2.differentiate('time') / 2.0
        
        K2 = K2.rename({'time':'rbin'})
        K2['rbin'] = np.sqrt(r2.values)
    
    return K2.rename('K2')

def num_CIST(
    PDF: xr.DataArray,
    lower: Optional[int] = 0.1,
    upper: Optional[int] = 0.9,
    maskout: Optional[List[float]] = [1e-6, 1e2],
    scaled2PiR: Optional[bool] = False
) -> xr.DataArray:
    """Calculate CIST numerically from PDF
    
    Parameters
    ----------
    PDF: xr.DataArray
        A given PDF for a specific regime
    scaled2PiR: boolean
        Whether the PDF has been multiplied by 2*Pi*r
    
    Returns
    -------
    cist: xr.DataArray
        Analytic prediction of CIST.
    """
    if scaled2PiR:
        CDF = cumul_dens_func(PDF)
    else:
        CDF = cumul_dens_func(PDF * 2 * np.pi * PDF.rbin)
    
    cist = ana_CIST(CDF, lower, upper, maskout=maskout)
    
    return cist.rename('cist')

