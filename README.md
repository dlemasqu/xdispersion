# xdispersion
An illustrative figure should be put here.


## 1. Introduction
The package is designed for analyzing relative dispersion behaviors of Lagrangian particle pairs (two-particle) in a statistical fashion.  It is based on the data structure of [`xarray`](https://xarray.dev/) and hence [`dask`](), so that one can leverage the power of multi-dimensional labelled arrays and out-of-core calculations.

## 2. How to do it nicely and efficiently
For Lagrangian data represented in multi-dimensional `xarray.DataArray`, one of the major problems is that the lengths of Lagrangian trajectories are not the same.  So it is impossible to use a `traj` dimension to represent different trajectories.  This is solved by introducing the [ragged array](https://clouddrift.org/) based on [Awkward Array](https://awkward-array.org/): All the trajectories are connected head to tail, forming an `obs` dimension.  This is memory and storage efficient, but requires extra efforts to retrieve each trajectories.

The second problem is how to represent the dataset of particle pairs.  Following the [ragged array](https://clouddrift.org/), one could still connected pairs of different lengths head to tail to form an `obs` dimension.  However, this is not storage efficient.  For example, if there are four [$n$] trajectories T1, T2, T3, and T4, through combination one would get six [$n\times(n-1)/2$] pairs:
- (T1, T2)
- (T1, T3)
- (T1, T4)
- (T2, T3)
- (T2, T4)
- (T3, T4)

In such a pair dataset, we need to replicate each trajectory three [$n-1$] times.  For a large number of particles, the size of the pair dataset will be generally unacceptable.

The solution here is to store the basic information of particle pairs `pinfo` (like initial locations, initial separation, initial time, and trajectory IDs), instead of duplicating all the data, and then do the analyzing using both `pinfo` and ragged-trajectory dataset:
```python
rd = cal_relative_dispersion(pinfo, trajs)
```

This gives a `DataArray` of `rd` which has two dimensions: `['pair', 'time']`.  Since the lengths of different pairs are not equal, one need to specify a maximum time of analysis to truncate or pad-with-nan along the time dimension.  Also, one can perform an estimate of the errorbar through bootstrapping along the `pair` dimension. 

Such a design allows one to filter the pair information `pinfo` and select those pairs based on specific criterions (e.g., initial separation is within a given `r0`):
```python
pinfo_original = sel_original_pairs(pinfo, r0)
```

Also, one can easily do some statistics and plots of the selected pairs:
```python
plot_map(pinfo)
histogram(pinfo)
```


---
## 2. How to install
**Requirements**
`xdispersion` is developed under the environment with `xarray` (=version 0.15.0), `dask` (=version 2.11.0), `numpy` (=version 1.15.4), `cartopy` (=version 0.17.0).  Older versions of these packages are not well tested.

**Install via pip**
```
pip install xdispersion # not yet
```

**Install via conda**
```
conda install -c conda-forge xdispersion # not yet
```

**Install from github**
```
git clone https://github.com/miniufo/xdispersion.git
cd xdispersion
python setup.py install
```


---
## 3. Examples


