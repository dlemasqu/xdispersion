# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from xdispersion.core import RelativeDispersion

def test_pair_info():
    def load_GLAD_drifters(dset, check_interval=True):
        ds = dset.drop_vars(['ID','rowsize'])
        
        drifters = []
        ids = []
        
        for ID, dr in ds.groupby('ids'):
            dr.attrs['ID'] = ID
            drifters.append(dr.swap_dims({'obs':'time'}).drop_vars('ids'))
        
        if check_interval: # check the time interval is 0.25 hour
            for dr in drifters:
                tt = dr.time.values
                if not ((tt[1:] - tt[:-1]) / np.timedelta64(1, 'h') == 0.25).all():
                    ids.append(dr.attrs['ID'])
    
        print(f'there are {len(drifters)} drifters in the dataset')
        
        return drifters, ids, dset
    
    drifters, ids, dset = load_GLAD_drifters(xr.open_dataset('./data/glad64.nc'))

    rd = RelativeDispersion(xpos='longitude', uvel='ve', time='time', Rearth=6371.2,
                            ypos='latitude' , vvel='vn', coord='latlon')
    
    # first way of getting information
    p_all  = rd.get_all_pairs_information(dset)
    cond   = np.logical_and(p_all.r0>0.08, p_all.r0<0.18)
    p_ori1 = p_all.where(cond).dropna('pair', how='all')
    
    # second way of getting information
    pairs  = rd.group_pairs(drifters)
    origin = rd.find_pairs(pairs, [0.08,  0.18], chancePair=False)
    p_ori2 = rd.get_pairs_information(origin)
    
    assert len(p_all['pair']) == 43518
    assert len(p_ori1['pair']) == 170
    assert len(p_ori2['pair']) == 170
    assert (p_ori1.tlen == p_ori2.tlen).all()
    assert (p_ori1.r0   == p_ori2.r0  ).all()
    assert (p_ori1.lon0 == p_ori2.lon0).all()
    assert (p_ori1.lat0 == p_ori2.lat0).all()


