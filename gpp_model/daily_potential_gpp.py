import xarray as xr
import numpy as np
import pandas as pd
import sys
import time
import random
import glob
import os
from pyrealm import pmodel
from matplotlib import pyplot as plt


def load_inputs(period = 'historical', mem_ind = 0):
    if period == 'historical':
        members = ['h'+str(x).zfill(3) for x in range(10,170)]
    if period == 'projected':
        members = ['s'+str(x).zfill(3) for x in range(10,170)]
    mem = members[mem_ind]
    
    da = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'tas_{period}_{mem}_daily.nc')['tas'][0,:,:,:]
    temp = da.to_numpy() - 273.15
    temp = temp.clip(min = -25, max = 80)
    
    da = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'vpd_{period}_{mem}_daily.nc')['vpd'][0,:,:,:]
    vpd = da.to_numpy()
    vpd = vpd.clip(min = 0, max = 1e4)
    
    da = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'ps_{period}_{mem}_daily.nc')['ps'][0,:,:,:]
    patm = da.to_numpy()
    
    da = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'ppfd_{period}_{mem}_daily.nc')['ppfd'][0,:,:,:]
    ppfd = da.to_numpy() * 1e6 / (60*60*24)
    
    da2 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                          f'rel_soilm_{period}_monthly.nc')['rel_soilm'][mem_ind,:,:,:].load()
    da2 = da2.interp(time = da.time, method = 'linear',
                     kwargs = {'fill_value': 'extrapolate'})
    da2 = da2.clip(min = 0, max = 1)
    rel_soilm = da2.to_numpy()
    
    da2 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                          f'alpha_{period}_monthly.nc')['alpha'][mem_ind,:,:,:]
    da2 = da2.interp(time = da.time, method = 'linear',
                     kwargs = {'fill_value': 'extrapolate'})
    da2 = da2.clip(min = 0, max = 1)
    alpha = da2.to_numpy()
    
    if period == 'historical':
        da2 = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                             'wildfires_theo_keeping/live/gpp_ensemble/'+
                             'co2_ppm_200001_200912_CMIP6.nc')['co2_ppm']
    elif period == 'projected':
        da2 = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                             'wildfires_theo_keeping/live/gpp_ensemble/'+
                             'co2_ppm_207501_208412_CMIP6.nc')['co2_ppm']
    da2 = da2.interp(latitude = da.latitude.to_numpy(),
                     longitude = da.longitude.to_numpy())
    da2 = da2.interp(time = da.time, method = 'linear',
                     kwargs = {'fill_value': 'extrapolate'})
    co2 = da2.to_numpy()
    
    del da, da2

    return temp, vpd, co2, patm, rel_soilm, alpha, ppfd


def gpp_stocker(temp, vpd, co2, patm, rel_soilm, alpha, fapar, ppfd):
    env = pmodel.PModelEnvironment(tc   = temp,
                                   vpd  = vpd,
                                   co2  = co2,
                                   patm = patm)
    sm_stress_stocker = pmodel.calc_soilmstress_stocker(soilm = rel_soilm, meanalpha = alpha)
    mod_c3 = pmodel.PModel(env)
    mod_c4 = pmodel.PModel(env,
                           method_optchi = 'c4',
                           method_jmaxlim = 'simple')
    mod_c3.estimate_productivity(fapar = fapar,
                                 ppfd  = ppfd)
    mod_c4.estimate_productivity(fapar = fapar,
                                 ppfd  = ppfd)
    unstressed_gpp = mod_c3.gpp + mod_c4.gpp
    gpp = (mod_c3.gpp + mod_c4.gpp)*sm_stress_stocker
    chi = pmodel.CalcOptimalChi(env = env).chi
    return gpp, chi


def optimal_gpp_output(mem_ind, period = 'historical'):
    if period == 'historical':
        members = ['h'+str(x).zfill(3) for x in range(10,170)]
    if period == 'projected':
        members = ['s'+str(x).zfill(3) for x in range(10,170)]
    mem = members[mem_ind]
    
    t0 = time.time()
    
    (temp, vpd, co2, patm, 
     rel_soilm, alpha, ppfd) = load_inputs(period = 'historical',
                                           mem_ind = mem_ind)
    
    print(f'Inputs Loaded: {(time.time() - t0)/3600:.2f} hours')
    sys.stdout.flush()
    
    fapar = np.ones_like(temp)
    
    gpp = np.zeros_like(temp)
    chi = np.zeros_like(temp)
    for i in range(gpp.shape[2]):
        if i%100 == 0:
            print(f'\t{i} of {gpp.shape[2]} found')
            sys.stdout.flush()
        gpp_temp, chi_temp = gpp_stocker(temp[:,:,i], vpd[:,:,i],
                                         co2[:,:,i], patm[:,:,i],
                                         rel_soilm[:,:,i], alpha[:,:,i],
                                         fapar[:,:,i], ppfd[:,:,i])
        gpp[:,:,i] = gpp_temp
        chi[:,:,i] = chi_temp
    
    print(f'GPP Found: {(time.time() - t0)/3600:.2f} hours')
    sys.stdout.flush()
    
    grid = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                           f'tas_{period}_{mem}_daily.nc')

    ds = xr.Dataset(data_vars = {'gpp': (['member','time','latitude','longitude'], gpp[np.newaxis,:,:,:])},
                    coords = {'member': grid.member.to_numpy(),
                              'time': grid.time.to_numpy(),
                              'latitude': grid.latitude.to_numpy(),
                              'longitude': grid.longitude.to_numpy()})
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                 f'gpp_potential_{period}_{mem}_daily.nc')
    
    ds = xr.Dataset(data_vars = {'chi': (['member','time','latitude','longitude'], chi[np.newaxis,:,:,:])},
                    coords = {'member': grid.member.to_numpy(),
                              'time': grid.time.to_numpy(),
                              'latitude': grid.latitude.to_numpy(),
                              'longitude': grid.longitude.to_numpy()})
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                 f'chi_potential_{period}_{mem}_daily.nc')
    
    print(f'Data Saved: {(time.time() - t0)/3600:.2f} hours')
    sys.stdout.flush()
    return
    
    
if __name__ == '__main__':
    values = list(np.arange(160))
    random.shuffle(values)
    if sys.argv[1] == 'hist':
        members = ['h'+str(x).zfill(3) for x in range(10,170)]
        for i in values:
            if os.path.exists('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                              f'gpp_potential_historical_{members[i]}_daily.nc'):
                pass
            else:
                optimal_gpp_output(i, period = 'historical')
    if sys.argv[1] == 'proj':
        members = ['s'+str(x).zfill(3) for x in range(10,170)]
        for i in values:
            if os.path.exists('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                              f'gpp_potential_projected_{members[i]}_daily.nc'):
                pass
            else:
                optimal_gpp_output(i, period = 'projected')