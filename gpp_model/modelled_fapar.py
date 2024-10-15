import xarray as xr
import pandas as pd
import numpy as np
from scipy.special import lambertw
import sys
import os
from matplotlib import pyplot as plt
from scipy.ndimage import convolve1d


def get_co2_pressure(member = 'h010', period = 'historical'):
    da = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                         f'bias_corrected/ps_{period}_'+
                         f'{member}_daily.nc')['ps'][0,:,:,:]
    
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
    co2 = da2 * da / 1e6
    return co2


def annual_fapar_max(member = 'h010', period = 'historical'):
    k = 0.5 # unitless
    Z = 13.86 # mol / m2 / yr
    f0 = 0.62 # unitless

    tas = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                          f'bias_corrected/tas_{period}_'+
                          f'{member}_daily.nc')['tas'][0,:,:,:] - 273.15 # celcius
    chi = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                          f'bias_corrected/chi_{period}_'+
                          f'{member}_daily.nc')['chi'][0,:,:,:] # 0-1
    gpp = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                          f'bias_corrected/gpp_potential_{period}_'+
                          f'{member}_daily.nc')['gpp'][0,:,:,:] * (24*60*60)/1e6 # mol/m2/day
    vpd = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                          f'bias_corrected/vpd_{period}_'+
                          f'{member}_daily.nc')['vpd'][0,:,:,:] # Pa
    pr = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                         f'bias_corrected/pr_{period}_'+
                         f'{member}_daily.nc')['pr'][0,:,:,:] * 1e3 / 0.018015 # mol/m2/day

    chi = xr.where(tas > 0, chi, np.nan).resample(time = 'Y').mean() # 0-1
    A0 = gpp.resample(time = 'Y').sum() # mol/m2/yr
    P = pr.resample(time = 'Y').sum() # mol/m2/yr
    D = xr.where(tas > 0, vpd, np.nan).resample(time = 'Y').mean() # Pa
    Ca = get_co2_pressure(member = member, period = period).resample(time = 'Y').mean()

    limit_1 = (1 - Z / (k*A0)).clip(min = 0, max = 1)
    limit_2 = (Ca * f0 * P * (1 - chi) / (1.6 * D * A0)).clip(min = 0, max = 1)

    fapar_max = np.stack([limit_1, limit_2])
    fapar_max = np.min(fapar_max, axis = 0)
    fapar_max = xr.Dataset(data_vars = {'fAPAR_max': (['time','latitude','longitude'],
                                                      fapar_max)},
                           coords = {'time': limit_1.time.to_numpy(),
                                     'latitude': limit_1.latitude.to_numpy(),
                                     'longitude': limit_1.longitude.to_numpy()})
    fapar_max = fapar_max.clip(min = 0)
    return fapar_max['fAPAR_max']


def get_lai_max(fapar_max):
    lai_max = (- 2 * np.log(1 - fapar_max)).to_dataset()
    lai_max = lai_max.rename({'fAPAR_max':'LAI_max'})
    lai_max = lai_max.clip(min = 0)
    return lai_max['LAI_max']



def final_fapar_output(member = 'h010', period = 'historical'):
    fapar_max = annual_fapar_max(member = member, period = period)
    fapar_max.to_netcdf(f'/rds/general/user/tk22/ephemeral/'+
                        f'gpp_debug/fapar_max_{period}_{mem}_daily.nc')
    lai_max = get_lai_max(fapar_max)
    fapar_max.to_netcdf(f'/rds/general/user/tk22/ephemeral/'+
                        f'gpp_debug/lai_max_{period}_{mem}_daily.nc')
    
    if period == 'historical':
        syr = '2000'
    elif period == 'projected':
        syr = '2075'
        
    lai_max['time'] = pd.date_range(start = f'{syr}-01-01T00:00:00.000000000',
                                    periods = 10, freq = 'YS') + pd.Timedelta(182, "d")
    
    sigma = 0.767
    tas = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                          f'bias_corrected/tas_{period}_'+
                          f'{member}_daily.nc')['tas'][0,:,:,:] - 273.15 # celcius
    gpp = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/'+
                          f'bias_corrected/gpp_potential_{period}_'+
                          f'{member}_daily.nc')['gpp'][0,:,:,:] * (24*60*60)/1e6 # mol/m2/day
    gsl =  xr.where(tas > 0, 1, 0).resample(time = 'Y').sum()
    A0 = gpp.resample(time = 'Y').sum()
    
    gsl['time'] = lai_max.time.to_numpy()
    A0['time'] = lai_max.time.to_numpy()
    fapar_max['time'] = lai_max.time.to_numpy()
    
    m = (sigma * lai_max * gsl) / (A0 * fapar_max)
    u = []
    for yr in m.time.dt.year.to_numpy():
        u.append(m.sel(time = (m.time.dt.year == yr))[0,:,:].to_numpy() * 
                 gpp.sel(time = (gpp.time.dt.year == yr)))
    u = xr.concat(u, dim = 'time').to_numpy()
    

    lai_max = lai_max.interp(time = gpp.time, method = 'nearest',
                             kwargs = {'fill_value': 'extrapolate'})
    lai_max = lai_max.to_numpy()
    
    lambert_input = -0.5 * u * np.exp (-0.5 * u)
    lambert_output = np.real(lambertw(lambert_input))
    lai_steady = u + 2 * lambert_output
    
    lai_steady[lai_steady >= lai_max] = lai_max[lai_steady >= lai_max]
    lai_steady[lai_steady < 0] = 0 # <- BOYA SETS TO NAN, but min > -1e5
    
    lai = convolve1d(lai_steady, np.ones(12), axis = 0, mode = 'wrap', origin = -6)
    fapar = 1 - np.exp(-0.5*lai)
    
    fapar = np.nan_to_num(fapar, nan = 0, posinf = 0, neginf = 0) # Setting NaNs to 0!
    
    fapar = xr.Dataset(data_vars = {'fAPAR': (['member','time','latitude','longitude'],
                                              fapar[np.newaxis,:,:,:])},
                       coords = {'member': [member],
                                 'time': gpp.time.to_numpy(),
                                 'latitude': gpp.latitude.to_numpy(),
                                 'longitude': gpp.longitude.to_numpy()})
    return fapar


if __name__ == '__main__':
    if sys.argv[1] == 'hist':
        members = ['h'+str(x).zfill(3) for x in range(10,170)]
        period = 'historical'
    if sys.argv[1] == 'proj':
        members = ['s'+str(x).zfill(3) for x in range(10,170)]
        period = 'projected'
    
    i = int(os.getenv('PBS_ARRAY_INDEX'))
    mem = members[i]
    print(mem)
    sys.stdout.flush()
    da = final_fapar_output(member = mem, period = period)
    da.to_netcdf('/rds/general/user/tk22/ephemeral/gpp_outputs/'+
                 f'fAPAR_{period}_{mem}_daily.nc')
    



