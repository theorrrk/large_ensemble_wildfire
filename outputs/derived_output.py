import xarray as xr
import pandas as pd
import numpy as np
import glob
import time
import sys
import os
from zipfile import ZipFile
import time


def gather_fire_seasons(seasonal_threshold, index, period = 'historical'):

    if period == 'historical':
        mode = 'h'
    if period == 'projected':
        mode = 's'

    t0 = time.time()

    mask = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                           'wildfires_theo_keeping/live/ensemble_data/'+
                           'mask_conus_2000_2009.nc').mask.to_numpy()

    paths = glob.glob('/rds/general/user/tk22/ephemeral/tmp/'+
                      f'p_????_{period}_annual_fires_705.nc')

    members = []
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(f'{i} of {len(paths)}\t({(time.time() - t0)/60:.2f} minutes)')
            sys.stdout.flush()
        ds = xr.open_dataset(path).load()
        members.append(ds.member.to_numpy()[0])
        temp = (ds.p > seasonal_threshold)
        temp = temp.resample(time = '1Y').sum() * mask

        if i == 0:
            lat = temp.latitude.to_numpy()
            lon = temp.longitude.to_numpy()
            yrs = np.arange(int(temp.time.dt.year[0]),int(temp.time.dt.year[-1])+1)
            fire_years = temp
        else:
            fire_years = np.concatenate((fire_years, temp), axis = 0)
        ds.close()
        temp.close()

    fire_years = xr.Dataset(data_vars = {'season_length': (
        ['member','time','latitude','longitude'], fire_years)},
                              coords = {'member': members,
                                        'time': yrs,
                                        'latitude': lat,
                                        'longitude': lon})
    return fire_years


def runner_h():
    period = 'h'
    index = 705

    paths = glob.glob('/rds/general/user/tk22/ephemeral/tmp/'+
                      f'p_{period}???_*_annual_fires_{index}.nc')
    print(f'{len(paths)} paths found.')

    ds = []
    for i,path in enumerate(sorted(paths)):
        if i%5 == 0:
            print(path)
        da = xr.open_dataset(path).p
        da = da.resample(time = 'Y').sum().load().to_dataset()
        ds.append(da)
    ds = xr.concat(ds, dim = 'member')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                     f'downscaled/annual_fires_{index}_historical.nc')
    ds.close()
    da.close()
    
    index = 705
    ds_re = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                            'wildfires_theo_keeping/live/ensemble_summaries/'+
                            f'p_class_B_19900101_20191231_obs_{index}.nc')
    mask = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                           'wildfires_theo_keeping/live/ensemble_data/'+
                           'mask_conus_2000_2009.nc').mask.to_numpy()
    # Threshold, half of the maximum fire week:
    seasonal_threshold = (ds_re.p.rolling(time = 7).mean().max(dim = 'time')/2).to_numpy()
    # Defining season length for reanalysis:
    length_re = (ds_re.p > seasonal_threshold)
    length_re = length_re.resample(time = '1Y').sum() * mask
    length_re = length_re.to_dataset().rename({'p':'season_length'})
    length_re.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                        'downscaled/season_length_reanalysis.nc')
    ds_hist = gather_fire_seasons(seasonal_threshold, index, period = 'historical')
    ds_hist.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                      'downscaled/season_length_historical.nc')
    return


def runner_s():
    period = 's'
    index = 705

    paths = glob.glob('/rds/general/user/tk22/ephemeral/tmp/'+
                      f'p_{period}???_*_annual_fires_{index}.nc')
    print(f'{len(paths)} paths found.')

    ds = []
    for i,path in enumerate(sorted(paths)):
        if i%5 == 0:
            print(path)
        da = xr.open_dataset(path).p
        da = da.resample(time = 'Y').sum().load().to_dataset()
        ds.append(da)
    ds = xr.concat(ds, dim = 'member')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                     f'downscaled/annual_fires_{index}_projected.nc')
    ds.close()
    da.close()
    index = 705
    ds_re = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                            'wildfires_theo_keeping/live/ensemble_summaries/'+
                            f'p_class_B_19900101_20191231_obs_{index}.nc')
    mask = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                           'wildfires_theo_keeping/live/ensemble_data/'+
                           'mask_conus_2000_2009.nc').mask.to_numpy()
    # Threshold, half of the maximum fire week:
    seasonal_threshold = (ds_re.p.rolling(time = 7).mean().max(dim = 'time')/2).to_numpy()
    # Defining season length for reanalysis:
    length_re = (ds_re.p > seasonal_threshold)
    length_re = length_re.resample(time = '1Y').sum() * mask
    length_re = length_re.to_dataset().rename({'p':'season_length'})
    length_re.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                        'downscaled/season_length_reanalysis.nc')
    ds_proj = gather_fire_seasons(seasonal_threshold, index, period = 'projected')
    ds_proj.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                      'downscaled/season_length_projected.nc')
    return


def gather(period = 'hist'):
    if period == 'hist':
        ds = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/tmp/'+
                               'p_h???_historical_annual_fires_705.nc')
        ds['p'] = ds['p'].astype('float32').load()
        paths = glob.glob('/rds/general/user/tk22/ephemeral/tmp/'+
                          'p_h???_historical_annual_fires_705.nc')
        for path in paths:
            os.remove(path)
        ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                     'predictors/downscaled/p_historical.nc')
    if period == 'proj':
        ds = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/tmp/'+
                               'p_s???_projected_annual_fires_705.nc')
        ds['p'] = ds['p'].astype('float32').load()
        paths = glob.glob('/rds/general/user/tk22/ephemeral/tmp/'+
                          'p_s???_projected_annual_fires_705.nc')
        for path in paths:
            os.remove(path)
        ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                     'predictors/downscaled/p_projected.nc')
    return
    


if __name__ == '__main__':
    if sys.argv[1] == 'hist':
        runner_h()
    if sys.argv[1] == 'proj':
        runner_s()
    if sys.argv[1] == 'gather':
        gather(period = sys.argv[2])