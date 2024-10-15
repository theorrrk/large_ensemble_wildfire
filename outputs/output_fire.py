import xarray as xr
import statsmodels as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from matplotlib import colors
import os
import datetime
import sys
import gc
import glob
import time as time_lib
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def build_fine_output(params, variables, thresh_row, index,
                      mem_ind = 0, period = 'historical', a = 1, b = 1):
    temp = xr.open_dataset('/rds/general/user/tk22/ephemeral/'+
                           f'ensemble_predictors/downscaled/vpd_{period}.nc')
    
    logit_p = 0 * temp['vpd'][0,:,:,:].to_numpy()

    member = [temp['vpd'].member.to_numpy()[mem_ind]]
    time = temp.time.to_numpy()
    latitude = temp.latitude.to_numpy()
    longitude = temp.longitude.to_numpy()

    temp.close()
    del temp

    data_dir = '/rds/general/user/tk22/ephemeral/ensemble_predictors/downscaled/'
    obs_dir = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/')   

    for param in params.keys():
        print(param)
        sys.stdout.flush()
        if param == 'Intercept':
            logit_p = logit_p + params['Intercept']
        else:
            try:
                path = min(glob.glob(data_dir + f'*{param}*{period}*.nc'), key = len)
                print('Adding Ensemble Variable.')
                sys.stdout.flush()
                temp = xr.open_dataset(path)[param][mem_ind,:,:,:]
                temp = temp.clip(min = float(thresh_row.loc['lo_' + param]),
                                 max = float(thresh_row.loc['hi_' + param]))
                if period == 'historical':
                    temp = temp.sel(time = temp.time.dt.year >= 2000)
                    temp = temp.sel(time = temp.time.dt.year <= 2009)
                if period == 'projected':
                    temp = temp.sel(time = temp.time.dt.year >= 2075)
                    temp = temp.sel(time = temp.time.dt.year <= 2084)
                temp = temp.to_numpy()
                logit_p = logit_p + params[param] * temp
                del temp
            except:
                print('Adding Static Variable.')
                sys.stdout.flush()
                candidate_paths = glob.glob(obs_dir + f'*{param}*.nc')
                new_path = min(candidate_paths, key=len)
                temp = xr.open_dataset(new_path)[param].clip(min = float(thresh_row.loc['lo_' + param]),
                                                             max = float(thresh_row.loc['hi_' + param]))

                temp = temp.sel(time = temp.time.dt.year >= 2000)
                temp = temp.sel(time = temp.time.dt.year <= 2009)

                temp['time'] = time
                doy_avg = temp.groupby('time.dayofyear').mean(dim='time')

                for doy in np.unique(temp.time.dt.dayofyear):
                    update_slice = doy_avg.sel(
                        dayofyear = doy_avg.dayofyear == doy)[0,:,:]
                    temp.loc[{'time': temp.time.dt.dayofyear == doy}] = update_slice

                logit_p = logit_p + params[param] * temp.to_numpy()
                del temp
                
    # Adding cell_area:
    temp = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                           'wildfires_theo_keeping/live/ensemble_data/'+
                           'cell_area_6min_19920101_20201231.nc')['cell_area']
    temp = temp.sel(time = temp.time.dt.year >= 2000)
    temp = temp.sel(time = temp.time.dt.year <= 2009)
    logit_p = logit_p + np.log(temp.to_numpy())
    
    p = np.exp(logit_p)/(1 + np.exp(logit_p))
    p = a * p ** b
    del logit_p
    ds = xr.Dataset(data_vars = {'p': (['member','time','latitude','longitude'],
                                       p[np.newaxis,:,:,:])},
                    coords = {'member': member,
                              'time': time,
                              'latitude': latitude,
                              'longitude': longitude})
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/tmp/'+
                 f'p_{member[0]}_{period}_annual_fires_{index}.nc')
    return


if __name__ == '__main__':
    index = 705
    thresh_row = pd.read_csv('/rds/general/user/tk22/home/paper_2/model/'+
                             f'step_3/output/output_summary_{index}.csv',    
                             index_col = 0)
    variables = [x[3:] for x in thresh_row.index if x[:3] == 'lo_']
    # Run model with observational data to find params:
    params = pd.read_csv('/rds/general/user/tk22/home/'+
                         f'paper_2/output/params_{index}.csv',
                         index_col = 0)['0']
    # Build observed dataset:
    a = float(thresh_row['value'].a)
    b = float(thresh_row['value'].b)

    if sys.argv[1] == 'hist':
        for i in range(160):
            print(f'\nBuilding Historical Member {i+1} of 160:\n')
            sys.stdout.flush()
            member = 'h'+str(i+10).zfill(3)
            path = f'/rds/general/user/tk22/ephemeral/tmp/p_{member}_historical_annual_fires_705.nc'
            if os.path.exists(path):
                pass
            else:
                build_fine_output(params, variables, thresh_row, index,
                                  mem_ind = i, period = 'historical',
                                  a = a, b = b)
    if sys.argv[1] == 'proj':
        for i in range(160):
            print(f'\nBuilding Projected Member {i+1} of 160:\n')
            sys.stdout.flush()
            member = 's'+str(i+10).zfill(3)
            path = f'/rds/general/user/tk22/ephemeral/tmp/p_{member}_projected_annual_fires_705.nc'
            if os.path.exists(path):
                pass
            else:
                build_fine_output(params, variables, thresh_row, index,
                                  mem_ind = i, period = 'projected',
                                  a = a, b = b)
        