import glob
import sys
import gc
import time
import json
import os

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_mask(ds_paths, pre_2002 = False):
    print('\nBuilding mask:')
    count = 0
    for path in ds_paths:
        ds = xr.open_dataset(path)
        ds = ds.sel(time = ds.time.dt.year <= 2020)
        ds = ds.sel(time = ds.time.dt.year >= 1992)
        count += 1
        var = [v for v in list(ds.variables) 
               if v not in ['latitude', 'longitude', 'time']][0]
        print('\t',var)
        sys.stdout.flush()
        if count == 1:
            mask = np.array(np.isnan(ds[var].to_numpy()), dtype = int)
        else:
            mask = mask + np.array(np.isnan(ds[var].to_numpy()), dtype = int)
        
        ds.close()
        del ds
    # Non-nan cells:
    mask = (mask == 0)
    if pre_2002 == False:
        # Clipping to not allow this time period:
        # (Due to increase in non-federal reporting)
        ds = xr.open_dataset(ds_paths[0])
        ds = ds.sel(time = ds.time.dt.year <= 2020)
        ds = ds.sel(time = ds.time.dt.year >= 1992)
        date_mask = ds.time.dt.year.to_numpy() >= 2002
        del ds
        mask = np.einsum('ijk,i->ijk',mask,date_mask)

    return mask


def build_inputs(N, start_run = 0, n_runs = 50):
    directory = ('/rds/general/user/tk22/projects/'+
                 'leverhulme_wildfires_theo_keeping/'+
                 'live/ensemble_data/')
    reanalysis_paths = glob.glob(directory + 'reanalysis_*_19900101_20211231.nc')
    landscape_paths = glob.glob(directory + 'landscape_*_19920101_20201231.nc')
    fire_path = glob.glob(directory + f'counts_B*_19920101_20201231.nc')
    size_path = glob.glob(directory + f'cell_area*_19920101_20201231.nc')

    ds_paths = reanalysis_paths + landscape_paths + fire_path + size_path

    mask = build_mask(ds_paths, pre_2002 = False)

    true_inds = np.where(mask)

    print('Checking all indices are True:')
    test = mask[true_inds]
    print(np.unique(test))
    sys.stdout.flush()
    
    for run in range(start_run, start_run + n_runs, 1):
        
        print(f'Run: {run}')
        sys.stdout.flush()

        indices = np.arange(len(true_inds[0]))
        indices = np.random.choice(indices, size = int(1.25*N), replace = False)

        sample_inds = (true_inds[0][indices],
                       true_inds[1][indices],
                       true_inds[2][indices])

        print('\tChecking all sample indices are True:')
        test = mask[sample_inds]
        print(np.unique(test))
        sys.stdout.flush()

        count = 0
        print('\n\tBuilding dataframe:')
        for path in ds_paths:
            count += 1
            ds = xr.open_dataset(path)
            ds = ds.sel(time = ds.time.dt.year <= 2020)
            ds = ds.sel(time = ds.time.dt.year >= 1992)
            var = [v for v in list(ds.variables) 
                   if v not in ['latitude', 'longitude', 'time']][0]
            ds = ds[var].to_numpy()[sample_inds]
            print('\t\t',var)
            if count == 1:
                df = pd.DataFrame({var: ds})
            else:
                df[var] = ds

        df = df.dropna() # as a precaution, should be none
        train, test = train_test_split(df, test_size = 0.2)
        
        train.to_csv((f'/rds/general/user/tk22/ephemeral/model_inputs/'+
                      f'genesis_inputs_train_{run}.csv'),
                     index = False, header = True)
        test.to_csv((f'/rds/general/user/tk22/ephemeral/model_inputs/'+
                     f'genesis_inputs_test_{run}.csv'),
                    index = False, header = True)
                      

    return


if __name__ == '__main__':
    segment = int(os.getenv('PBS_ARRAY_INDEX'))
    build_inputs(1e6, start_run = 10*segment, n_runs = 10)