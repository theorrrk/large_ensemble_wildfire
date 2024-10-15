from matplotlib import colors
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import statsmodels as sm
import os
import sys
import glob
import time
import json
import random
import warnings



def build_mask(ds_paths, variables, pre_2002 = False):
    print('\nBuilding mask:')
    count = 0
    for path in ds_paths:
        ds = xr.open_dataset(path)
        ds = ds.sel(time = ds.time.dt.year <= 2020)
        ds = ds.sel(time = ds.time.dt.year >= 1992)
        var = [v for v in list(ds.variables) 
               if v not in ['latitude', 'longitude', 'time']][0]
        if var in variables:
            count += 1
            print('\t',var)
            sys.stdout.flush()
            if count == 1:
                mask = np.array(np.isnan(ds[var].to_numpy()), dtype = int)
            else:
                mask = mask + np.array(np.isnan(ds[var].to_numpy()), dtype = int)
        else:
            pass        
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
        mask = np.einsum('ijk,i->ijk',mask,date_mask)
    return mask


def build_inputs(N, variables, fire_class = 'B'):

    directory = ('/rds/general/user/tk22/projects/'+
                 'leverhulme_wildfires_theo_keeping/'+
                 'live/ensemble_data/')
    reanalysis_paths = glob.glob(directory + 'reanalysis_*_ERA5_19900101_20211231.nc')
    landscape_paths  = glob.glob(directory + 'landscape_*_19920101_20201231.nc')
    reanalysis_paths = [r_path for r_path in reanalysis_paths if True in [v in r_path for v in variables]]
    landscape_paths  = [l_path for l_path in landscape_paths if True in [v in l_path for v in variables]]
    fire_path = glob.glob(directory + f'counts_{fire_class}*_19920101_20201231.nc')
    size_path = glob.glob(directory + f'cell_area*_19920101_20201231.nc')

    ds_paths = reanalysis_paths + landscape_paths + fire_path + size_path

    mask = build_mask(ds_paths, variables, pre_2002 = False)

    true_inds = np.where(mask)

    print('Checking all indices are True:')
    test = mask[true_inds]
    print(np.unique(test))
    sys.stdout.flush()

    indices = np.arange(len(true_inds[0]))
    indices = np.random.choice(indices, size = int(1.25*N), replace = False)

    sample_inds = (true_inds[0][indices],
                   true_inds[1][indices],
                   true_inds[2][indices])

    print('Checking all sample indices are True:')
    test = mask[sample_inds]
    print(np.unique(test))
    sys.stdout.flush()

    count = 0
    print('\nBuilding dataframe:')
    for path in ds_paths:
        ds = xr.open_dataset(path)
        var = [v for v in list(ds.variables) 
               if v not in ['latitude', 'longitude', 'time']][0]
        if var in variables + ['counts', 'cell_area']:
            count += 1
            ds = ds[var].to_numpy()[sample_inds]
            print('\t',var)
            if count == 1:
                df = pd.DataFrame({var: ds})
            else:
                df[var] = ds
        else:
            pass

    df = df.dropna() # as a precaution, should be none
    train, test = train_test_split(df, test_size = 0.2)
    print('Test and train built:')
    print(f'Test length: {len(test)}')
    print(f'Train length: {len(train)}')
    return train, test


def apply_thresh(test, train, thresh_row, predictor_list):
    temp_test = test.copy(deep = True)
    temp_train = train.copy(deep = True)
    # Clipping to thresholds:
    for var in predictor_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_train[var].loc[temp_train[var] < float(thresh_row.loc['lo_'+var])] = float(thresh_row.loc['lo_'+var])
            temp_train[var].loc[temp_train[var] > float(thresh_row.loc['hi_'+var])] = float(thresh_row.loc['hi_'+var])
            temp_test[var].loc[temp_test[var] < float(thresh_row.loc['lo_'+var])] = float(thresh_row.loc['lo_'+var])
            temp_test[var].loc[temp_test[var] > float(thresh_row.loc['hi_'+var])] = float(thresh_row.loc['hi_'+var])
    print("Thresholds applied")
    sys.stdout.flush()
    return temp_test, temp_train


def model_fit(test, train, formula, summary = False):
    # 2: Run a model and return: max_vif, aic and auc given a list of predictors
    #    Takes roughly 1.3 N microseconds, where N is the length of test and train.
    #    Therefore N = 10**7 makes a lot of sense at ~13 seconds per ru~, as does
    model = smf.glm(formula,
                    data = train,
                    offset = np.log(train['cell_area']),
                    family = sm.genmod.families.family.Binomial(
                    sm.genmod.families.links.Logit())).fit(disp = 0)
    if summary == True:
        print(model.summary())
        sys.stdout.flush()
    print('Model fitted')
    sys.stdout.flush()
    test['p'] = model.predict(exog = test)
    auc = roc_auc_score(test.sort_values(by = 'counts').counts,
                        test.sort_values(by = 'counts').p)
    aic = model.aic
    # Find variance-covariance matrix of the coefficients:
    cov = model.cov_params()
    # Find correlation matrix of the coefficients:
    corr = cov / model.bse / np.array(model.bse)[:,None]
    # Inversion of correlation matrix = partial correlations between coefs
    # Diagonal of that is the correlation of of variable with all others
    # Note that the intercept is excluded by [1:, 1:]
    max_vif = np.max(np.diag(np.linalg.inv(corr.values[1:, 1:])))
    print(f'AIC = {aic}')
    print(f'AUC = {auc}')
    print(f'Max VIF = {max_vif}')
    sys.stdout.flush()
    if summary == True:
        return model.params, aic, auc, max_vif
    return aic, auc, max_vif


def build_dataset(variables, thresh_row, fire_class = 'B', index = 0):
    formula = 'counts ~ ' + ' + '.join(variables)
    print(formula)
    sys.stdout.flush()
    # Building dataset:
    dataset_paths = []
    ds_dir = ('/rds/general/user/tk22/projects/leverhulme_'+
              'wildfires_theo_keeping/live/ensemble_data/')
    dataset_paths.append(glob.glob(ds_dir + f'counts_{fire_class}_*19920101_20201231.nc')[0])
    for var in variables+['cell_area']:
        new_paths = glob.glob(ds_dir + f'*{var}*_199?0101_202?1231.nc')
        new_path = min(new_paths, key=len)
        dataset_paths.append(new_path)       
    
    # Building test-train:
    train = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                         f'genesis_inputs_train_{index}.csv'))
    test = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                        f'genesis_inputs_test_{index}.csv'))
    print(f'Train dataframe:\n{train}\n\n')
    print(f'Test dataframe:\n{test}\n\n')
    
    # Applying thresholds:
    test, train = apply_thresh(test, train, thresh_row, variables)
    
    # Fitting:
    params, aic, auc, max_vif = model_fit(test, train, formula, summary = True)
    del test, train
    
    # Building input array:    
    temp = xr.open_dataset('/rds/general/user/tk22/projects/'+
                           'leverhulme_wildfires_theo_keeping/'+
                           f'live/ensemble_data/cell_area_'+
                           '6min_19920101_20201231.nc')
    temp = temp.sel(time = temp.time.dt.year <= 2020)
    temp = temp.sel(time = temp.time.dt.year >= 1992)
    logit_p = 0 * temp['cell_area'].to_numpy()
    time = temp.time.to_numpy()
    latitude = temp.latitude.to_numpy()
    longitude = temp.longitude.to_numpy()
    temp.close()
    del temp
    
    ### TRY REWRITING WITH A NUMPY ARRAY!
    for param in params.keys():
        if param == 'Intercept':
            logit_p = logit_p + params['Intercept']
        else:
            print(param)
            sys.stdout.flush()
            
            # Waiting 1 minute between each new dataset to reduce IO load.
            
            new_paths = glob.glob(ds_dir + f'*{param}*_199?0101_202?1231.nc')
            new_path = min(new_paths, key=len)
            
            temp = xr.open_dataset(new_path)[param].clip(min = float(thresh_row.loc['lo_' + param]),
                                                         max = float(thresh_row.loc['hi_' + param]))
            temp = temp.sel(time = temp.time.dt.year <= 2020)
            temp = temp.sel(time = temp.time.dt.year >= 1992)
            new_data = temp.to_numpy()
            logit_p = logit_p + params[param] * new_data
            temp.close()
            del temp, new_data
    
    temp = xr.open_dataset('/rds/general/user/tk22/projects/'+
                           'leverhulme_wildfires_theo_keeping/'+
                           'live/ensemble_data/cell_area_'+
                           '6min_19920101_20201231.nc')['cell_area']
    temp = temp.sel(time = temp.time.dt.year <= 2020)
    temp = temp.sel(time = temp.time.dt.year >= 1992)
    logit_p = logit_p + np.log(temp.to_numpy())
    temp.close()
    del temp
    p = np.exp(logit_p)/(1 + np.exp(logit_p))
    del logit_p
    da = xr.Dataset(data_vars = {'p': (['time','latitude','longitude'],p)},
                    coords = {'time': time,
                              'latitude': latitude,
                              'longitude': longitude})['p']
    del p

    return da, aic, auc, max_vif