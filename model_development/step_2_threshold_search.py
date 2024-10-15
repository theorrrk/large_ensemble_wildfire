import glob
import sys
import gc
import time
import json
import warnings
import os

import xarray as xr
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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


def define_threshold_options(variables):
    print('\nBuilding threshold options dictionary:')
    threshs = {}
    for var in variables:
        da = xr.open_mfdataset('/rds/general/user/tk22/projects/leverhulme_'+
                               'wildfires_theo_keeping/live/ensemble_data/'+
                               f'*{var}*199?0101_202?1231.nc')[var].load()
        da = da.sel(time = da.time.dt.year <= 2020)
        da = da.sel(time = da.time.dt.year >= 1992)
        print(f'\t{var}')
        sys.stdout.flush()
        
        min0    = float(da.min().load())
        max0    = float(da.max().load())
        perc1   = float(da.quantile(0.0001).load())
        perc99  = float(da.quantile(0.9999).load())
        std0    = float(da.std().load())
        mean0   = float(da.mean().load())
        
        da.close()
        del da

        arr = list(np.flip(np.arange(mean0, perc1, -std0/4)))[:-1]
        arr = arr + list(np.arange(mean0, perc99, std0/4))
        arr = [min0, perc1] + arr
        arr = arr + [perc99, max0]
        # Removing steps if less than eighth of standard deviation:
        new_arr = []
        for i, element in enumerate(reversed(arr)):
            if i == 0:
                prior_appended_value = element
                new_arr.append(element)
            else:
                if (prior_appended_value - element) < (std0/8):
                    pass # As gap too small to add threshold
                else:
                    prior_appended_value = element
                    new_arr.append(element)
        if min0 not in new_arr:
            new_arr.append(min0) # So full span of variable
        new_arr.reverse()
        threshs[var] = new_arr
    print('\n')
    sys.stdout.flush()
    return threshs


def build_summary(threshs, ind_df, variables, aic, auc):
    # Definining summary dataframe:
    summary_df = pd.DataFrame(columns = (['lo_' + var for var in variables] + 
                                         ['hi_' + var for var in variables] + 
                                         ['AIC', 'AUC']))
    new_row = ([threshs[var][ind_df.loc['min_ind'][var]] for var in variables] + 
               [threshs[var][ind_df.loc['max_ind'][var]] for var in variables] +
               [aic, auc])
    summary_df.loc[len(summary_df)] = new_row
    return summary_df


def initial_threshold_indices(variables):
    ind_df = pd.DataFrame(columns = variables)
    ind_df.loc[len(ind_df)] = list(np.zeros(len(variables), dtype = int))
    ind_df.loc[len(ind_df)] = list(-np.ones(len(variables), dtype = int))
    ind_df['index'] = ['min_ind', 'max_ind']
    ind_df = ind_df.set_index('index')
    return ind_df


def run_model(test, train, predictor_list):
    # 2: Run a model and return: max_vif, aic and auc given a list of predictors
    #    Takes roughly 1.3 N microseconds, where N is the length of test and train.
    #    Therefore N = 10**7 makes a lot of sense at ~13 seconds per run
    if len(predictor_list) == 0:
        formula = 'counts ~ 1'
    else:
        formula = 'counts ~ ' + ' + '.join(predictor_list)
    
    #print(formula)
    sys.stdout.flush()
    
    model = smf.glm(formula,
                    data = train,
                    offset = np.log(train['cell_area']),
                    family = sm.genmod.families.family.Binomial(
                    sm.genmod.families.links.Logit())).fit(disp = 0)    
    
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
    return float(aic), float(auc)


def tightening_multistep(var, variables, ind_df, threshs,
                         test, train, direction = 'max', steps = 1):
    # Checking if reduction can be made to thresholds:
    ok_steps = (len(threshs[var]) - ind_df[var].iloc[0] 
                + ind_df[var].iloc[1] - 1)
    if steps > ok_steps:
        return 10**10, np.nan
    
    # Tightening thresholds for that variable by "steps" steps
    temp_ind_df = ind_df.copy(deep = True)
    ttest  = test.copy(deep = True)
    ttrain = train.copy(deep = True)
    if direction == 'max':
        # Tighten thresholds for that variable:
        temp_ind_df.loc['max_ind'][var] += -steps
    if direction == 'min':
        temp_ind_df.loc['min_ind'][var] += steps
    # Defining thresholds:
    lo_threshs = [threshs[v][temp_ind_df.loc['min_ind'][v]] for v in variables]
    hi_threshs = [threshs[v][temp_ind_df.loc['max_ind'][v]] for v in variables]
    # Trimming data:
    for k,v in enumerate(variables):
        ttest[v].clip(lower = lo_threshs[k],
                      upper = hi_threshs[k],
                      inplace = True)
        ttrain[v].clip(lower = lo_threshs[k],
                       upper = hi_threshs[k],
                       inplace = True)
    # Run model:
    return run_model(ttest, ttrain, variables)


def update_thresholds(threshs, ind_df, summary_df, upper_aics, lower_aics, auc,
                      steps = 1, variables = ['VPD_night', 'GPP_1y']):
    # Find best of both clippings:
    best_upper = upper_aics[np.argmin(upper_aics)]
    best_lower = lower_aics[np.argmin(lower_aics)]
    aic = min([best_upper, best_lower])
    # Updating dataframes:
    if best_upper <= best_lower:
        # Upper clipping case:
        ind_df.iloc[1][variables[np.argmin(upper_aics)]] += -steps
        new_row = ([threshs[var][ind_df.loc['min_ind'][var]]
                    for var in variables] + 
                   [threshs[var][ind_df.loc['max_ind'][var]]
                    for var in variables] +
                   [aic, auc])
        summary_df.loc[len(summary_df)] = new_row
    else:
        # Lower clipping case:
        ind_df.iloc[0][variables[np.argmin(lower_aics)]] += steps
        new_row = ([threshs[var][ind_df.loc['min_ind'][var]]
                    for var in variables] + 
                   [threshs[var][ind_df.loc['max_ind'][var]]
                    for var in variables] +
                   [aic, auc])
        summary_df.loc[len(summary_df)] = new_row
    return ind_df, summary_df


def calculate_thresholds(variables, index = 1, fire_class = 'B'):
    start_time = time.time()
    # 1: Build basic test and train datasets.
    
    train = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                         f'genesis_inputs_train_{index}.csv'))
    test = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                        f'genesis_inputs_test_{index}.csv'))
    print(f'Train dataframe:\n{train}\n\n')
    print(f'Test dataframe:\n{test}\n\n')
    sys.stdout.flush()
    
    # 2: Create a dictionary of high and low thresholds (initially max and min) 
    threshs = define_threshold_options(variables)

    # 3: Find the AIC of the basic model (no clipping)
    aic, auc = run_model(test, train, variables)
    ind_df = initial_threshold_indices(variables) # Initial threshold values.
    summary_df = build_summary(threshs, ind_df, variables, aic, auc)

    # 4: Create compact tightening functions:
    tighten_upper = lambda var, n: tightening_multistep(var, variables,
                                                       ind_df, threshs,
                                                       test, train,
                                                       direction = 'max',
                                                       steps = n)
    tighten_lower = lambda var, n: tightening_multistep(var, variables,
                                                       ind_df, threshs,
                                                       test, train,
                                                       direction = 'min',
                                                       steps = n)

    # 5: Establish a loop (for loop with break condition)
    for count in range(10**5):
        sys.stdout.flush()
        print('\nCOUNT:', count+1)

        # a: Ratchet in from the upper threshold and find the best AIC.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            upper_aics = [tighten_upper(var, 1)[0] for var in variables]
            upper_aucs = [tighten_upper(var, 1)[1] for var in variables]
            best_upper = upper_aics[np.argmin(upper_aics)]
            best_upper_auc = upper_aucs[np.argmin(upper_aics)]

        # b: Ratchet in from the lower threshold and find the best AIC.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lower_aics = [tighten_lower(var, 1)[0] for var in variables]
            lower_aucs = [tighten_lower(var, 1)[1] for var in variables]
            best_lower = lower_aics[np.argmin(lower_aics)]
            best_lower_auc = lower_aucs[np.argmin(lower_aics)]

        # c: Check if an improvement by step of 2:
        improvement =  ((best_upper < aic - 2) or 
                        (best_lower < aic - 2))

        # d: If either (a) or (b) an improvement on the previous value.
        if improvement == True:
            # i: Update the target AIC to beat.
            print(f'Old AIC: {aic}')
            sys.stdout.flush()
            if best_upper <= best_lower:
                aic = best_upper
                auc = best_upper_auc
            else:
                aic = best_lower
                auc = best_lower_auc
            clip_type = ['Upper','Lower'][np.argmin([best_upper,best_lower])]
            if clip_type == 'Upper':
                clipped_var = variables[np.argmin(upper_aics)]
            elif clip_type == 'Lower':
                clipped_var = variables[np.argmin(lower_aics)]
            print(f'{clipped_var} : {clip_type} threshold tightened by 1 step')
            print(f'New AIC: {aic}. (AUC = {auc})')
            sys.stdout.flush()

            # ii: Find the best performing new thresholds.
            ind_df, summary_df = update_thresholds(threshs, ind_df, summary_df,
                                                   upper_aics, lower_aics, auc,
                                                   steps = 1,
                                                   variables = variables)
            summary_df.to_csv(('/rds/general/user/tk22/home/paper_2/model/step_2/'+
                               f'output/class_{fire_class}/'
                               f'threshold_summary_{index}.csv'))

        # e: If neither (a) nor (b) an improvement:
        if improvement == False:

            # i: Define max number of steps that could be tightened.
            max_steps = np.max(np.array([len(v) for v in threshs.values()])
                               - np.array(ind_df.iloc[0] - ind_df.iloc[1]))

            # ii: For each variable, ratchet in iteratively from upper thresh 
            for N in range(2, max_steps+1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    upper_aics = [tighten_upper(var, N)[0] for var in variables]
                    upper_aucs = [tighten_upper(var, N)[1] for var in variables]
                best_upper = upper_aics[np.argmin(upper_aics)]
                best_upper_auc = upper_aucs[np.argmin(upper_aics)]
                if best_upper < aic - 2:
                    upper_steps = N
                    improvement = True
                    break

            # iii: For each variable, ratchet in iteratively from lower thresh 
            for N in range(2, max_steps+1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lower_aics = [tighten_lower(var, N)[0] for var in variables]
                    lower_aucs = [tighten_lower(var, N)[1] for var in variables]
                best_lower = lower_aics[np.argmin(lower_aics)]
                best_lower_auc = lower_aucs[np.argmin(lower_aics)]
                if best_lower < aic - 2:
                    lower_steps = N
                    improvement = True
                    break

            # iv: 
            if improvement == True:
                # 1: Was lower or upper clipping better?
                upper_best = (best_upper <= best_lower)
                print(f'Old AIC: {aic}')

                clip_type = ['Upper','Lower'][np.argmin([best_upper,best_lower])]
                if clip_type == 'Upper':
                    clipped_var = variables[np.argmin(upper_aics)]
                elif clip_type == 'Lower':
                    clipped_var = variables[np.argmin(lower_aics)]

                if upper_best:
                    print((f'{clipped_var} : {clip_type} threshold '
                           +f'tightened by {upper_steps} steps'))
                    n_steps = upper_steps
                else:
                    print((f'{clipped_var} : {clip_type} threshold '+
                           f'tightened by {lower_steps} steps'))
                    n_steps = lower_steps


                # 2: Update the target AIC to beat.
                if upper_best:
                    aic = best_upper
                    auc = best_upper_auc
                else:
                    aic = best_lower
                    auc = best_lower_auc

                print(f'New AIC: {aic}. (AUC = {auc})')

                # 3: Find the best performing new thresholds.
                ind_df, summary_df = update_thresholds(threshs, ind_df, summary_df,
                                                       upper_aics, lower_aics, auc, 
                                                       steps = n_steps,
                                                       variables = variables)
                summary_df.to_csv(('/rds/general/user/tk22/home/paper_2/model/step_2/'+
                                   f'output/class_{fire_class}/'
                                   f'threshold_summary_{index}.csv'))


            # v: Break loop if no improvement.
            else:
                print('Jumping thresholds gives no further improvement.')
                break
    total_seconds = time.time() - start_time
    print(f'\n\n\nTOTAL RUN TIME:\n\n{(total_seconds / (60**2)):.2f} hours')
    sys.stdout.flush()
    return


def variable_set(fire_class = 'B'):
    all_set = []
    for i in range(0,1001):
        try:
            path = ('/rds/general/user/tk22/home/paper_2/model/step_1/'+
                    f'selected_vars/class_{fire_class}/predictor_list_{i}.txt')
            with open(path, 'r') as file:
                data = file.read().split('", "')
                data = [datum.split('"') for datum in data]
                data = [item for sublist in data for item in sublist]
                data = data[1:-1]
                data.sort()
            all_set.append(data)
        except FileNotFoundError:
            pass
    unique_set = [list(x) for x in set(tuple(x) for x in all_set)]
    return unique_set


def specific_variable(index, fire_class = 'B'):
    path = ('/rds/general/user/tk22/home/paper_2/model/step_1/'+
            f'selected_vars/class_{fire_class}/predictor_list_{index}.txt')
    with open(path, 'r') as file:
        data = file.read().split('", "')
        data = [datum.split('"') for datum in data]
        data = [item for sublist in data for item in sublist]
        data = sorted(data[1:-1])
    return data


def main(index = 1, fire_class = 'B'):
    unique_sets = variable_set(fire_class = fire_class)
    print(f'Number of unique sets = {len(unique_sets)}')
    variables = specific_variable(index, fire_class = fire_class)
    print(f'{variables}')
    sys.stdout.flush()
    calculate_thresholds(variables, index = index,
                         fire_class = fire_class)
    return


if __name__ == '__main__':
    index = int(os.getenv('PBS_ARRAY_INDEX')) # 0 to 10000
    fire_class = sys.argv[1]
    main(index = index, fire_class = fire_class)