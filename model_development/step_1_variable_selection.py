import glob
import sys
import gc
import time
import json
import os

import xarray as xr
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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


def build_inputs(N, fire_class = 'B'):

    directory = ('/rds/general/user/tk22/projects/'+
                 'leverhulme_wildfires_theo_keeping/'+
                 'live/ensemble_data/')
    reanalysis_paths = glob.glob(directory + 'reanalysis_*_19900101_20211231.nc')
    landscape_paths = glob.glob(directory + 'landscape_*_19920101_20201231.nc')
    fire_path = glob.glob(directory + f'counts_{fire_class}*_19920101_20201231.nc')
    size_path = glob.glob(directory + f'cell_area*_19920101_20201231.nc')

    ds_paths = reanalysis_paths + landscape_paths + fire_path + size_path

    mask = build_mask(ds_paths, pre_2002 = False)

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
        count += 1
        ds = xr.open_dataset(path)
        ds = ds.sel(time = ds.time.dt.year <= 2020)
        ds = ds.sel(time = ds.time.dt.year >= 1992)
        var = [v for v in list(ds.variables) 
               if v not in ['latitude', 'longitude', 'time']][0]
        ds = ds[var].to_numpy()[sample_inds]
        print('\t',var)
        if count == 1:
            df = pd.DataFrame({var: ds})
        else:
            df[var] = ds


    df = df.dropna() # as a precaution, should be none
    train, test = train_test_split(df, test_size = 0.2)

    return train, test



def model_fit(test, train, predictor_list):
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
    return aic, auc, max_vif

def variable_selection_forwards(df):
    # Selection: of those with max_vif < 5, max(AIC) given AUC in top quartile/half
    #            unless max_vif all > 5, then must be < 10.
    temp_1 = df[df['Max VIF'] < 5]
    temp_2 = df[df['Max VIF'] < 10]
    if len(temp_1) > 0:
        # Finding min(AIC):
        var = temp_1.iloc[np.argmin(temp_1['AIC'])]['New Variable']
        aic = np.min(temp_1['AIC'])
    elif len(temp_2) > 0:
        # Finding min(AIC):
        var = temp_2.iloc[np.argmin(temp_2['AIC'])]['New Variable']
        aic = np.min(temp_2['AIC'])
    else:
        print('FAILED: VIFs too high.')
    return var, aic

def forwards_step(J, test, train, prior_predictors, whitelist, fire_class = 'C', step = 1):
    # 3: A forwards step function; it runs (2) adding one new variable from a
    #    whitelist, and finds the optimal new variable. Saves df of stats from (2).
    results = []
    for i, new_var in enumerate(whitelist):
        result = model_fit(test, train, prior_predictors + [new_var])
        results.append(result)
        gc.collect()
    results = [tuple([whitelist[i]]+list(r)) for i,r in enumerate(results)]
    df = pd.DataFrame(results, columns = ['New Variable', 'AIC', 'AUC', 'Max VIF'])
    df.to_csv(f'/rds/general/user/tk22/home/paper_2/model/step_1/outputs/'+
              f'class_{fire_class}/run_{J}/'+
              f'run_{J}_step_{step}_results_forwards_{J}th_run.csv', index = False)
    return df

def variable_selection_switch(df):
    # Selection: of those with max_vif < 5, max(AIC) given AUC in top quartile/half
    #            unless max_vif all > 5, then must be < 10.
    # First VIF condition:
    temp_1 = df[df['Max VIF'] < 5]
    temp_2 = df[df['Max VIF'] < 10]
    if len(temp_1) > 0:
        # Finding min(AIC):
        new_var = temp_1.iloc[np.argmin(temp_1['AIC'])]['New Variable']
        old_var = temp_1.iloc[np.argmin(temp_1['AIC'])]['Old Variable']
        aic = np.min(temp_1['AIC'])
    elif len(temp_2) > 0:
        # Finding min(AIC):
        new_var = temp_2.iloc[np.argmin(temp_2['AIC'])]['New Variable']
        old_var = temp_2.iloc[np.argmin(temp_2['AIC'])]['Old Variable']
        aic = np.min(temp_2['AIC'])
    else:
        print('FAILED: VIFs too high.')
    return new_var, old_var, aic


def switch_step(J, test, train, prior_predictors, whitelist, fire_class = 'C', step = 1):
    # 4: A backwards-forwards step function; it runs (2) substituting one new 
    #    variable from a whitelist for each old one, and finds the optimal new 
    #    variable. Saves df of stats from (2).

    results = []
    for k,old_var in enumerate(prior_predictors):
        print(f'{k+1} steps of {len(prior_predictors)} complete.')
        sys.stdout.flush()

        for i, new_var in enumerate(whitelist):
            predictors = list((set(prior_predictors) - set([old_var])).union(set([new_var])))
            result = model_fit(test, train, predictors)
            results.append(result)
            gc.collect()
    
    c = 0
    final_results = []
    for old_var in prior_predictors:
        for new_var in whitelist:
            final_results.append(tuple([old_var] + [new_var] + list(results[c])))
            c += 1

    df = pd.DataFrame(final_results, columns = ['Old Variable', 'New Variable', 'AIC', 'AUC', 'Max VIF'])
    df.to_csv(f'/rds/general/user/tk22/home/paper_2/model/step_1/outputs/'+
              f'class_{fire_class}/run_{J}/'+
              f'run_{J}_step_{step}_results_switch_{J}th_run.csv', index = False)
    return df



def runner(J, fire_class = 'B'):
    old_predictor_list = []
    new_predictor_list = []

    train = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                         f'genesis_inputs_train_{J}.csv'))
    test = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                        f'genesis_inputs_test_{J}.csv'))
    print(f'Train dataframe:\n{train}\n\n')
    print(f'Test dataframe:\n{test}\n\n')
    sys.stdout.flush()

    whitelist = list(set(test.columns) - set(['counts', 'cell_area']))

    old_aic = 10**10

    for step in range(1000):
        print(f'\n\n\nStep {step}')
        sys.stdout.flush()

        # First break condition:
        if len(new_predictor_list) < len(old_predictor_list):
            print('BREAK CONDITION 1')
            # As no benefit to adding a variable,
            # since duplicate selected.
            break

        if sorted(old_predictor_list) == sorted(new_predictor_list):
            # Forwards step:
            print('\n\tForwards step.')
            t0 = time.time()
            was_forwards = True

            reduced_whitelist = list(set(whitelist) - set(new_predictor_list))

            df = forwards_step(J, test, train, new_predictor_list,
                               reduced_whitelist, fire_class = fire_class, step = step)

            var, new_aic = variable_selection_forwards(df)

            old_predictor_list = new_predictor_list.copy()

            new_predictor_list.append(var)
            
            print(f'\n\t{(time.time()-t0)/60:.1f} minutes.')



        else:
            if sorted(old_predictor_list) != sorted(new_predictor_list):
                # Switch step (we repeat until stable):
                print('\n\tSwitch step:\n')
                was_forwards = False
                t0 = time.time()
                
                df = switch_step(J, test, train, new_predictor_list,
                                 whitelist, fire_class = fire_class, step = step)

                new_var, old_var, new_aic = variable_selection_switch(df)

                old_predictor_list = new_predictor_list.copy()

                new_predictor_list.remove(old_var)
                new_predictor_list.append(new_var)

                # In case we have added a duplicate value to the list:
                # Keeping only unique variables:
                new_predictor_list = list(set(new_predictor_list))
                
                print(f'\n\t{(time.time()-t0)/60:.1f} minutes.')


        print(f'\tOld predictors:\n\t\t{old_predictor_list}')
        print(f'\tNew predictors:\n\t\t{new_predictor_list}')
        print(f'\n\tAIC = {new_aic}')

        # Second break condition:
        if was_forwards == True:
            if new_aic > old_aic - 2: 
                print('BREAK CONDITION 2')
                # As model NOT considered significantly better if 
                # 2 AIC units lesser, Burnham and Anderson (2004)
                break

        old_aic = new_aic


        # Third break condition:
        if was_forwards == True:
            if len(new_predictor_list) > 12:
                # Exceeding max predictors count:
                print('BREAK CONDITION 3')
                new_predictor_list = old_predictor_list.copy()
                break


    list_path = (f'/rds/general/user/tk22/home/paper_2/model/step_1/'+
                 f'selected_vars/class_{fire_class}/predictor_list_{J}.txt')
    with open(list_path, "w") as file:
        json.dump(new_predictor_list, file)
    with open(list_path, "r") as file:
        new_predictor_list = json.load(file)
        print(new_predictor_list)
    sys.stdout.flush()
    return

if __name__ == '__main__':
    J = int(os.getenv('PBS_ARRAY_INDEX'))
    fire_class = sys.argv[1]
    
    runner(J, fire_class = fire_class)