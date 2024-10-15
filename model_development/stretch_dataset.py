import numpy as np
import xarray as xr
import pandas as pd
import time
import warnings
import sys


def opt_step(arr1, arr2, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Optimising:
        temp = arr1 ** b
        a = float(np.nanmean(arr2))/float(np.nanmean(temp))
        temp = a * arr1 ** b
        # Evaluation metrics:
        mod = np.nanmean(temp, axis = 0)
        obs = np.nanmean(arr2, axis = 0)
        # How well do non-nan quantiles match?
        obs_qs = []
        mod_qs = []
        for q in np.linspace(0,1,10001):
            obs_qs.append(np.nanquantile(np.log10(obs), q))
            mod_qs.append(np.nanquantile(np.log10(mod), q))
        try:
            res = np.array(obs_qs)[np.isfinite(obs_qs)] - np.array(mod_qs)[np.isfinite(obs_qs)]
            rss = np.nanmean(res**2)
        except:
            rss = np.nan
    return a, b, rss


def one_chain(arr1, arr2, lower_b, upper_b, n_steps, a_vals, b_vals, rss_vals):
    temp_a_vals = []
    temp_b_vals = []
    temp_rss_vals = []
    for b in np.linspace(lower_b, upper_b, n_steps):
        a, b, rss = opt_step(arr1, arr2, b)
        print(f'\ta = {a:.10f}, b = {b:.10f}, rss = {rss:.10f}')
        sys.stdout.flush()
        # a values:
        a_vals.append(a)
        temp_a_vals.append(a)
        # b values:
        b_vals.append(b)
        temp_b_vals.append(b)
        # rss values:
        rss_vals.append(rss)
        temp_rss_vals.append(rss)
    return a_vals, temp_a_vals, b_vals, temp_b_vals, rss_vals, temp_rss_vals


def stretch_dataset(ds, mask, decimals = 11):
    # Target values:
    a_vals = []
    b_vals = []
    rss_vals = []
    # Building arrays:
    arr1 = ds.p.to_numpy()
    arr2 = ds.counts.to_numpy()
    # Masking:
    arr1 = np.einsum('ijk,jk->ijk', arr1, mask)
    arr2 = np.einsum('ijk,jk->ijk', arr2, mask)
    # Tightening steps:
    t0 = time.time()
    n_steps = 21
    lower_b, upper_b = 1, 3
    
    for iteration in range(1, decimals):
        print(f'Iteration {iteration}: {(time.time() - t0)/60:.1f} minutes')
        sys.stdout.flush()
        # Running a chain:
        (a_vals, temp_a_vals, 
         b_vals, temp_b_vals, 
         rss_vals, temp_rss_vals) = one_chain(arr1, arr2, 
                                              lower_b, upper_b, n_steps, 
                                              a_vals, b_vals, rss_vals)
        sys.stdout.flush()
        # Getting minimum:
        min_index = np.argmin(temp_rss_vals)
        step_width = (upper_b - lower_b) / (n_steps - 1)
        min_b = temp_b_vals[min_index]
        # Checking if bounds are correct:
        if min_b == lower_b:
            while min_b == lower_b:
                offset = (upper_b - lower_b) / 2
                lower_b = lower_b - offset
                upper_b = upper_b - offset
                # Running offset chain:
                (a_vals, temp_a_vals, 
                 b_vals, temp_b_vals, 
                 rss_vals, temp_rss_vals) = one_chain(arr1, arr2, 
                                                      lower_b, upper_b, n_steps, 
                                                      a_vals, b_vals, rss_vals)
                sys.stdout.flush()
                # Getting minimum:
                min_index = np.argmin(temp_rss_vals)
                min_b = temp_b_vals[min_index]
                
        elif min_b == upper_b:
            while min_b == upper_b:
                offset = (upper_b - lower_b) / 2
                lower_b = lower_b + offset
                upper_b = upper_b + offset
                # Running offset chain:
                (a_vals, temp_a_vals, 
                 b_vals, temp_b_vals, 
                 rss_vals, temp_rss_vals) = one_chain(arr1, arr2, 
                                                      lower_b, upper_b, n_steps, 
                                                      a_vals, b_vals, rss_vals)
                sys.stdout.flush()
                # Getting minimum:
                min_index = np.argmin(temp_rss_vals)
                min_b = temp_b_vals[min_index]
            
        else:
            # ALL IS GOOD AN WE CAN PROCEED BY SETTING NEW BOUNDS:
            lower_b = min_b - step_width
            upper_b = min_b + step_width
        
    df = pd.DataFrame({'a'  : a_vals,
                       'b'  : b_vals,
                       'rss': rss_vals})
    return df