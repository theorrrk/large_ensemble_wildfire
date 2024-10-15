import xarray as xr
from datetime import time as dtt
import pandas as pd
import numpy as np
import sys
import glob
import random
import os
import warnings
from matplotlib import pyplot as plt
from matplotlib import colors


def moment_scatterplots(variable):    
    paths = glob.glob('/rds/general/user/tk22/home/paper_2/bias_correction/'+
                      f'moment_test/{variable}_*.csv')
    for i, path in enumerate(paths):
        temp = pd.read_csv(path)
        temp['hour'] = int(path[-8:-4])
        if i == 0:
            df = temp.copy(deep = True)
        else:
            df = pd.concat([df, temp])

    df_obs = df.loc[df.member == 'obs']
    df_ens = df.loc[[x[0]=='h' for x in df.member]]

    df_ens_sample = df_ens.groupby(['lat','lon','month','hour']).sample(1)

    df_obs = df_obs.sort_values(['lat','lon','month','hour'])
    df_ens_sample = df_ens_sample.sort_values(['lat','lon','month','hour'])

    fig, axs = plt.subplots(2,2,figsize=(8,8))

    cmap = 'copper'#'GnBu'#'jet'

    stat = 'mean'
    uber_min = np.min([df_obs[stat].min(),df_ens_sample[stat].min()])
    uber_max = np.max([df_obs[stat].max(),df_ens_sample[stat].max()])
    heatmap, xedges, yedges = np.histogram2d(df_ens_sample[stat], df_obs[stat], bins = 100,
                                             range = ((uber_min, uber_max),
                                                      (uber_min, uber_max)))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im0 = axs[0,0].imshow(heatmap.T, extent = extent, origin='lower', 
                          norm = colors.LogNorm(vmin = 1), cmap = cmap)
    plt.colorbar(im0, ax = axs[0,0], aspect = 25, shrink = 0.6)
    axs[0,0].set_title(f'Mean', fontsize = 10)
    axs[0,0].set_xlim((uber_min, uber_max))
    axs[0,0].set_ylim((uber_min, uber_max))
    axs[0,0].set_xlabel('Ensemble Data')
    axs[0,0].set_ylabel('Reanalysis Data')
    nan_mask = np.logical_or(np.array(np.isnan(df_obs[stat])),
                             np.array(np.isnan(df_ens_sample[stat])))
    R2 = metrics.r2_score(df_obs[stat].loc[~nan_mask],
                          df_ens_sample[stat].loc[~nan_mask])
    axs[0,0].text(0.03, 0.97, f'R$^2$ = {R2:.3f}', ha='left', va='top', transform=axs[0,0].transAxes)

    stat = 'var'
    uber_min = np.min([df_obs[stat].min(),df_ens_sample[stat].min()])
    uber_max = np.max([df_obs[stat].max(),df_ens_sample[stat].max()])
    heatmap, xedges, yedges = np.histogram2d(df_ens_sample[stat], df_obs[stat], bins = 100,
                                             range = ((uber_min, uber_max),
                                                      (uber_min, uber_max)))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im0 = axs[0,1].imshow(heatmap.T, extent = extent, origin='lower', 
                          norm = colors.LogNorm(vmin = 1), cmap = cmap)
    plt.colorbar(im0, ax = axs[0,1], aspect = 25, shrink = 0.6)
    axs[0,1].set_title(f'Variance', fontsize = 10)
    axs[0,1].set_xlim((uber_min, uber_max))
    axs[0,1].set_ylim((uber_min, uber_max))
    axs[0,1].set_xlabel('Ensemble Data')
    axs[0,1].set_ylabel('Reanalysis Data')
    nan_mask = np.logical_or(np.array(np.isnan(df_obs[stat])),
                             np.array(np.isnan(df_ens_sample[stat])))
    R2 = metrics.r2_score(df_obs[stat].loc[~nan_mask],
                          df_ens_sample[stat].loc[~nan_mask])
    axs[0,1].text(0.03, 0.97, f'R$^2$ = {R2:.3f}', ha='left', va='top', transform=axs[0,1].transAxes)

    stat = 'skew'
    uber_min = np.min([df_obs[stat].min(),df_ens_sample[stat].min()])
    uber_max = np.max([df_obs[stat].max(),df_ens_sample[stat].max()])
    heatmap, xedges, yedges = np.histogram2d(df_ens_sample[stat], df_obs[stat], bins = 100,
                                             range = ((uber_min, uber_max),
                                                      (uber_min, uber_max)))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im0 = axs[1,0].imshow(heatmap.T, extent = extent, origin='lower', 
                          norm = colors.LogNorm(vmin = 1), cmap = cmap)
    plt.colorbar(im0, ax = axs[1,0], aspect = 25, shrink = 0.6)
    axs[1,0].set_title(f'Skewness', fontsize = 10)
    axs[1,0].set_xlim((uber_min, uber_max))
    axs[1,0].set_ylim((uber_min, uber_max))
    axs[1,0].set_xlabel('Ensemble Data')
    axs[1,0].set_ylabel('Reanalysis Data')
    nan_mask = np.logical_or(np.array(np.isnan(df_obs[stat])),
                             np.array(np.isnan(df_ens_sample[stat])))
    R2 = metrics.r2_score(df_obs[stat].loc[~nan_mask],
                          df_ens_sample[stat].loc[~nan_mask])
    axs[1,0].text(0.03, 0.97, f'R$^2$ = {R2:.3f}', ha='left', va='top', transform=axs[1,0].transAxes)

    stat = 'kurt'
    uber_min = np.min([df_obs[stat].min(),df_ens_sample[stat].min()])
    uber_max = np.max([df_obs[stat].max(),df_ens_sample[stat].max()])
    heatmap, xedges, yedges = np.histogram2d(df_ens_sample[stat], df_obs[stat], bins = 100,
                                             range = ((uber_min, uber_max),
                                                      (uber_min, uber_max)))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im0 = axs[1,1].imshow(heatmap.T, extent = extent, origin='lower', 
                          norm = colors.LogNorm(vmin = 1), cmap = cmap, filternorm = False)
    plt.colorbar(im0, ax = axs[1,1], aspect = 25, shrink = 0.6)
    axs[1,1].set_title(f'Kurtosis', fontsize = 10)
    axs[1,1].set_xlim((uber_min, uber_max))
    axs[1,1].set_ylim((uber_min, uber_max))
    axs[1,1].set_xlabel('Ensemble Data')
    axs[1,1].set_ylabel('Reanalysis Data')
    nan_mask = np.logical_or(np.array(np.isnan(df_obs[stat])),
                             np.array(np.isnan(df_ens_sample[stat])))
    R2 = metrics.r2_score(df_obs[stat].loc[~nan_mask],
                          df_ens_sample[stat].loc[~nan_mask])
    axs[1,1].text(0.03, 0.97, f'R$^2$ = {R2:.3f}', ha='left', va='top', transform=axs[1,1].transAxes)

    plt.tight_layout()
    plt.savefig('/rds/general/user/tk22/home/paper_2/bias_correction/'+
                f'figures/scatterplot_{variable}.png',
                bbox_inches = 'tight', facecolor = 'white', dpi = 1200)
    plt.show()
    return


def comparative_maps(variable, period):
    if period == 'hist':
        mem = 'h'+str(random.randint(10,169)).zfill(3)
        paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                          f'{variable}_historical_{mem}_*.nc')
    if period == 'proj':
        mem = 's'+str(random.randint(10,169)).zfill(3)
        paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                          f'{variable}_projected_{mem}_*.nc')
    hours = [xr.open_dataset(path).time.dt.time.data[0].strftime("%H%M")
             for path in paths]
    
    for hour in hours:
        observations = glob.glob('/rds/general/user/tk22/projects/leverhulme_'+
                                 'wildfires_theo_keeping/live/ensemble_data/'+
                                 f'processing/{variable}*era5.nc')
        ds2 = xr.open_dataset(min(observations,key = len))
        ds2.sel(time = ds2.time.dt.hour == int(hour[:2]))
        da2 = ds2[variable].mean(dim = 'time')
        
        if period == 'hist':
            members = ['h'+str(x).zfill(3) for x in range(10,170)]
        if period == 'proj':
            members = ['s'+str(x).zfill(3) for x in range(10,170)]
        
        for i,mem in enumerate(members):
            if period == 'hist':
                ds1 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                                      f'{variable}_historical_{mem}_{hour}.nc')
            if period == 'proj':
                ds1 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                                      f'{variable}_projected_{mem}_{hour}.nc')
            if i == 0:
                da1 = ds1[variable].mean(dim = ['member','time'])
            else:
                da1 += ds1[variable].mean(dim = ['member','time'])
                
        da1 = da1 / len(['h'+str(x).zfill(3) for x in range(10,170)])

        fig, axs = plt.subplots(2,1,figsize = (7,7))
        
        vmin = min([da1.min(),da2.min()])
        vmax = max([da1.max(),da2.max()])

        da1.plot(cmap = 'jet', ax = axs[0], vmin = vmin, vmax = vmax)
        axs[0].set_title(f'Randomly Selected Member: {mem}')
        da2.plot(cmap = 'jet', ax = axs[1], vmin = vmin, vmax = vmax)
        axs[1].set_title(f'Observation')

        plt.suptitle(f'{variable} Comparative Map: {hour} UTC')
        plt.tight_layout()
        plt.savefig('/rds/general/user/tk22/home/paper_2/bias_correction/'+
                    f'figures/comparative_map_{variable}_{hour}_{period}.png',
                    bbox_inches = 'tight', facecolor = 'white', dpi = 900)
        plt.show()
    return


def comparative_histograms(variable, period):
    if period == 'hist':
        mem = 'h'+str(random.randint(10,169)).zfill(3)
        paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                          f'{variable}_historical_{mem}_*.nc')
    if period == 'proj':
        mem = 's'+str(random.randint(10,169)).zfill(3)
        paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                          f'{variable}_projected_{mem}_*.nc')
    hours = [xr.open_dataset(path).time.dt.time.data[0].strftime("%H%M")
             for path in paths]

    for hour in hours:
        count = 0
        if period == 'hist':
            members = ['h'+str(x).zfill(3) for x in range(10,170)]
        if period == 'proj':
            members = ['s'+str(x).zfill(3) for x in range(10,170)]
        for i,mem in enumerate(members):
            count += 1
            if variable == 'sfcwind':
                if period == 'hist':
                    da1 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corr'+
                                          f'ected/sfcwind_rescaled_historical_{mem}_{hour}'+
                                          '.nc')['sfcwind_rescaled'].to_numpy()
                if period == 'proj':
                    da1 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corr'+
                                          f'ected/sfcwind_rescaled_projected_{mem}_{hour}'+
                                          '.nc')['sfcwind_rescaled'].to_numpy()
                    
            else:
                if period == 'hist':
                    da1 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                                          f'{variable}_historical_{mem}_{hour}'+
                                          '.nc')[variable].to_numpy()
                if period == 'proj':
                    da1 = xr.open_dataset('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                                          f'{variable}_projected_{mem}_{hour}'+
                                          '.nc')[variable].to_numpy()
            dmin = np.nanmin(da1)
            dmax = np.nanmax(da1)

            if i == 0:
                hist,edgs = np.histogram(da1[~np.isnan(da1)], density = True,
                                         bins = np.linspace(dmin,dmax,21))
                hist1 = hist
            else:
                hist,edgs = np.histogram(da1[~np.isnan(da1)], density = True,
                                         bins = np.linspace(dmin,dmax,21))
                hist1 += hist
        hist1 = hist1 / count

        observations = glob.glob('/rds/general/user/tk22/projects/leverhulme_'+
                                 'wildfires_theo_keeping/live/ensemble_data/'+
                                 f'processing/{variable}*era5.nc')
        da2 = xr.open_dataset(min(observations,key = len))[variable]
        da2 = da2.sel(time = da2.time.dt.hour == int(hour[:2])).to_numpy()
        hist2,edgs = np.histogram(da2[~np.isnan(da2)], density = True,
                                  bins = np.linspace(dmin,dmax,21))

        plt.figure(figsize = (7,3))
        plt.title(f'{variable}Comparative PDFs: ({hour} UTC)')
        plt.bar((edgs[1:]+edgs[:-1])/2, hist1, width = edgs[1]-edgs[0],
                alpha = 0.5, color = 'indianred', label = 'Ensemble')
        plt.bar((edgs[1:]+edgs[:-1])/2, hist2, width = edgs[1]-edgs[0],
                alpha = 0.5, color = 'cornflowerblue', label = 'Observation')
        plt.legend()
        plt.savefig('/rds/general/user/tk22/home/paper_2/bias_correction/'+
                    f'figures/comparative_hist_{variable}_{hour}_{period}.png',
                    bbox_inches = 'tight', facecolor = 'white', dpi = 900)
        plt.show()
    return
        
        
        
def run_summary_figures(variable, period):
    if variable == 'sfcwind':
        moment_scatterplots('sfcwind_variance_scaled'+'_'+period)
        comparative_maps(variable,period)
        comparative_histograms(variable,period)
        
    else:    
        moment_scatterplots(variable+'_'+period)
        comparative_maps(variable,period)
        comparative_histograms(variable,period)
    return


def temporally_smooth_observation(x_oh_a):
    # Ensuring just for 30 year window:
    x_oh_a = x_oh_a.sel(time = x_oh_a.time.dt.year >= 1990)
    x_oh_a = x_oh_a.sel(time = x_oh_a.time.dt.year <= 2019)
    # Getting the rolling mean (15 days each side):
    x_oh_a_tilde = x_oh_a.rolling(time = 31, center = True).mean()
    # Getting dayofyear average:
    doy_average = x_oh_a_tilde.groupby('time.dayofyear').mean()
    # Updating x_oh_a_tilde by dayofyear:
    for i in range(len(x_oh_a_tilde.time)):
        doy = int(x_oh_a_tilde.time[i].dt.dayofyear)
        if doy == 366:
            doy = 365 # Avoiding leap year undersampling
        doy_index = doy - 1
        x_oh_a_tilde[i,:,:] = doy_average[doy_index,:,:]
    # Paranoid tidying:
    x_oh_a_tilde.load()
    doy_average.close()
    # Trimming down to 10 years:
    x_oh_a_tilde = x_oh_a_tilde.sel(time = x_oh_a_tilde.time.dt.year >= 2000)
    x_oh_a_tilde = x_oh_a_tilde.sel(time = x_oh_a_tilde.time.dt.year <= 2009)
    del doy_average
    return x_oh_a_tilde


def similar_baseline_model(x_mh_b):
    '''
    The original function here also smoothed by a spatial interpolation of 2pi*gridscale.
    I have removed that phase as it led to significant bias in the anomaly
    (i.e. not summing to 0)
    '''
    #x_mh_b.load()
    x_mh_b_base = x_mh_b.copy(deep = True)
    # Getting dayofyear average:
    doy_average = x_mh_b_base.groupby('time.dayofyear').mean()[:,:365,:,:] # Avoiding leap year undersampling
    # Smoothing with moving 31 day window:
    doy_average.load()
    doy_average_smoothed = doy_average.copy(deep = True)
    doy_average_smoothed.load()
    for doy in list(doy_average.dayofyear.data):
        print(f'\t\tDOY: {doy}')
        sys.stdout.flush()
        index = doy - 1
        if (index >= 15) and (index < 350):
            avg = doy_average[:,index-15:index+15+1,:,:].mean(axis = 1)
            doy_average_smoothed[:,index,:,:] = avg

        elif index < 15:
            avg = xr.concat([doy_average[:,:index+15+1,:,:],
                             doy_average[:,-(15-index):,:,:]], 'dayofyear').mean(axis = 1)
            doy_average_smoothed[:,index,:,:] = avg

        elif index >= 350:
            avg = xr.concat([doy_average[:,:(15-365+index+1),:,:],
                             doy_average[:,index-15:,:,:]], 'dayofyear').mean(axis = 1)
            doy_average_smoothed[:,index,:,:] = avg
            
    doy_average_smoothed.load()
    x_mh_b_base.load()

    # Updating x_mh_b_base by dayofyear:
    for i in range(len(x_mh_b_base.time)):
        if i % 100 == 0:
            print(f'\t\t{i} steps of {len(x_mh_b_base.time)}')
            sys.stdout.flush()
        doy = int(x_mh_b_base.time[i].dt.dayofyear)
        if doy == 366:
            doy = 365 # Avoiding leap year undersampling
        doy_index = doy - 1
        x_mh_b_base[:,i,:,:] = doy_average_smoothed[:,doy_index,:,:]
    # Paranoid tidying:
    doy_average.close()
    doy_average_smoothed.close()
    del doy_average, doy_average_smoothed, avg
    x_mh_b_base.load()
    return x_mh_b_base


def interpolate_d(d_mh_b, by = None, ttime = None, period = 'hist', var = 'tas'):
    for i, mem in enumerate(list(d_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        if var == 'GPP':
            da = d_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                          longitude = by.longitude.data,
                                                          method = 'nearest')
        else:
            da = d_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                          longitude = by.longitude.data)
        da = da.expand_dims(dim = 'member')
        if period == 'hist':
            da.to_netcdf(f'/rds/general/user/tk22/ephemeral/tmp/d_mh_a_{var}_{mem}_{period}_daily.nc')
        elif period == 'proj':
            da.to_netcdf(f'/rds/general/user/tk22/ephemeral/tmp/d_mp_a_{var}_{mem}_{period}_daily.nc')
        del da
    #d_mh_a = xr.open_mfdataset(f'/rds/general/user/tk22/ephemeral/tmp/d_mh_b_member_*.nc')
    return #d_mh_a


def interpolate_q(q_mh_b, by = None, ttime = None, period = 'hist', var = 'tas'):
    for i, mem in enumerate(list(q_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        if var == 'GPP':
            da = q_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                          longitude = by.longitude.data,
                                                          method = 'nearest')
        else:
            da = q_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                          longitude = by.longitude.data)
        da = da.expand_dims(dim = 'member')
        if period == 'hist':
            da.to_netcdf(f'/rds/general/user/tk22/ephemeral/tmp/q_mh_a_{var}_{mem}_{period}_daily.nc')
        elif period == 'proj':
            da.to_netcdf(f'/rds/general/user/tk22/ephemeral/tmp/q_mp_a_{var}_{mem}_{period}_daily.nc')
        del da
    #d_mh_a = xr.open_mfdataset(f'/rds/general/user/tk22/ephemeral/tmp/d_mh_b_member_*.nc')
    return #d_mh_a


def bc_downscaled_output_d(x_mh_b, x_oh_a_tilde, variable = 'tas', ttime = None, period = 'hist', var = 'tas'):
    for i, mem in enumerate(list(x_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        if period == 'hist':
            da = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/tmp/'+
                                 f'd_mh_a_{var}_{mem}_{period}_daily.nc')
        elif period == 'proj':
            da = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/tmp/'+
                                 f'd_mp_a_{var}_{mem}_{period}_daily.nc')
        var = list(set(list(da.variables)) - 
                   set(['member','time','latitude','longitude']))[0]
        da = da[var]
        da = da + x_oh_a_tilde
                                 
        if variable == 'snc':
            da = da.clip(min = 0, max = 100)
            
        if variable == 'fAPAR':
            da = da.clip(min = 0, max = 1)
                                 
        if variable == 'GPP':
            da = da.clip(min = 0)
                                 

        if period == 'hist':
            da.to_netcdf('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'{variable}_historical_{mem}_daily.nc')
        elif period == 'proj':
            da.to_netcdf('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'{variable}_projected_{mem}_daily.nc')
    return


def bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = 'pr', ttime = None, period = 'hist', var = 'tas'):
    for i, mem in enumerate(list(x_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        if period == 'hist':
            da = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/tmp/'+
                                 f'q_mh_a_{var}_{mem}_{period}_daily.nc')
        elif period == 'proj':
            da = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/tmp/'+
                                 f'q_mp_a_{var}_{mem}_{period}_daily.nc')
        var = list(set(list(da.variables)) - 
                   set(['member','time','latitude','longitude']))[0]
        da = da[var]
        da = da * x_oh_a_tilde
                                 
        if variable == 'snc':
            da = da.clip(min = 0, max = 100)
            
        if variable == 'fAPAR':
            da = da.clip(min = 0, max = 1)
                                 
        if variable == 'GPP':
            da = da.clip(min = 0)

        if period == 'hist':
            da.to_netcdf(f'/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'{variable}_historical_{mem}_daily.nc')
        elif period == 'proj':
            da.to_netcdf(f'/rds/general/user/tk22/ephemeral/bias_corrected/'+
                         f'{variable}_projected_{mem}_daily.nc')
    return




                         
def rescale_variance(variable = None, ttime = None):
    df = pd.read_csv('/rds/general/user/tk22/home/paper_2/bias_correction/'+
                     f'moment_test/{variable}_daily.csv')
    df_obs = df.loc[df.member == 'obs']
    df_ens = df.loc[[x[0]=='h' for x in df.member]]
    df_ens_sample = df_ens.groupby(['lat','lon','month']).sample(1)

    df_obs = df_obs.sort_values(['lat','lon','month'])
    df_ens_sample = df_ens_sample.sort_values(['lat','lon','month'])

    s = np.nanmean(np.array(df_obs['var']) / np.array(df_ens_sample['var']))
    output_paths = glob.glob(('/rds/general/user/tk22/ephemeral/bias_corrected/'+
                              f'{variable}_historical_*_daily.nc'))

    count = 0
    mean = 0
    for output_path in output_paths:                      
        da = xr.open_dataset(output_path)[variable][0,:,:,:]
        #da_doy_avg = da.groupby('time.dayofyear').mean(dim = 'time')

        #for i in range(1,367):
        #    if i == 366:
        #        da.loc[da.time.dt.dayofyear == i] = da_doy_avg.sel(dayofyear = 365)
        #    else:
        #        da.loc[da.time.dt.dayofyear == i] = da_doy_avg.sel(dayofyear = i)
        #del da_doy_avg
        count += 1

        #if count == 1:
        #    mean = da.copy(deep = True)
        #else:
        #    mean += da.copy(deep = True)
        mean += float(da.mean())
        del da
    mean = mean / count
                              
    print(f'Mean of {variable} found\n\n')
    print(mean)
    print('\n\n')
    sys.stdout.flush()
                              
    for i,output_path in enumerate(output_paths):
        print(output_path)
        sys.stdout.flush()
        
        da = xr.open_dataset(output_path)[variable]
        da = da * np.sqrt(s) - np.sqrt(s) * mean + mean
        da = da.load()

        #os.remove(output_path)
        temp_output_path = output_path.replace(variable, variable + '_rescaled') # remove later 
        da.to_netcdf(temp_output_path)
                    
        if i == 0:
            df = pd.read_csv('/rds/general/user/tk22/home/paper_2/bias_correction/'+
                             f'moment_test/{variable}_daily.csv')
            df = df.loc[df.member == 'obs']
            df.to_csv(('/rds/general/user/tk22/home/paper_2/bias_correction/'+
                           f'moment_test/{variable}_variance_scaled_{period}_'+
                           f'daily.csv'),
                           header = True, mode = 'w', index = False)
            del df
        
        mem = str(da.member.data[0])

        print(f'Member {mem} rescaled.')
        sys.stdout.flush()
                  
    return


def present_multiplicative_variance_corrected_run(variable = 'sfcwind', ttime = None):
    '''
    Input Variables:
    >   x_mh_a_hat:     x_oh_a_tilde * q_mh_a
    
    Processed Variables:
    >   s:              Scale parameter.
    
    Output Variable:
    >   x_mh_a_hat:     x_oh_a_tilde * q_mh_a
    '''
    
    # 6: Rescaling variance:
    rescale_variance(variable = variable, ttime = ttime)
    print('6: Variance rescaled in final output.')
    sys.stdout.flush()
    return                 
                         

def present_multiplicative_run(x_oh_a, x_mh_b, variable = 'pr', ttime = None):
    '''
    Input Variables:
    >   x_oh_a:         Variable data, observed historical, reanalysis grid.
    >   x_mh_b:         Variable data, modelled historical, ensemble grid.
    N.B. These must be data from a specific hour of day, e.g. 1400 local time.
    
    Processed Variables:
    >   x_oh_a_tilde:   Temporally smoothed x_oh_a, with a 15 days of year 
                        rolling window to either side.
    >   x_mh_b_base:    The mean of 'similar cells' on the x_mh_b grid. 'Similar 
                        cells' are defined as cells within a distance of:
                        gridscale*pi degrees and 15 days of year to either side.
    >   q_mh_b:         Delta from mean: x_mh_b / x_mh_b_base.
    >   q_mh_a:         Bilinear interpolation of d_mh_b onto the reanalysis 
                        grid.
    
    Output Variable:
    >   x_mh_a_hat:     x_oh_a_tilde * q_mh_a
    '''
    # 0: Clearing target directory:
    #paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
    #                  f'{variable}_*_h???_{ttime.strftime("%H%M")}.nc')
    #for path in paths:
    #    os.remove(path)
    #print('1: Cleared old data out.')
    #sys.stdout.flush()
                  
    # 1: Find x_oh_a_tilde from x_oh_a.
    x_oh_a_tilde = temporally_smooth_observation(x_oh_a)
    x_oh_a.close()
    print('1: Variable calculated: x_oh_a_tilde.')
    sys.stdout.flush()
    
    # 2: Find x_mh_b_base from x_mh_b.
    x_mh_b_base = similar_baseline_model(x_mh_b)
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mh_b and x_mh_b_base.
    x_mh_b.load()
    x_mh_b_base.load()
    print('Inputs loaded.')
    sys.stdout.flush()
    q_mh_b = x_mh_b / x_mh_b_base
    print('Delta calculated loaded.')
    sys.stdout.flush()
    q_mh_b.load()
    x_mh_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mh_b.')
    sys.stdout.flush()
    
    # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
    interpolate_q(q_mh_b, by = x_oh_a_tilde, ttime = ttime,
                  period = 'hist', var = variable)
    del q_mh_b
    print('4: Variable calculated: q_mh_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           ttime = ttime, period = 'hist', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mh_a_hat.')
    sys.stdout.flush()
    
    # 6: Tidying up directories (if complete):
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/tmp/?_mh_a_{variable}*_daily.nc')
    for path in paths:
        os.remove(path)
                      
    # 7: Making plots:
    if variable in ['snc','GPP']:
        target = 160
    else:
        target = 8*160
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_*.nc')
    if len(paths) == target:
        run_summary_figures(variable,'hist')
    
    #target = 8*160
    #paths = glob.glob(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_*.nc')
    #if len(paths) == target:
    #    members = list(np.unique([path.split('_')[-2] for path in paths]))
    #    for mem in members:
    #        ds = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/'+
    #                               f'bias_corrected/{variable}_{mem}_*.nc').load()
    #        ds.to_netcdf(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_{mem}.nc')
    #for path in paths:
    #    os.remove(path)
    #paths = glob.glob('/rds/general/user/tk22/ephemeral/tmp/*')
    #for path in paths:
    #    os.remove(path)
    
    return
    
    
def present_additive_run(x_oh_a, x_mh_b, variable = 'tas', ttime = None):
    '''
    Input Variables:
    >   x_oh_a:         Variable data, observed historical, reanalysis grid.
    >   x_mh_b:         Variable data, modelled historical, ensemble grid.
    N.B. These must be data from a specific hour of day, e.g. 1400 local time.
    
    Processed Variables:
    >   x_oh_a_tilde:   Temporally smoothed x_oh_a, with a 15 days of year 
                        rolling window to either side.
    >   x_mh_b_base:    The mean of 'similar cells' on the x_mh_b grid. 'Similar 
                        cells' are defined as cells within a distance of:
                        gridscale*pi degrees and 15 days of year to either side.
    >   d_mh_b:         Delta from mean: x_mh_b - x_mh_b_base.
    >   d_mh_a:         Bilinear interpolation of d_mh_b onto the reanalysis 
                        grid.
    
    Output Variable:
    >   x_mh_a_hat:     x_oh_a_tilde + d_mh_a
    '''
    # 0: Clearing target directory:
    #paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
    #                  f'{variable}_*_h???_{ttime.strftime("%H%M")}.nc')
    #for path in paths:
    #    os.remove(path)
    #print('1: Cleared old data out.')
    #sys.stdout.flush()
                      
    # 1: Find x_oh_a_tilde from x_oh_a.
    x_oh_a_tilde = temporally_smooth_observation(x_oh_a)
    x_oh_a.close()
    print('1: Variable calculated: x_oh_a_tilde.')
    sys.stdout.flush()
    
    # 2: Find x_mh_b_base from x_mh_b.
    x_mh_b_base = similar_baseline_model(x_mh_b)
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mh_b and x_mh_b_base.
    d_mh_b = x_mh_b - x_mh_b_base
    d_mh_b.load()
    x_mh_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: d_mh_b.')
    sys.stdout.flush()
    
    # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
    interpolate_d(d_mh_b, by = x_oh_a_tilde, ttime = ttime,
                  period = 'hist', var = variable)
    del d_mh_b
    print('4: Variable calculated: d_mh_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_d(x_mh_b, x_oh_a_tilde, variable = variable,
                           ttime = ttime, period = 'hist', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mh_a_hat.')
    sys.stdout.flush()
    
    # 6: Tidying up directories (if complete):
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/tmp/?_mh_a_{variable}*_daily.nc')
    for path in paths:
        os.remove(path)
                      
    # 7: Making plots:
    if variable in ['snc','GPP']:
        target = 160
    else:
        target = 8*160
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_*.nc')
    if len(paths) == target:
        run_summary_figures(variable,'hist')
                      
                      
    #target = 8*160
    #paths = glob.glob(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_*.nc')
    #if len(paths) == target:
    #    members = list(np.unique([path.split('_')[-2] for path in paths]))
    #    for mem in members:
    #        ds = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/'+
    #                               f'bias_corrected/{variable}_{mem}_*.nc').load()
    #        ds.to_netcdf(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_{mem}.nc')
    #for path in paths:
    #    os.remove(path)
    #paths = glob.glob('/rds/general/user/tk22/ephemeral/tmp/*')
    #for path in paths:
    #    os.remove(path)
    return


def future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b, variable = 'tas', ttime = None):
    '''
    Input Variables:
    >   x_oh_a:         Variable data, observed historical, reanalysis grid.
    >   x_mh_b:         Variable data, modelled historical, ensemble grid.
    >   x_mp_b:         Variable data, modelled projection, ensemble grid.
    N.B. These must be data from a specific hour of day, e.g. 1400 local time.
    
    Processed Variables:
    >   x_oh_a_tilde:   Temporally smoothed x_oh_a, with a 15 days of year 
                        rolling window to either side.
    >   x_mh_b_base:    The mean of 'similar cells' on the x_mh_b grid. 'Similar 
                        cells' are defined as cells within a distance of:
                        gridscale*pi degrees and 15 days of year to either side.
    >   d_mh_b:         Delta from mean: x_mh_b - x_mh_b_base.
    >   d_mh_a:         Bilinear interpolation of d_mh_b onto the reanalysis 
                        grid.
    
    Output Variable:
    >   x_mh_a_hat:     x_oh_a_tilde + d_mh_a
    '''
    # 0: Clearing target directory:
    #paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
    #                  f'{variable}_*_s???_{ttime.strftime("%H%M")}.nc')
    #for path in paths:
    #    os.remove(path)
    #print('1: Cleared old data out.')
    #sys.stdout.flush()
                      
    # 1: Find x_oh_a_tilde from x_oh_a.
    x_oh_a_tilde = temporally_smooth_observation(x_oh_a)
    x_oh_a_tilde['time'] = x_mp_b['time']
    x_oh_a.close()
    print('1: Variable calculated: x_oh_a_tilde.')
    sys.stdout.flush()
    
    # 2: Find x_mh_b_base from x_mh_b.
    x_mh_b['member'] = x_mp_b['member'] # i.e. so coordinates properly match.
    x_mh_b['time'] = x_mp_b['time']     # i.e. so coordinates properly match.
    x_mh_b_base = similar_baseline_model(x_mh_b)
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mp_b and x_mh_b_base (i.e. delta into the future)
    x_mp_b.load()
    x_mh_b_base.load()
    print('Inputs loaded.')
    q_mp_b = x_mp_b / x_mh_b_base
    print('Delta calculated loaded.')
    sys.stdout.flush()
    q_mp_b.load()
    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mp_b.')
    sys.stdout.flush()
    
    # 4: Find d_mp_a from d_mp_b.
    interpolate_q(q_mp_b, by = x_oh_a_tilde, ttime = ttime,
                  period = 'proj', var = variable)
    del q_mp_b
    print('4: Variable calculated: q_mp_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           ttime = ttime, period = 'proj', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mp_a_hat.')
    sys.stdout.flush()
    
    # 6: Tidying up directories (if complete):
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/tmp/?_mp_a_{variable}*_daily.nc')
    for path in paths:
        os.remove(path)
                      
    # 7: Making plots:
    if variable in ['snc','GPP']:
        target = 160
    else:
        target = 8*160
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_*.nc')
    if len(paths) == target:
        run_summary_figures(variable,'proj')
    
    return



def future_additive_run(x_oh_a, x_mh_b, x_mp_b, variable = 'tas', ttime = None):
    '''
    Input Variables:
    >   x_oh_a:         Variable data, observed historical, reanalysis grid.
    >   x_mh_b:         Variable data, modelled historical, ensemble grid.
    >   x_mp_b:         Variable data, modelled projection, ensemble grid.
    N.B. These must be data from a specific hour of day, e.g. 1400 local time.
    
    Processed Variables:
    >   x_oh_a_tilde:   Temporally smoothed x_oh_a, with a 15 days of year 
                        rolling window to either side.
    >   x_mh_b_base:    The mean of 'similar cells' on the x_mh_b grid. 'Similar 
                        cells' are defined as cells within a distance of:
                        gridscale*pi degrees and 15 days of year to either side.
    >   d_mh_b:         Delta from mean: x_mh_b - x_mh_b_base.
    >   d_mh_a:         Bilinear interpolation of d_mh_b onto the reanalysis 
                        grid.
    
    Output Variable:
    >   x_mh_a_hat:     x_oh_a_tilde + d_mh_a
    '''
    # 0: Clearing target directory:
    #paths = glob.glob('/rds/general/user/tk22/ephemeral/bias_corrected/'+
    #                  f'{variable}_*_s???_{ttime.strftime("%H%M")}.nc')
    #for path in paths:
    #    os.remove(path)
    #print('1: Cleared old data out.')
    #sys.stdout.flush()
                      
    # 1: Find x_oh_a_tilde from x_oh_a.
    x_oh_a_tilde = temporally_smooth_observation(x_oh_a)
    x_oh_a_tilde['time'] = x_mp_b['time']
    x_oh_a.close()
    print('1: Variable calculated: x_oh_a_tilde.')
    sys.stdout.flush()
    
    # 2: Find x_mh_b_base from x_mh_b.
    x_mh_b['member'] = x_mp_b['member'] # i.e. so coordinates properly match.
    x_mh_b['time'] = x_mp_b['time']
    x_mh_b_base = similar_baseline_model(x_mh_b)
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mp_b and x_mh_b_base (i.e. delta into the future)
    d_mp_b = x_mp_b - x_mh_b_base
    d_mp_b.load()
    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: d_mp_b.')
    sys.stdout.flush()
    
    # 4: Find d_mp_a from d_mp_b.
    interpolate_d(d_mp_b, by = x_oh_a_tilde, ttime = ttime, period = 'proj', var = variable)
    del d_mp_b
    print('4: Variable calculated: d_mp_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_d(x_mh_b, x_oh_a_tilde, variable = variable, ttime = ttime, period = 'proj', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mp_a_hat.')
    sys.stdout.flush()
    
    # 6: Tidying up directories (if complete):
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/tmp/?_mp_a_{variable}*_daily.nc')
    for path in paths:
        os.remove(path)

    # 7: Making plots:
    if variable in ['snc','GPP']:
        target = 160
    else:
        target = 8*160
    paths = glob.glob(f'/rds/general/user/tk22/ephemeral/bias_corrected/{variable}_*.nc')
    if len(paths) == target:
        run_summary_figures(variable,'proj')    
    return


def run_timestep(obs_path, ens_path_hist, ens_path_proj, variable = 'tas',
                 mode = 'add', period = 'hist', H = 12, M = 0):
    # 1: Load observation and ensemble data for only that time of day.
    ttime = dtt(H,M)
    ds1 = xr.open_dataset(obs_path)
    ds1 = ds1.sel(time = (ds1.time.dt.time == ttime))[variable]
    ds2 = xr.open_dataset(ens_path_hist)
    ds2 = ds2.sel(time = (ds2.time.dt.time == ttime))[variable]
    x_oh_a = ds1.copy(deep = True)
    x_mh_b = ds2.copy(deep = True)
    del ds1, ds2           
    print('Pre Loaded Data.\n\n')
    sys.stdout.flush()


    # 2: Run bias-correction:
    if period == 'hist':
        if mode == 'add':
            present_additive_run(x_oh_a, x_mh_b, variable = variable, ttime = ttime)
        elif mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            present_multiplicative_run(x_oh_a, x_mh_b, variable = variable, ttime = ttime)
    elif period == 'proj':
        ds3 = xr.open_dataset(ens_path_proj)
        ds3 = ds3.sel(time = (ds3.time.dt.time == ttime))[variable]
        x_mp_b = ds3.copy(deep = True)
        del ds3
        if mode == 'add':
            future_additive_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = ttime)
        if mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            x_mp_b = x_mp_b.clip(min = 0)
            future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = ttime)
    
    return


def overall_runner(variable = 'tas', period = 'hist', hour = 12):
    if variable == 'tas':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/tas_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/tas_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/tas_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'add', period = period,
                     H = hour, M = 0)
              
    if variable == 'das':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/das_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/das_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/das_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'add', period = period,
                     H = hour, M = 0)
            
    if variable == 'sfcwind':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/sfcwind_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/sfcwind_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/sfcwind_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'mult', period = period,
                     H = hour, M = 0)
        
    if variable == 'pr':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/pr_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/pr_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/pr_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'mult', period = period,
                     H = hour, M = 30)
        
    if variable == 'mrsos':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/mrsos_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/mrsos_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/mrsos_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'mult', period = period,
                     H = hour, M = 0)
              
    if variable == 'snc':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/snc_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/snc_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/snc_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'add', period = period,
                     H = hour, M = 0)
              
    if variable == 'GPP':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/GPP_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/GPP_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/GPP_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'add', period = period,
                     H = hour, M = 0)
        
    if variable == 'rlds':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/rlds_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/rlds_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/rlds_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'add', period = period,
                     H = hour, M = 30) #3hr is M= 30
              
    if variable == 'rsds':
        obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                    'wildfires_theo_keeping/live/ensemble_data/'+
                    'processing/rsds_19900101_20191231_era5.nc')
        ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/rsds_20000101_20091231_lentis.nc')
        ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                          'processing/rsds_20750101_20841231_lentis.nc')


        run_timestep(obs_path, ens_path_hist, ens_path_proj,
                     variable = variable, mode = 'add', period = period,
                     H = hour, M = 30) #3hr is M= 30

    return