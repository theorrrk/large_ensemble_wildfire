import xarray as xr
from datetime import time as dtt
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import glob
import random
import os
from xarray_einstats import stats as xr_stats
from scipy import stats as stats
import warnings
from matplotlib import pyplot as plt
from sklearn import metrics
from matplotlib import colors
import tempfile
import shutil
import datetime as dt


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
    x_mh_b.load()
    x_mh_b_base = x_mh_b.copy(deep = True)
    # Getting dayofyear average:
    doy_average = x_mh_b_base.groupby('time.dayofyear').mean()[:,:365,:,:] # Avoiding leap year undersampling
    # Smoothing with moving 31 day window:
    doy_average_smoothed = doy_average.copy(deep = True)
    for doy in list(doy_average.dayofyear.data):
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
    # Updating x_mh_b_base by dayofyear:
    for i in range(len(x_mh_b_base.time)):
        doy = int(x_mh_b_base.time[i].dt.dayofyear)
        if doy == 366:
            doy = 365 # Avoiding leap year undersampling
        doy_index = doy - 1
        x_mh_b_base[:,i,:,:] = doy_average_smoothed[:,doy_index,:,:]
    # Paranoid tidying:
    x_mh_b_base.load()
    doy_average.close()
    del doy_average, doy_average_smoothed, avg
    return x_mh_b_base


def interpolate_d(d_mh_b, by = None, period = 'hist', var = 'tas', hour = None):
    tmp = tempfile.gettempdir()
    for i, mem in enumerate(list(d_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        da = d_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                      longitude = by.longitude.data)
        da = da.expand_dims(dim = 'member')
        da.load()
        da.astype('float32').to_dataset()
        if period == 'hist':
            if hour == None:
                da.to_netcdf(f'{tmp}/d_mh_a_{var}_{mem}_{period}.nc')
            else:
                da.to_netcdf(f'{tmp}/d_mh_a_{var}_{mem}_{period}_{hour}.nc')
        elif period == 'proj':
            if hour == None:
                da.to_netcdf(f'{tmp}/d_mp_a_{var}_{mem}_{period}.nc')
            else:
                da.to_netcdf(f'{tmp}/d_mp_a_{var}_{mem}_{period}_{hour}.nc')
        del da
    return


def interpolate_d_by_mem(d_mh_b, by = None, period = 'hist', var = 'tas', hour = None, i = 0):
    tmp = tempfile.gettempdir()
    mem = list(d_mh_b.member.data)[i]
    da = d_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                  longitude = by.longitude.data)
    da = da.expand_dims(dim = 'member')
    da.load()
    da.astype('float32')
    return da


def interpolate_q(q_mh_b, by = None, period = 'hist', var = 'tas', hour = None):
    tmp = tempfile.gettempdir()
    for i, mem in enumerate(list(q_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        da = q_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                      longitude = by.longitude.data)
        da = da.expand_dims(dim = 'member')
        da.load()
        da.astype('float32').to_dataset()
        if period == 'hist':
            if hour == None:
                da.to_netcdf(f'{tmp}/q_mh_a_{var}_{mem}_{period}.nc')
            else:
                da.to_netcdf(f'{tmp}/q_mh_a_{var}_{mem}_{period}_{hour}.nc')
        elif period == 'proj':
            if hour == None:
                da.to_netcdf(f'{tmp}/q_mp_a_{var}_{mem}_{period}.nc')
            else:
                da.to_netcdf(f'{tmp}/q_mp_a_{var}_{mem}_{period}_{hour}.nc')
        del da
    return


def interpolate_q_by_mem(q_mh_b, by = None, period = 'hist', var = 'tas', hour = None, i = 0):
    tmp = tempfile.gettempdir()
    mem = list(q_mh_b.member.data)[i]

    da = q_mh_b[i,:,:,:].copy(deep = True).interp(latitude = by.latitude.data,
                                                  longitude = by.longitude.data)
    da = da.expand_dims(dim = 'member')
    da.load()
    da.astype('float32')
    return da


def bc_downscaled_output_d(x_mh_b, x_oh_a_tilde, variable = 'tas',
                           period = 'hist', var = 'tas', hour = None):
    tmp = tempfile.gettempdir()
    for i, mem in enumerate(list(x_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        if period == 'hist':
            if hour == None:
                da = xr.open_dataset(f'{tmp}/d_mh_a_{var}_{mem}_{period}.nc').load()
                os.remove(f'{tmp}/d_mh_a_{var}_{mem}_{period}.nc')
            else:
                da = xr.open_dataset(f'{tmp}/d_mh_a_{var}_{mem}_{period}_{hour}.nc').load()
                os.remove(f'{tmp}/d_mh_a_{var}_{mem}_{period}_{hour}.nc')            

        elif period == 'proj':
            if hour == None:
                da = xr.open_dataset(f'{tmp}/d_mp_a_{var}_{mem}_{period}.nc').load()
                os.remove(f'{tmp}/d_mp_a_{var}_{mem}_{period}.nc')
            else:
                da = xr.open_dataset(f'{tmp}/d_mp_a_{var}_{mem}_{period}_{hour}.nc').load()
                os.remove(f'{tmp}/d_mp_a_{var}_{mem}_{period}_{hour}.nc') 
        
        da = da[var]
        da = da + x_oh_a_tilde
                                 
        if variable == 'snc':
            da = da.clip(min = 0, max = 100)
                                 
        if variable == 'GPP':
            da = da.clip(min = 0)
                                 
        df = get_moments_ens(da, member = mem)
        if hour == None:
            df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                       f'moment_test/{variable}_{period}.csv'),
                       header = False, mode = 'a', index = False)
        else:
            df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                       f'moment_test/{variable}_{hour}_{period}.csv'),
                       header = False, mode = 'a', index = False)
        
        if period == 'hist':
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_historical_{mem}.nc')
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_{hour}_historical_{mem}.nc')
        elif period == 'proj':
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_projected_{mem}.nc')
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_{hour}_projected_{mem}.nc')
    return



def bc_downscaled_output_d_by_mem(da, x_mh_b, x_oh_a_tilde, variable = 'tas',
                                  period = 'hist', hour = None, i = 0):
    tmp = tempfile.gettempdir()
    mem = list(x_mh_b.member.data)[i]
    da = da + x_oh_a_tilde
    if variable == 'snc':
        da = da.clip(min = 0, max = 100)               
    if variable == 'GPP':
        da = da.clip(min = 0)                 
    df = get_moments_ens(da, member = mem)
    if hour == None:
        df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                   f'moment_test/{variable}_{period}.csv'),
                  header = False, mode = 'a', index = False)
    else:
        df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                   f'moment_test/{variable}_{hour}_{period}.csv'),
                  header = False, mode = 'a', index = False)
    return da




def bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = 'pr', period = 'hist',
                           var = 'tas', hour = None):
    tmp = tempfile.gettempdir()
    for i, mem in enumerate(list(x_mh_b.member.data)):
        print(f'\t{mem}')
        sys.stdout.flush()
        if period == 'hist':
            if hour == None:
                da = xr.open_dataset(f'{tmp}/q_mh_a_{var}_{mem}_{period}.nc').load()
                os.remove(f'{tmp}/q_mh_a_{var}_{mem}_{period}.nc')
            else:
                da = xr.open_dataset(f'{tmp}/q_mh_a_{var}_{mem}_{period}_{hour}.nc').load()
                os.remove(f'{tmp}/q_mh_a_{var}_{mem}_{period}_{hour}.nc') 
        elif period == 'proj':
            if hour == None:
                da = xr.open_dataset(f'{tmp}/q_mp_a_{var}_{mem}_{period}.nc').load()
                os.remove(f'{tmp}/q_mp_a_{var}_{mem}_{period}.nc')
            else:
                da = xr.open_dataset(f'{tmp}/q_mp_a_{var}_{mem}_{period}_{hour}.nc').load()
                os.remove(f'{tmp}/q_mp_a_{var}_{mem}_{period}_{hour}.nc') 
        
        da = da[var]
        da = da * x_oh_a_tilde
                                 
        if variable == 'snc':
            da = da.clip(min = 0, max = 100)
                                 
        if variable == 'GPP':
            da = da.clip(min = 0)
            
        df = get_moments_ens(da, member = mem)
        if hour == None:
            df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                       f'moment_test/{variable}_{period}.csv'),
                       header = False, mode = 'a', index = False)
        else:
            df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                       f'moment_test/{variable}_{hour}_{period}.csv'),
                       header = False, mode = 'a', index = False)
        
        if period == 'hist':
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_historical_{mem}.nc')
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_{hour}_historical_{mem}.nc')
        elif period == 'proj':
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_projected_{mem}.nc')
            if hour == None:
                da.astype('float32').to_netcdf(f'{tmp}/{variable}_{hour}_projected_{mem}.nc')
    return


def bc_downscaled_output_q_by_mem(da, x_mh_b, x_oh_a_tilde, variable = 'tas',
                                  period = 'hist', hour = None, i = 0):
    tmp = tempfile.gettempdir()
    mem = list(x_mh_b.member.data)[i]
    da = da * x_oh_a_tilde
    if variable == 'snc':
        da = da.clip(min = 0, max = 100)               
    if variable == 'GPP':
        da = da.clip(min = 0)                 
    df = get_moments_ens(da, member = mem)
    if hour == None:
        df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                   f'moment_test/{variable}_{period}.csv'),
                  header = False, mode = 'a', index = False)
    else:
        df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                   f'moment_test/{variable}_{hour}_{period}.csv'),
                  header = False, mode = 'a', index = False)
    return da


def get_moments_obs(x_oh_a):
    warnings.filterwarnings("ignore", category = RuntimeWarning) 
    lats = x_oh_a.latitude.to_numpy()
    lons = x_oh_a.longitude.to_numpy()
    clim_step = int(np.floor(0.7 * 2 * np.pi / 0.1))
    count = 0
    for lat_ind in range(int(np.ceil((np.max(lats) - np.min(lats)) / (0.1 * clim_step)))):
        for lon_ind in range(int(np.ceil((np.max(lons) - np.min(lons)) / (0.1 * clim_step)))):
            lat =  np.min(lats) + (lat_ind + 0.5) * clim_step * 0.1 #(lats[lat_ind*clim_step] + lats[(lat_ind+1)*clim_step])/2
            lon =  np.min(lons) + (lon_ind + 0.5) * clim_step * 0.1 #(lons[lon_ind*clim_step] + lons[(lon_ind+1)*clim_step])/2
            data = x_oh_a[:,lat_ind*clim_step:(lat_ind+1)*clim_step,lon_ind*clim_step:(lon_ind+1)*clim_step]
            for month in range(1,13,1):
                count += 1
                temp = data.isel(time = data.time.dt.month == month)
                mean = float(temp.mean())
                vari = float(temp.std()) ** 2
                skew = float(xr_stats.skew(temp, nan_policy = 'omit'))
                kurt = float(xr_stats.kurtosis(temp, nan_policy = 'omit'))
                appendage = ['obs', lat, lon, month, mean, vari, skew, kurt]
                if count == 1:
                    df = pd.DataFrame(data = {'member': 'obs', 'lat': lat,
                                              'lon': lon, 'month': month,
                                              'mean': mean, 'var': vari,
                                              'skew': skew, 'kurt': kurt},
                                      index = [0])

                else:
                    df.loc[len(df.index)] = appendage
    return df

                         
def get_moments_ens(da, member = 'h010'):
    # ONLY WORKS FOR ONE MEMBER:
    warnings.filterwarnings("ignore", category = RuntimeWarning) 
    lats = da.latitude.to_numpy()
    lons = da.longitude.to_numpy()

    clim_step = int(np.floor(0.7 * 2 * np.pi / 0.1))
    count = 0
    for lat_ind in range(int(np.ceil((np.max(lats) - np.min(lats)) / (0.1 * clim_step)))):
        for lon_ind in range(int(np.ceil((np.max(lons) - np.min(lons)) / (0.1 * clim_step)))):
            lat =  np.min(lats) + (lat_ind + 0.5) * clim_step * 0.1 #(lats[lat_ind*clim_step] + lats[(lat_ind+1)*clim_step])/2
            lon =  np.min(lons) + (lon_ind + 0.5) * clim_step * 0.1 #(lons[lon_ind*clim_step] + lons[(lon_ind+1)*clim_step])/2
            data = da[:,lat_ind*clim_step:(lat_ind+1)*clim_step,lon_ind*clim_step:(lon_ind+1)*clim_step]
            for month in range(1,13,1):
                count += 1
                temp = data.isel(time = data.time.dt.month == month)
                mean = float(temp.mean())
                vari = float(temp.std()) ** 2
                skew = float(xr_stats.skew(temp, nan_policy = 'omit'))
                kurt = float(xr_stats.kurtosis(temp, nan_policy = 'omit'))
                appendage = [member, lat, lon, month, mean, vari, skew, kurt]
                if count == 1:
                    df = pd.DataFrame(data = {'member': member, 'lat': lat,
                                              'lon': lon, 'month': month,
                                              'mean': mean, 'var': vari,
                                              'skew': skew, 'kurt': kurt},
                                      index = [0])
                else:
                    df.loc[len(df.index)] = appendage
    return df


def rescale_variance(da, variable = 'sfcwind'):
    df = pd.read_csv('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                     f'final_moments/{variable}.csv')
    df_obs = df.loc[df.member == 'obs']
    df_ens = df.loc[[x[0]=='h' for x in df.member]]
    df_ens_sample = df_ens.groupby(['lat','lon','month']).sample(1)

    df_obs = df_obs.sort_values(['lat','lon','month'])
    df_ens_sample = df_ens_sample.sort_values(['lat','lon','month'])

    m = np.nanmean(df_ens['mean'])
    old_s = np.sqrt(np.nanmean(df_ens['var']))
    new_s = np.sqrt(np.nanmean(df_obs['var']))

    da = da.astype('float32').load()
    da = (da - m).load()
    da = (da / old_s).load()
    da = (da * new_s).load()
    da = (da + m).load()
    da = da.astype('float32').load()
    return da


def to_local_time(ds, variable):
    print('Converting to local time:')
    sys.stdout.flush()
    ds = ds.load()
    ds = ds.transpose('time','latitude','longitude')
    for i,lon in enumerate(ds.longitude.to_numpy()):
        # Finding timeszone offset:
        tz_offset = lon/15 # in hours
        if i%25 == 0:
            print(f'{i}/{len(ds.longitude.to_numpy())}: At longitude {lon:.1f}°,'+
                  f' timezone offset is {tz_offset:.2f} hours')
            sys.stdout.flush()
        # Copying slice of data-array:
        da = ds[variable][:,:,i].copy(deep = True)
        # Converting to local timezone:
        da['time'] = da.time.to_numpy() + pd.Timedelta(hours = tz_offset)
        # Interpolating to original grid (now local):
        da = da.interp(time = ds.time.data,
                       method = 'nearest')
        # Updating dataset with local values:
        ds[variable][:,:,i] = da
    ds.load()
    return ds[variable]


def load_hourly_member(variable = 'das', period = 'historical', mem_ind = 3):
    for i,h in enumerate(['0','3','6','9','12','15','18','21']):
        print(f'Building data-array, step {i+1} of 8.')
        da = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/tmp/'+
                             f'{variable}_{h}_{period}_{mem_ind}.nc').load()

        if i == 0:
            arr = da[variable].astype('float32').to_numpy()[:,:,np.newaxis,:,:]
            members = da.member.to_numpy()
            dates = da.time.to_numpy() - pd.Timedelta(f'{h}h')
            hours = pd.timedelta_range(start="0h", freq="3h", periods=8)
            lats = da.latitude.to_numpy()
            lons = da.longitude.to_numpy()
        else:
            arr = np.concatenate([arr,da[variable].astype('float32').to_numpy()[:,:,np.newaxis,:,:]],
                                 axis = 2)
        da.close()
        del da

    ds = xr.Dataset(data_vars = {variable: (['member','date','hour','latitude','longitude'],arr)},
                    coords = {'member': members,
                              'date': dates,
                              'hour': hours,
                              'latitude': lats,
                              'longitude': lons})

    ds = ds.assign_coords(valid_time = ds.date + ds.hour)
    ds = ds.stack(datetime = ("date", "hour"))
    ds = (ds.drop_vars("datetime")
          .rename_dims({"datetime": "time"})
          .rename_vars({"valid_time": "time"}))
    ds = ds.transpose('member','time','latitude','longitude')
    return ds

def arden_buck(T):
    P = np.zeros_like(T)
    P[T >= 0] = 611.21 * np.exp((18.678 - T[T >= 0]/234.5)*(T[T >= 0]/(257.14 + T[T >= 0])))
    P[T < 0]  = 611.15 * np.exp((23.036 - T[T < 0]/333.7)*(T[T < 0]/(279.82 + T[T < 0])))
    return P

def make_vpd(T, D):
    sys.stdout.flush()
    T = T - 273.15
    P = arden_buck(T)
    del T
    D = D - 273.15
    Ps = arden_buck(D)
    del D
    VPD = P - Ps
    sys.stdout.flush()
    del Ps, P
    return VPD

def to_local_time(ds, variable = 'tas'):
    for i,lon in enumerate(ds.longitude.to_numpy()):
        # Finding timeszone offset:
        tz_offset = lon/15 # in hours
        if i%25 == 0:
            print(f'{i}/{len(ds.longitude.to_numpy())}: At longitude {lon:.1f}°,'+
                  f' timezone offset is {tz_offset:.2f} hours')
        # Copying slice of data-array:
        da = ds[variable][:,:,:,i].copy(deep = True)
        # Converting to local timezone:
        da['time'] = da.time.to_numpy() + pd.Timedelta(hours = tz_offset)
        # Interpolating to original grid (now local):
        da = da.interp(time = ds.time.data,
                       method = 'nearest')
        # Updating dataset with local values:
        ds[variable][:,:,:,i] = da
    ds['time'] = pd.date_range(start = ds.time.to_numpy()[0],
                               periods = len(ds.time), freq = '3H')
    return ds

def dryness_runner(period = 'historical', mem_ind = 0):
    tmp_dir = '/rds/general/user/tk22/ephemeral/tmp'
    # Loading inputs:
    ds1 = load_hourly_member(variable = 'das',
                             period = period,
                             mem_ind = mem_ind)
    ds1 = to_local_time(ds1, variable = 'das')
    ds2 = load_hourly_member(variable = 'tas',
                             period = period,
                             mem_ind = mem_ind)
    ds2 = to_local_time(ds2, variable = 'tas')
    # Building DTR and saving to tmp:
    ds = (ds2['tas'].resample(time = 'D').max() - 
          ds2['tas'].resample(time = 'D').min())
    ds['time'] = ds['time'].to_numpy() + pd.Timedelta(hours = 12)
    ds = ds.to_dataset().rename({'tas':'dtr'})
    ds = ds.transpose('member','time','latitude','longitude')
    ds['dtr'] = ds['dtr'].astype('float32')
    ds.to_netcdf(f'{tmp_dir}/dtr_{period}_{mem_ind}.nc')
    ds.close()
    # Building VPD and saving to tmp:
    ds = ds2.copy(deep = True)
    ds = ds.rename({'tas':'vpd'})
    for i in range(len(ds.longitude)):
        if i%25 == 0:
            print(f'{i}/{len(ds.longitude)} of VPD calculated.')
        D = ds1['das'][:,:,:,i].to_numpy()
        T = ds2['tas'][:,:,:,i].to_numpy()
        ds['vpd'][:,:,:,i] = make_vpd(T, D)
    ds = ds.sel(time = np.logical_and(ds.time.dt.time > dt.time(4,0),
                                      ds.time.dt.time < dt.time(20,0)))
    ds = ds.resample(time = 'D').mean()
    ds = ds.transpose('member','time','latitude','longitude')
    ds['vpd'] = ds['vpd'].astype('float32')
    ds.to_netcdf(f'{tmp_dir}/vpd_{period}_{mem_ind}.nc')
    ds.close()
    # Removing the input datasets:
    for h in [0,3,6,9,12,15,18,21]:
        os.remove(f'/rds/general/user/tk22/ephemeral/tmp/'+
                  f'tas_{h}_{period}_{mem_ind}.nc')
        os.remove(f'/rds/general/user/tk22/ephemeral/tmp/'+
                  f'das_{h}_{period}_{mem_ind}.nc')
    return


def wind_runner(period = 'historical', mem_ind = 0):
    tmp_dir = '/rds/general/user/tk22/ephemeral/tmp'
    # Loading inputs:
    ds = load_hourly_member(variable = 'sfcwind',
                             period = period,
                             mem_ind = mem_ind)
    ds = to_local_time(ds, variable = 'sfcwind')
    ds = ds.sel(time = np.logical_and(ds.time.dt.time > dt.time(4,0),
                                      ds.time.dt.time < dt.time(20,0)))
    ds = ds.resample(time = 'D').mean()
    ds = ds.transpose('member','time','latitude','longitude')
    ds['sfcwind'] = ds['sfcwind'].astype('float32')
    ds.to_netcdf(f'{tmp_dir}/sfcwind_{period}_{mem_ind}.nc')
    ds.close()
    # Removing the input datasets:
    for h in [0,3,6,9,12,15,18,21]:
        os.remove(f'/rds/general/user/tk22/ephemeral/tmp/'+
                  f'sfcwind_{h}_{period}_{mem_ind}.nc')
    return


def final_moments(var = 'vpd'):
    mask = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                           'wildfires_theo_keeping/live/ensemble_data/'+
                           'mask_conus_2000_2009.nc').mask.to_numpy()
    
    re_path = min(glob.glob('/rds/general/user/tk22/projects/'+
                            'leverhulme_wildfires_theo_keeping/'+
                            f'live/ensemble_data/*{var}*.nc'), key = len)
    ens_path = min(glob.glob('/rds/general/user/tk22/ephemeral/'+
                             'ensemble_predictors/downscaled/'+
                             f'*{var}*historical.nc'), key = len)

    da_re  = xr.open_dataset(re_path)[var]
    da_ens = xr.open_dataset(ens_path)[var]

    da_re = da_re.load()
    df = get_moments_obs(da_re*mask)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'final_moments/{var}.csv'),
              header = True, mode = 'w', index = False)
    da_re.close()

    for i,mem in enumerate(da_ens.member.to_numpy()):
        temp = da_ens[i,:,:,:].copy(deep = True).load()
        df = get_moments_ens(temp*mask, member = mem)
        df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                   f'final_moments/{var}.csv'),
                  header = False, mode = 'a', index = False)
        temp.close()
        da_ens.close()
    return


if __name__ == '__main__':
    final_moments(var = sys.argv[1])