from bc_funcs import *


def precipitation_historical(variable = 'pr'):
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
    # 0: Load data:
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')[variable]    
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_hist.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    q_mh_b = x_mh_b / x_mh_b_base
    q_mh_b.load()
    x_mh_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mh_b.')
    sys.stdout.flush()
    
    # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
    interpolate_q(q_mh_b, by = x_oh_a_tilde,
                  period = 'hist', var = variable)
    del q_mh_b
    print('4: Variable calculated: q_mh_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           period = 'hist', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mh_a_hat.')
    sys.stdout.flush()
    
    # 6: Final output:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_historical_*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    # Saving Daily Precip:
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/pr_historical.nc')
    # Building 5-Day Precip:
    ds = ds.rename({'pr': 'pr_5d'})
    for i in range(160):
        temp = ds['pr_5d'][i,:,:,:].to_numpy()
        # Add 5 days from end of 2000 to beginning:
        temp = np.concatenate([temp[366-5:366,:,:],temp],axis = 0)
        temp = np.cumsum(temp, axis = 0)
        temp = (temp[5:,:,:] - temp[:-5,:,:]) / 5
        ds['pr_5d'][i,:,:,:] = temp
    ds['pr_5d'] = ds['pr_5d'].astype('float32')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/pr_5d_historical.nc')
    print('6: Final arrays saved to ephemeral.')
    sys.stdout.flush()
    return


def precipitation_projected(variable = 'pr'):
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
    # 0: Load data:
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    proj_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20750101_20841231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    shutil.copy(proj_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')[variable] 
    x_mp_b = xr.open_dataset(f'{tmp}/{variable}_20750101_20841231_lentis.nc')[variable]
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_proj.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    q_mp_b = x_mp_b / x_mh_b_base
    q_mp_b.load()
    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mp_b.')
    sys.stdout.flush()
    
    # 4: Find d_mp_a from d_mp_b.
    interpolate_q(q_mp_b, by = x_oh_a_tilde,
                  period = 'proj', var = variable)
    del q_mp_b
    print('4: Variable calculated: q_mp_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           period = 'proj', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mp_a_hat.')
    sys.stdout.flush()
    
    # 6: Final output:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_projected_*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    # Saving Daily Precip:
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/pr_projected.nc')
    # Building 5-Day Precip:
    ds = ds.rename({'pr': 'pr_5d'})
    for i in range(160):
        temp = ds['pr_5d'][i,:,:,:].to_numpy()
        # Add 5 days from end of 2075 to beginning:
        temp = np.concatenate([temp[365-5:365,:,:],temp],axis = 0)
        temp = np.cumsum(temp, axis = 0)
        temp = (temp[5:,:,:] - temp[:-5,:,:]) / 5
        ds['pr_5d'][i,:,:,:] = temp
    ds['pr_5d'] = ds['pr_5d'].astype('float32')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/pr_5d_projected.nc')
    print('6: Final arrays saved to ephemeral.')
    sys.stdout.flush()
    return


def windspeed_historical(variable = 'sfcwind', hour = 0):
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
    # 0: Load data:
    ttime = dtt(hour,0)
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')
    x_oh_a = x_oh_a.sel(time = (x_oh_a.time.dt.time == ttime))[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')
    x_mh_b = x_mh_b.sel(time = (x_mh_b.time.dt.time == ttime))[variable]
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_{hour}_hist.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    q_mh_b = x_mh_b / x_mh_b_base
    q_mh_b.load()
    x_mh_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mh_b.')
    sys.stdout.flush()
    
    for i in range(160):
        # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
        da = interpolate_q_by_mem(q_mh_b, by = x_oh_a_tilde, period = 'hist',
                                  var = variable, hour = hour, i = i)
        #del d_mh_b
        print(f'4.{i}: Variable calculated: q_mh_a.')
        sys.stdout.flush()
    
        # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
        da = bc_downscaled_output_q_by_mem(da, x_mh_b, x_oh_a_tilde, variable = variable,
                                           period = 'hist', hour = hour, i = i)

        #del x_oh_a_tilde
        print(f'5.{i}: Variable calculated: x_mh_a_hat.')
        sys.stdout.flush()
    
        # 6: Final output:
        tmp = tempfile.gettempdir()
        da = da.astype('float32').to_dataset()
        da = da.transpose('member','time','latitude','longitude')
        da.to_netcdf('/rds/general/user/tk22/ephemeral/tmp/'+
                     f'{variable}_{hour}_historical_{i}.nc')
        print(f'6.{i}: Final arrays saved to ephemeral.')
        sys.stdout.flush()
    return


def windspeed_projected(variable = 'sfcwind', hour = 0):
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
    # 0: Load data:
    ttime = dtt(hour,0)
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    proj_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20750101_20841231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    shutil.copy(proj_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')
    x_oh_a = x_oh_a.sel(time = (x_oh_a.time.dt.time == ttime))[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')
    x_mh_b = x_mh_b.sel(time = (x_mh_b.time.dt.time == ttime))[variable]
    x_mp_b = xr.open_dataset(f'{tmp}/{variable}_20750101_20841231_lentis.nc')
    x_mp_b = x_mp_b.sel(time = (x_mp_b.time.dt.time == ttime))[variable]
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_{hour}_proj.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    q_mp_b = x_mp_b / x_mh_b_base
    q_mp_b.load()
    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mp_b.')
    sys.stdout.flush()
    
    for i in range(160):
        # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
        da = interpolate_q_by_mem(q_mp_b, by = x_oh_a_tilde, period = 'proj',
                                  var = variable, hour = hour, i = i)
        #del d_mh_b
        print(f'4.{i}: Variable calculated: q_mh_a.')
        sys.stdout.flush()
    
        # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
        da = bc_downscaled_output_q_by_mem(da, x_mh_b, x_oh_a_tilde, variable = variable,
                                           period = 'proj', hour = hour, i = i)

        #del x_oh_a_tilde
        print(f'5.{i}: Variable calculated: x_mh_a_hat.')
        sys.stdout.flush()
    
        # 6: Final output:
        tmp = tempfile.gettempdir()
        da = da.astype('float32').to_dataset()
        da = da.transpose('member','time','latitude','longitude')
        da.to_netcdf('/rds/general/user/tk22/ephemeral/tmp/'+
                     f'{variable}_{hour}_projected_{i}.nc')
        print(f'6.{i}: Final arrays saved to ephemeral.')
        sys.stdout.flush()


    return


def snow_cover_historical(variable = 'snc'):
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
    # 0: Load data:
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')[variable]    
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_hist.csv'),
              header = True, mode = 'w', index = False)
    del df
    
    # 1: Find x_oh_a_tilde from x_oh_a.
    x_oh_a_tilde = temporally_smooth_observation(x_oh_a)
    x_oh_a.close()
    print('1: Variable calculated: x_oh_a_tilde.')
    sys.stdout.flush()
    
    # 2: Find x_mh_b_base from x_mh_b.
    x_mh_b_base = similar_baseline_model(x_mh_b)
    x_mh_b_base = x_mh_b_base.clip(0.000001)
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mh_b and x_mh_b_base.
    q_mh_b = x_mh_b / x_mh_b_base
    q_mh_b.load()
    x_mh_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mh_b.')
    sys.stdout.flush()
    
    # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
    interpolate_q(q_mh_b, by = x_oh_a_tilde,
                  period = 'hist', var = variable)
    del q_mh_b
    print('4: Variable calculated: q_mh_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           period = 'hist', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mh_a_hat.')
    sys.stdout.flush()
    
    # 6: Final output:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_historical_*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    # Saving Daily Snow:
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/snc_historical.nc')
    print('6: Final arrays saved to ephemeral.')
    sys.stdout.flush()
    return


def snow_cover_projected(variable = 'snc'):
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
    # 0: Load data:
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    proj_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20750101_20841231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    shutil.copy(proj_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')[variable] 
    x_mp_b = xr.open_dataset(f'{tmp}/{variable}_20750101_20841231_lentis.nc')[variable]
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_proj.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    x_mh_b_base = x_mh_b_base.clip(0.000001)
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mp_b and x_mh_b_base (i.e. delta into the future)
    q_mp_b = x_mp_b / x_mh_b_base
    q_mp_b.load()
    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mp_b.')
    sys.stdout.flush()
    
    # 4: Find d_mp_a from d_mp_b.
    interpolate_q(q_mp_b, by = x_oh_a_tilde,
                  period = 'proj', var = variable)
    del q_mp_b
    print('4: Variable calculated: q_mp_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           period = 'proj', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mp_a_hat.')
    sys.stdout.flush()
    
    # 6: Final output:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_projected*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    # Saving Daily Snow:
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/snc_projected.nc')
    print('6: Final arrays saved to ephemeral.')
    sys.stdout.flush()
    return


def gpp_historical(variable = 'gpp'):
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
    # 0: Load data:
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')[variable]    
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_hist.csv'),
              header = True, mode = 'w', index = False)
    del df
    
    # 1: Find x_oh_a_tilde from x_oh_a.
    x_oh_a_tilde = temporally_smooth_observation(x_oh_a)
    x_oh_a.close()
    print('1: Variable calculated: x_oh_a_tilde.')
    sys.stdout.flush()
    
    # 2: Find x_mh_b_base from x_mh_b.
    x_mh_b_base = similar_baseline_model(x_mh_b)
    x_mh_b_base = x_mh_b_base.clip(min = 0.0001) # Clipping so no div0 error
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mh_b and x_mh_b_base.
    q_mh_b = x_mh_b / x_mh_b_base
    q_mh_b.load()
    x_mh_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mh_b.')
    sys.stdout.flush()
    
    # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
    interpolate_q(q_mh_b, by = x_oh_a_tilde,
                  period = 'hist', var = variable)
    del q_mh_b
    print('4: Variable calculated: q_mh_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           period = 'hist', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mh_a_hat.')
    sys.stdout.flush()
    
    # 6: Final output:
    # Building and saving GPP_50d:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_historical_*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    ds = ds.rename({'gpp': 'GPP_50d'})
    for i in range(160):
        temp = ds['GPP_50d'][i,:,:,:].to_numpy()
        # Add 50 days from end of 2000 to beginning:
        temp = np.concatenate([temp[366-50:366,:,:],temp],axis = 0)
        temp = np.cumsum(temp, axis = 0)
        temp = (temp[50:,:,:] - temp[:-50,:,:]) / 50
        ds['GPP_50d'][i,:,:,:] = temp
    ds['GPP_50d'] = ds['GPP_50d'].astype('float32')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/GPP_50d_historical.nc')
    ds.close()
    del ds, temp
    
    # Building and saving GPP_1yr:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_historical_*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    ds = ds.rename({'gpp': 'GPP_1yr'})
    for i in range(160):
        temp = ds['GPP_1yr'][i,:,:,:].to_numpy()
        # Add 365 days from end of 2000 to beginning:
        temp = np.concatenate([temp[366-365:366,:,:],temp],axis = 0)
        temp = np.cumsum(temp, axis = 0)
        temp = (temp[365:,:,:] - temp[:-365,:,:]) / 365
        ds['GPP_1yr'][i,:,:,:] = temp
    ds['GPP_1yr'] = ds['GPP_1yr'].astype('float32')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/GPP_1yr_historical.nc')
    ds.close()
    del ds, temp
    
    print('6: Final arrays saved to ephemeral.')
    sys.stdout.flush()
    return


def gpp_projected(variable = 'gpp'):
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
    # 0: Load data:
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    proj_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20750101_20841231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    shutil.copy(proj_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')[variable] 
    x_mp_b = xr.open_dataset(f'{tmp}/{variable}_20750101_20841231_lentis.nc')[variable]
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_proj.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    x_mh_b_base = x_mh_b_base.clip(min = 0.0001) # Clipping so no div0 error
    print('2: Variable calculated: x_mh_b_base.')
    sys.stdout.flush()
    
    # 3: Find d_mh_b from x_mp_b and x_mh_b_base (i.e. delta into the future)
    q_mp_b = x_mp_b / x_mh_b_base
    q_mp_b.load()
    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: q_mp_b.')
    sys.stdout.flush()
    
    # 4: Find d_mp_a from d_mp_b.
    interpolate_q(q_mp_b, by = x_oh_a_tilde,
                  period = 'proj', var = variable)
    del q_mp_b
    print('4: Variable calculated: q_mp_a.')
    sys.stdout.flush()
    
    # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
    bc_downscaled_output_q(x_mh_b, x_oh_a_tilde, variable = variable,
                           period = 'proj', var = variable)
    del x_oh_a_tilde
    print('5: Variable calculated: x_mp_a_hat.')
    sys.stdout.flush()
    
    # 6: Final output:
    # Building and saving GPP_50d:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_projected_*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    ds = ds.rename({'gpp': 'GPP_50d'})
    for i in range(160):
        temp = ds['GPP_50d'][i,:,:,:].to_numpy()
        # Add 50 days from end of 2075 to beginning:
        temp = np.concatenate([temp[365-50:365,:,:],temp],axis = 0)
        temp = np.cumsum(temp, axis = 0)
        temp = (temp[50:,:,:] - temp[:-50,:,:]) / 50
        ds['GPP_50d'][i,:,:,:] = temp
    ds['GPP_50d'] = ds['GPP_50d'].astype('float32')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/GPP_50d_projected.nc')
    ds.close()
    del ds, temp
    
    # Building and saving GPP_1yr:
    tmp = tempfile.gettempdir()
    ds = xr.open_mfdataset(f'{tmp}/{variable}_projected_*.nc').load()
    ds[variable] = ds[variable].astype('float32')
    ds = ds.transpose('member','time','latitude','longitude')
    ds = ds.rename({'gpp': 'GPP_1yr'})
    for i in range(160):
        temp = ds['GPP_1yr'][i,:,:,:].to_numpy()
        # Add 365 days from end of 2075 to beginning:
        temp = np.concatenate([temp[365-365:365,:,:],temp],axis = 0)
        temp = np.cumsum(temp, axis = 0)
        temp = (temp[365:,:,:] - temp[:-365,:,:]) / 365
        ds['GPP_1yr'][i,:,:,:] = temp
    ds['GPP_1yr'] = ds['GPP_1yr'].astype('float32')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/GPP_1yr_projected.nc')
    ds.close()
    del ds, temp
    
    print('6: Final arrays saved to ephemeral.')
    sys.stdout.flush()
    return


def templike_historical(variable = 'tas', hour = 0):
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
    # 0: Load data:
    ttime = dtt(hour,0)
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')
    x_oh_a = x_oh_a.sel(time = (x_oh_a.time.dt.time == ttime))[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')
    x_mh_b = x_mh_b.sel(time = (x_mh_b.time.dt.time == ttime))[variable]
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_{hour}_hist.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    
    for i in range(160):
        # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
        da = interpolate_d_by_mem(d_mh_b, by = x_oh_a_tilde, period = 'hist',
                                  var = variable, hour = hour, i = i)
        #del d_mh_b
        print(f'4.{i}: Variable calculated: d_mh_a.')
        sys.stdout.flush()
    
        # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
        da = bc_downscaled_output_d_by_mem(da, x_mh_b, x_oh_a_tilde, variable = variable,
                                           period = 'hist', hour = hour, i = i)

        #del x_oh_a_tilde
        print(f'5.{i}: Variable calculated: x_mh_a_hat.')
        sys.stdout.flush()
    
        # 6: Final output:
        tmp = tempfile.gettempdir()
        da = da.astype('float32').to_dataset()
        da = da.transpose('member','time','latitude','longitude')
        da.to_netcdf('/rds/general/user/tk22/ephemeral/tmp/'+
                     f'{variable}_{hour}_historical_{i}.nc')
        print(f'6.{i}: Final arrays saved to ephemeral.')
        sys.stdout.flush()
    return


def templike_projected(variable = 'tas', hour = 0):
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
    # 0: Load data:
    ttime = dtt(hour,0)
    re_path = ('/rds/general/user/tk22/projects/leverhulme_'+
               'wildfires_theo_keeping/live/ensemble_data/'+
               f'processing/{variable}_19900101_20191231_era5.nc')
    re_file = re_path.split('processing/')[1]
    hist_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20000101_20091231_lentis.nc')
    proj_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_data/'+
                 f'processing/{variable}_20750101_20841231_lentis.nc')
    tmp = tempfile.gettempdir()
    shutil.copy(re_path, tmp)
    shutil.copy(hist_path, tmp)
    shutil.copy(proj_path, tmp)
    x_oh_a = xr.open_dataset(f'{tmp}/{re_file}')
    x_oh_a = x_oh_a.sel(time = (x_oh_a.time.dt.time == ttime))[variable]
    x_mh_b = xr.open_dataset(f'{tmp}/{variable}_20000101_20091231_lentis.nc')
    x_mh_b = x_mh_b.sel(time = (x_mh_b.time.dt.time == ttime))[variable]
    x_mp_b = xr.open_dataset(f'{tmp}/{variable}_20750101_20841231_lentis.nc')
    x_mp_b = x_mp_b.sel(time = (x_mp_b.time.dt.time == ttime))[variable]
    df = get_moments_obs(x_oh_a)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'moment_test/{variable}_{hour}_proj.csv'),
              header = True, mode = 'w', index = False)
    del df
    
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
    
    # 3: Find d_mp_b from x_mp_b and x_mh_b_base (i.e. delta into the future)
    d_mp_b = x_mp_b - x_mh_b_base
    d_mp_b.load()
    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base
    print('3: Variable calculated: d_mp_b.')
    sys.stdout.flush()
    
    for i in range(160):
        # 4: Find d_mh_a from d_mh_b. (Calling it: x_mh_a_hat to reduce data use.)
        da = interpolate_d_by_mem(d_mp_b, by = x_oh_a_tilde, period = 'proj',
                                  var = variable, hour = hour, i = i)
        #del d_mh_b
        print(f'4.{i}: Variable calculated: d_mp_a.')
        sys.stdout.flush()
    
        # 5: Find x_mh_a_hat from x_oh_a_tilde and d_mh_a.
        da = bc_downscaled_output_d_by_mem(da, x_mh_b, x_oh_a_tilde, variable = variable,
                                           period = 'proj', hour = hour, i = i)

        #del x_oh_a_tilde
        print(f'5.{i}: Variable calculated: x_mh_a_hat.')
        sys.stdout.flush()
    
        # 6: Final output:
        tmp = tempfile.gettempdir()
        da = da.astype('float32').to_dataset()
        da = da.transpose('member','time','latitude','longitude')
        da.to_netcdf('/rds/general/user/tk22/ephemeral/tmp/'+
                     f'{variable}_{hour}_projected_{i}.nc')
        print(f'6.{i}: Final arrays saved to ephemeral.')
        sys.stdout.flush()
    return


def rescale_sfcwind():
    variable = 'sfcwind'
    ds = xr.open_dataset('/rds/general/user/tk22/ephemeral/ensemble_'+
                         'predictors/downscaled/sfcwind_historical.nc')
    for i,mem in enumerate(ds.member.to_numpy()):
        da = ds['sfcwind'][i,:,:,:].copy(deep = True)
        da = rescale_variance(da, variable = 'sfcwind')
        da = da.expand_dims(dim = 'member')
        da.to_netcdf('/rds/general/user/tk22/ephemeral/tmp/'+
                     f'{variable}_historical_{mem}.nc')
        da.close()
        ds.close()
    ds = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/tmp/'+
                           f'{variable}_historical_*.nc')
    ds['sfcwind'] = ds['sfcwind'].astype('float32').load()
    os.remove('/rds/general/user/tk22/ephemeral/ensemble_'+
              'predictors/downscaled/sfcwind_historical.nc')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/sfcwind_historical.nc')
    for path in glob.glob(f'/rds/general/user/tk22/ephemeral/tmp/{variable}_historical_*.nc'):
        os.remove(path)

    
    ds = xr.open_dataset('/rds/general/user/tk22/ephemeral/ensemble_'+
                         'predictors/downscaled/sfcwind_projected.nc')
    for i,mem in enumerate(ds.member.to_numpy()):
        da = ds['sfcwind'][i,:,:,:].copy(deep = True)
        da = rescale_variance(da, variable = 'sfcwind')
        da = da.expand_dims(dim = 'member')
        da.to_netcdf('/rds/general/user/tk22/ephemeral/tmp/'+
                     f'{variable}_projected_{mem}.nc')
        da.close()
        ds.close()
    ds = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/tmp/'+
                           f'{variable}_projected_*.nc')
    ds['sfcwind'] = ds['sfcwind'].astype('float32').load()
    os.remove('/rds/general/user/tk22/ephemeral/ensemble_'+
              'predictors/downscaled/sfcwind_projected.nc')
    ds.to_netcdf('/rds/general/user/tk22/ephemeral/ensemble_'+
                 'predictors/downscaled/sfcwind_projected.nc')
    for path in glob.glob(f'/rds/general/user/tk22/ephemeral/tmp/{variable}_projected_*.nc'):
        os.remove(path)
    return

def build_dtr():
    return

def build_vpd():
    return

if __name__ == '__main__':
    if sys.argv[1] == 'pr':
        if sys.argv[2] == 'hist':
            precipitation_historical(variable = 'pr')
        if sys.argv[2] == 'proj':
            precipitation_projected(variable = 'pr')
    
    if sys.argv[1] == 'sfcwind':
        if sys.argv[2] == 'hist':
            if sys.argv[3] == '0':
                windspeed_historical(variable = 'sfcwind', hour = 0)
            if sys.argv[3] == '3':
                windspeed_historical(variable = 'sfcwind', hour = 3)
            if sys.argv[3] == '6':
                windspeed_historical(variable = 'sfcwind', hour = 6)
            if sys.argv[3] == '9':
                windspeed_historical(variable = 'sfcwind', hour = 9)
            if sys.argv[3] == '12':
                windspeed_historical(variable = 'sfcwind', hour = 12)
            if sys.argv[3] == '15':
                windspeed_historical(variable = 'sfcwind', hour = 15)
            if sys.argv[3] == '18':
                windspeed_historical(variable = 'sfcwind', hour = 18)
            if sys.argv[3] == '21':
                windspeed_historical(variable = 'sfcwind', hour = 21)
    
    if sys.argv[1] == 'sfcwind':
        if sys.argv[2] == 'proj':
            if sys.argv[3] == '0':
                windspeed_projected(variable = 'sfcwind', hour = 0)
            if sys.argv[3] == '3':
                windspeed_projected(variable = 'sfcwind', hour = 3)
            if sys.argv[3] == '6':
                windspeed_projected(variable = 'sfcwind', hour = 6)
            if sys.argv[3] == '9':
                windspeed_projected(variable = 'sfcwind', hour = 9)
            if sys.argv[3] == '12':
                windspeed_projected(variable = 'sfcwind', hour = 12)
            if sys.argv[3] == '15':
                windspeed_projected(variable = 'sfcwind', hour = 15)
            if sys.argv[3] == '18':
                windspeed_projected(variable = 'sfcwind', hour = 18)
            if sys.argv[3] == '21':
                windspeed_projected(variable = 'sfcwind', hour = 21)
    
    if sys.argv[1] == 'snc':
        if sys.argv[2] == 'hist':
            snow_cover_historical(variable = 'snc')
        if sys.argv[2] == 'proj':
            snow_cover_projected(variable = 'snc')
    
    if sys.argv[1] == 'gpp':
        if sys.argv[2] == 'hist':
            gpp_historical(variable = 'gpp')
        if sys.argv[2] == 'proj':
            gpp_projected(variable = 'gpp')
    
    if sys.argv[1] == 'tas':
        if sys.argv[2] == 'hist':
            if sys.argv[3] == '0':
                templike_historical(variable = 'tas', hour = 0)
            if sys.argv[3] == '3':
                templike_historical(variable = 'tas', hour = 3)
            if sys.argv[3] == '6':
                templike_historical(variable = 'tas', hour = 6)
            if sys.argv[3] == '9':
                templike_historical(variable = 'tas', hour = 9)
            if sys.argv[3] == '12':
                templike_historical(variable = 'tas', hour = 12)
            if sys.argv[3] == '15':
                templike_historical(variable = 'tas', hour = 15)
            if sys.argv[3] == '18':
                templike_historical(variable = 'tas', hour = 18)
            if sys.argv[3] == '21':
                templike_historical(variable = 'tas', hour = 21)
    
    if sys.argv[1] == 'tas':
        if sys.argv[2] == 'proj':
            if sys.argv[3] == '0':
                templike_projected(variable = 'tas', hour = 0)
            if sys.argv[3] == '3':
                templike_projected(variable = 'tas', hour = 3)
            if sys.argv[3] == '6':
                templike_projected(variable = 'tas', hour = 6)
            if sys.argv[3] == '9':
                templike_projected(variable = 'tas', hour = 9)
            if sys.argv[3] == '12':
                templike_projected(variable = 'tas', hour = 12)
            if sys.argv[3] == '15':
                templike_projected(variable = 'tas', hour = 15)
            if sys.argv[3] == '18':
                templike_projected(variable = 'tas', hour = 18)
            if sys.argv[3] == '21':
                templike_projected(variable = 'tas', hour = 21)
    
    if sys.argv[1] == 'das':
        if sys.argv[2] == 'hist':
            if sys.argv[3] == '0':
                templike_historical(variable = 'das', hour = 0)
            if sys.argv[3] == '3':
                templike_historical(variable = 'das', hour = 3)
            if sys.argv[3] == '6':
                templike_historical(variable = 'das', hour = 6)
            if sys.argv[3] == '9':
                templike_historical(variable = 'das', hour = 9)
            if sys.argv[3] == '12':
                templike_historical(variable = 'das', hour = 12)
            if sys.argv[3] == '15':
                templike_historical(variable = 'das', hour = 15)
            if sys.argv[3] == '18':
                templike_historical(variable = 'das', hour = 18)
            if sys.argv[3] == '21':
                templike_historical(variable = 'das', hour = 21)
    
    if sys.argv[1] == 'das':
        if sys.argv[2] == 'proj':
            if sys.argv[3] == '0':
                templike_projected(variable = 'das', hour = 0)
            if sys.argv[3] == '3':
                templike_projected(variable = 'das', hour = 3)
            if sys.argv[3] == '6':
                templike_projected(variable = 'das', hour = 6)
            if sys.argv[3] == '9':
                templike_projected(variable = 'das', hour = 9)
            if sys.argv[3] == '12':
                templike_projected(variable = 'das', hour = 12)
            if sys.argv[3] == '15':
                templike_projected(variable = 'das', hour = 15)
            if sys.argv[3] == '18':
                templike_projected(variable = 'das', hour = 18)
            if sys.argv[3] == '21':
                templike_projected(variable = 'das', hour = 21)
                
    if sys.argv[1] == 'final':
        if sys.argv[2] == 'hist':
            if sys.argv[3] == 'dry':
                step = int(sys.argv[4])*10
                for i in range(step, step+10):
                    dryness_runner(period = 'historical', mem_ind = i)
            if sys.argv[3] == 'wind':
                step = int(sys.argv[4])*10
                for i in range(step, step+10):
                    wind_runner(period = 'historical', mem_ind = i)
        if sys.argv[2] == 'proj':
            if sys.argv[3] == 'dry':
                step = int(sys.argv[4])*10
                for i in range(step, step+10):
                    dryness_runner(period = 'projected', mem_ind = i)
            if sys.argv[3] == 'wind':
                step = int(sys.argv[4])*10
                for i in range(step, step+10):
                    wind_runner(period = 'projected', mem_ind = i)
                    
    if sys.argv[1] == 'new_sfcwind':
        rescale_sfcwind()
        