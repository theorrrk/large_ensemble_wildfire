from bias_correction_functions import *


def tas_runner(period = 'hist'):
    variable = 'tas'
    mode = 'add'
    
    # Run
    obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                'wildfires_theo_keeping/live/ensemble_data/'+
                'processing/tas_19900101_20191231_era5.nc')
    ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/tas_20000101_20091231_lentis.nc')
    ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/tas_20750101_20841231_lentis.nc')
    
    # 1: Load observation and ensemble data for only that time of day.
    x_oh_a = xr.open_dataset(obs_path).load()
    x_oh_a = x_oh_a.rename({list(x_oh_a.data_vars)[0]: variable})
    x_oh_a = x_oh_a[variable]
    x_mh_b = xr.open_dataset(ens_path_hist).load()
    x_mh_b = x_mh_b.rename({list(x_mh_b.data_vars)[0]: variable})
    x_mh_b = x_mh_b[variable]
    print('Pre Loaded Data.\n\n')
    sys.stdout.flush()
    x_oh_a = x_oh_a.resample(time = 'D').mean()
    x_oh_a['time'] = pd.date_range("1990-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 10957)
    x_mh_b_list = []
    for i in range(160):
        x_mh_b_list.append(x_mh_b[i,:,:,:].resample(time = 'D').mean())
    x_mh_b = xr.concat(x_mh_b_list, dim = 'member')
    del x_mh_b_list
    x_mh_b['time'] = pd.date_range("2000-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 3653)

    # 2: Run bias-correction:
    if period == 'hist':
        if mode == 'add':
            present_additive_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
        elif mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            present_multiplicative_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
    elif period == 'proj':
        x_mp_b = xr.open_dataset(ens_path_proj).load()
        x_mp_b = x_mp_b.rename({list(x_mp_b.data_vars)[0]: variable})
        x_mp_b = x_mp_b[variable]
        x_mp_b_list = []
        for i in range(160):
            x_mp_b_list.append(x_mp_b[i,:,:,:].resample(time = 'D').mean())
        x_mp_b = xr.concat(x_mp_b_list, dim = 'member')
        del x_mp_b_list
        x_mp_b['time'] = pd.date_range("2075-01-01T12:00:00.000000000",
                                       freq = pd.DateOffset(days = 1),
                                       periods = 3653)
        if mode == 'add':
            future_additive_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
        if mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            x_mp_b = x_mp_b.clip(min = 0)
            future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
    return



def ps_runner(period = 'hist'):
    variable = 'ps'
    mode = 'add'
    
    # Run
    obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                'wildfires_theo_keeping/live/ensemble_data/'+
                'processing/ps_19900101_20191231_era5.nc')
    ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/ps_20000101_20091231_lentis.nc')
    ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/ps_20750101_20841231_lentis.nc')
    
    # 1: Load observation and ensemble data for only that time of day.
    x_oh_a = xr.open_dataset(obs_path).load()
    x_oh_a = x_oh_a.rename({list(x_oh_a.data_vars)[0]: variable})
    x_oh_a = x_oh_a[variable]
    x_mh_b = xr.open_dataset(ens_path_hist).load()
    x_mh_b = x_mh_b.rename({list(x_mh_b.data_vars)[0]: variable})
    x_mh_b = x_mh_b[variable]
    print('Pre Loaded Data.\n\n')
    sys.stdout.flush()
    #x_oh_a = x_oh_a.resample(time = 'D').mean() <- Already done
    x_oh_a['time'] = pd.date_range("1990-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 10957)
    x_mh_b_list = []
    for i in range(160):
        x_mh_b_list.append(x_mh_b[i,:,:,:].resample(time = 'D').mean())
    x_mh_b = xr.concat(x_mh_b_list, dim = 'member')
    del x_mh_b_list
    x_mh_b['time'] = pd.date_range("2000-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 3653)

    # 2: Run bias-correction:
    if period == 'hist':
        if mode == 'add':
            present_additive_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
        elif mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            present_multiplicative_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
    elif period == 'proj':
        x_mp_b = xr.open_dataset(ens_path_proj).load()
        x_mp_b = x_mp_b.rename({list(x_mp_b.data_vars)[0]: variable})
        x_mp_b = x_mp_b[variable]
        x_mp_b_list = []
        for i in range(160):
            x_mp_b_list.append(x_mp_b[i,:,:,:].resample(time = 'D').mean())
        x_mp_b = xr.concat(x_mp_b_list, dim = 'member')
        del x_mp_b_list
        x_mp_b['time'] = pd.date_range("2075-01-01T12:00:00.000000000",
                                       freq = pd.DateOffset(days = 1),
                                       periods = 3653)
        if mode == 'add':
            future_additive_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
        if mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            x_mp_b = x_mp_b.clip(min = 0)
            future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
    return
    
    
def pr_runner(period = 'hist'):
    variable = 'pr'
    mode = 'mult'
    # Run
    obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                'wildfires_theo_keeping/live/ensemble_data/'+
                'processing/pr_19900101_20191231_era5.nc')
    ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/pr_20000101_20091231_lentis.nc')
    ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/pr_20750101_20841231_lentis.nc')
    
    # 1: Load observation and ensemble data for only that time of day.
    x_oh_a = xr.open_dataset(obs_path).load()
    x_oh_a = x_oh_a.rename({list(x_oh_a.data_vars)[0]: variable})
    x_oh_a = x_oh_a[variable]
    x_mh_b = xr.open_dataset(ens_path_hist).load()
    x_mh_b = x_mh_b.rename({list(x_mh_b.data_vars)[0]: variable})
    x_mh_b = x_mh_b[variable]
    print('Pre Loaded Data.\n\n')
    sys.stdout.flush()
    #x_oh_a = x_oh_a.resample(time = 'D').mean() <- Already done
    x_oh_a['time'] = pd.date_range("1990-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 10957)
    x_mh_b['time'] = pd.date_range("2000-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 3653)


    # 2: Run bias-correction:
    if period == 'hist':
        if mode == 'add':
            present_additive_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
        elif mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            present_multiplicative_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
    elif period == 'proj':
        x_mp_b = xr.open_dataset(ens_path_proj).load()
        x_mp_b = x_mp_b.rename({list(x_mp_b.data_vars)[0]: variable})
        x_mp_b = x_mp_b[variable]
        x_mp_b_list = []
        for i in range(160):
            x_mp_b_list.append(x_mp_b[i,:,:,:].resample(time = 'D').mean())
        x_mp_b = xr.concat(x_mp_b_list, dim = 'member')
        del x_mp_b_list
        x_mp_b['time'] = pd.date_range("2075-01-01T12:00:00.000000000",
                                       freq = pd.DateOffset(days = 1),
                                       periods = 3653)
        if mode == 'add':
            future_additive_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
        if mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            x_mp_b = x_mp_b.clip(min = 0)
            future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
    return
    
    
def vpd_runner(period = 'hist'):
    variable = 'vpd'
    mode = 'mult'
    # Run
    obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                'wildfires_theo_keeping/live/ensemble_data/'+
                'processing/vpd_19900101_20191231_era5.nc')
    ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/vpd_20000101_20091231_lentis.nc')
    ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/vpd_20750101_20841231_lentis.nc')
    
    # 1: Load observation and ensemble data for only that time of day.
    x_oh_a = xr.open_dataset(obs_path).load()
    x_oh_a = x_oh_a.rename({list(x_oh_a.data_vars)[0]: variable})
    x_oh_a = x_oh_a[variable]
    x_mh_b = xr.open_dataset(ens_path_hist).load()
    x_mh_b = x_mh_b.rename({list(x_mh_b.data_vars)[0]: variable})
    x_mh_b = x_mh_b[variable]
    print('Pre Loaded Data.\n\n')
    sys.stdout.flush()
    #x_oh_a = x_oh_a.resample(time = 'D').mean() <- Already done
    x_oh_a['time'] = pd.date_range("1990-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 10957)
    x_mh_b_list = []
    for i in range(160):
        x_mh_b_list.append(x_mh_b[i,:,:,:].resample(time = 'D').mean())
    x_mh_b = xr.concat(x_mh_b_list, dim = 'member')
    del x_mh_b_list
    x_mh_b['time'] = pd.date_range("2000-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 3653)


    # 2: Run bias-correction:
    if period == 'hist':
        if mode == 'add':
            present_additive_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
        elif mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            present_multiplicative_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
    elif period == 'proj':
        x_mp_b = xr.open_dataset(ens_path_proj).load()
        x_mp_b = x_mp_b.rename({list(x_mp_b.data_vars)[0]: variable})
        x_mp_b = x_mp_b[variable]
        x_mp_b_list = []
        for i in range(160):
            x_mp_b_list.append(x_mp_b[i,:,:,:].resample(time = 'D').mean())
        x_mp_b = xr.concat(x_mp_b_list, dim = 'member')
        del x_mp_b_list
        x_mp_b['time'] = pd.date_range("2075-01-01T12:00:00.000000000",
                                       freq = pd.DateOffset(days = 1),
                                       periods = 3653)
        if mode == 'add':
            future_additive_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
        if mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            x_mp_b = x_mp_b.clip(min = 0)
            future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
    return


def ppfd_runner(period = 'hist'):
    variable = 'ppfd'
    mode = 'mult'
    # Run
    obs_path = ('/rds/general/user/tk22/projects/leverhulme_'+
                'wildfires_theo_keeping/live/ensemble_data/'+
                'processing/par_19900101_20191231_era5.nc')
    ens_path_hist = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/rsds_20000101_20091231_lentis.nc')
    ens_path_proj = ('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/ensemble_data/'+
                     'processing/rsds_20750101_20841231_lentis.nc')
    
    # 1: Load observation and ensemble data for only that time of day.
    x_oh_a = xr.open_dataset(obs_path).load()
    x_oh_a = x_oh_a.rename({list(x_oh_a.data_vars)[0]: variable})
    x_mh_b = xr.open_dataset(ens_path_hist).load()
    x_mh_b = x_mh_b.rename({list(x_mh_b.data_vars)[0]: variable})
    x_oh_a = x_oh_a[variable]
    x_mh_b = x_mh_b[variable]
    print('Pre Loaded Data.\n\n')
    sys.stdout.flush()
    #x_oh_a = x_oh_a.resample(time = 'D').mean() <- Already done
    x_oh_a['time'] = pd.date_range("1990-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 10957)
    x_mh_b_list = []
    for i in range(160):
        x_mh_b_list.append(x_mh_b[i,:,:,:].resample(time = 'D').mean())
    x_mh_b = xr.concat(x_mh_b_list, dim = 'member')
    del x_mh_b_list
    x_mh_b['time'] = pd.date_range("2000-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 3653)

    # 2: Run bias-correction:
    if period == 'hist':
        if mode == 'add':
            present_additive_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
        elif mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            present_multiplicative_run(x_oh_a, x_mh_b, variable = variable, ttime = 12)
    elif period == 'proj':
        x_mp_b = xr.open_dataset(ens_path_proj).load()
        x_mp_b = x_mp_b.rename({list(x_mp_b.data_vars)[0]: variable})
        x_mp_b = x_mp_b[variable]
        x_mp_b_list = []
        for i in range(160):
            x_mp_b_list.append(x_mp_b[i,:,:,:].resample(time = 'D').mean())
        x_mp_b = xr.concat(x_mp_b_list, dim = 'member')
        del x_mp_b_list
        x_mp_b['time'] = pd.date_range("2075-01-01T12:00:00.000000000",
                                       freq = pd.DateOffset(days = 1),
                                       periods = 3653)
        if mode == 'add':
            future_additive_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
        if mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            x_mp_b = x_mp_b.clip(min = 0)
            future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b, variable = variable, ttime = 12)
    return


def smooth_x_oh_a(x_oh_a):
    x_oh_a_tilde = x_oh_a.copy(deep = True)
    month_average = x_oh_a_tilde.groupby('time.month').mean()
    for i in range(len(x_oh_a_tilde.time)):
        month = int(x_oh_a_tilde.time[i].dt.month)
        month_index = month - 1
        x_oh_a_tilde[i,:,:] = month_average[month_index,:,:]
    # Paranoid tidying:
    x_oh_a_tilde.load()
    month_average.close()
    del month_average
    x_oh_a_tilde = x_oh_a_tilde.sel(time = x_oh_a_tilde.time.dt.year >= 2000)
    x_oh_a_tilde = x_oh_a_tilde.sel(time = x_oh_a_tilde.time.dt.year <= 2009)
    return x_oh_a_tilde



def similar_baseline_model(x_mh_b):
    x_mh_b.load()
    x_mh_b_base = x_mh_b.copy(deep = True)
    month_average = x_mh_b_base.groupby('time.month').mean()
    
    for i in range(len(x_mh_b_base.time)):
        month = int(x_mh_b_base.time[i].dt.month)
        month_index = month - 1
        x_mh_b_base[:,i,:,:] = month_average[:,month_index,:,:]
        
    month_average.close()
    del month_average
    
    member_average = x_mh_b_base.mean(dim = 'member')
    
    for i in range(len(x_mh_b_base.member)):
        x_mh_b_base[i,:,:,:] = member_average
        
    member_average.close()
    del member_average
        
    # Paranoid tidying:
    x_mh_b_base.load()

    return x_mh_b_base


def bias_correct_monthly_present(variable = 'tas', mode = 'additive'):
    # 1.0: Load/ Generate monthly data:
    print(f'{variable}: {mode}')
    sys.stdout.flush()
    x_mh_b = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                             'wildfires_theo_keeping/live/gpp_ensemble/'+
                             f'{variable}_200001_200912_LENTIS.nc')[variable]
    x_mh_b = x_mh_b.transpose('member','time','latitude','longitude')
    path = glob.glob('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/gpp_ensemble/'+
                     f'{variable}_199001_201912_*.nc')[0]
    x_oh_a = xr.open_dataset(path)[variable]
    if variable == 'mrso':
        x_oh_a = x_oh_a.fillna(1.)
    #x_oh_a = x_oh_a.resample(time = '1M').mean()
    #x_oh_a['time'] = pd.date_range("1990-01-15T00:00:00.000000000",
    #                               freq = pd.DateOffset(months = 1),
    #                               periods = 360)
    
    if mode == 'multiplicative':
        x_mh_b = x_mh_b.clip(min = 0)
        x_oh_a = x_oh_a.clip(min = 0)


    # 1.1: Temporarily smooth x_oh_a:
    print('Step 1.1')
    sys.stdout.flush()
    x_oh_a_tilde = smooth_x_oh_a(x_oh_a)
    x_oh_a.close()

    # 1.2: Find x_mh_b_base from x_mh_b:
    print('Step 1.2')
    sys.stdout.flush()
    x_mh_b_base = similar_baseline_model(x_mh_b)

    # 1.3: Find d_mh_b from x_mh_b and x_mh_b_base:
    print('Step 1.3')
    sys.stdout.flush()
    if mode == 'additive':
        d_mh_b = x_mh_b - x_mh_b_base
        d_mh_b.load()

    elif mode == 'multiplicative':
        q_mh_b = x_mh_b / x_mh_b_base
        q_mh_b.load()

    x_mh_b.close()
    x_mh_b_base.close()
    del x_mh_b_base

    # 1.4: Find d_mh_a from d_mh_b:
    print('Step 1.4')
    sys.stdout.flush()
    if mode == 'additive':
        d_mh_a = xr.Dataset(data_vars = {variable: (['member','time','latitude','longitude'],
                                                    np.zeros((160,120,251,581)))},
                            coords = {'member': x_mh_b.member.to_numpy(),
                                      'time': x_mh_b.time.to_numpy(),
                                      'latitude': x_oh_a.latitude.to_numpy(),
                                      'longitude': x_oh_a.longitude.to_numpy()})[variable]
        for i in range(len(d_mh_a.member)):
            d_mh_a[i,:,:,:] = d_mh_b[i,:,:,:].interp(latitude = x_oh_a.latitude.data,
                                                     longitude = x_oh_a.longitude.data)

    if mode == 'multiplicative':
        q_mh_a = xr.Dataset(data_vars = {variable: (['member','time','latitude','longitude'],
                                                    np.zeros((160,120,251,581)))},
                            coords = {'member': x_mh_b.member.to_numpy(),
                                      'time': x_mh_b.time.to_numpy(),
                                      'latitude': x_oh_a.latitude.to_numpy(),
                                      'longitude': x_oh_a.longitude.to_numpy()})[variable]
        for i in range(len(q_mh_a.member)):
            q_mh_a[i,:,:,:] = q_mh_b[i,:,:,:].interp(latitude = x_oh_a.latitude.data,
                                                     longitude = x_oh_a.longitude.data)

    # 1.5: Find x_mh_a_tilde:
    print('Step 1.5')
    sys.stdout.flush()
    if mode == 'additive':
        x_mh_a_hat = x_oh_a_tilde + d_mh_a
        d_mh_a.close()
        del d_mh_a
    if mode == 'multiplicative':
        x_mh_a_hat = x_oh_a_tilde * q_mh_a
        q_mh_a.close()
        del q_mh_a
        
    x_mh_a_hat.load()
    
    x_oh_a_tilde.close()
    x_oh_a.close()
    x_mh_b.close()
    del x_oh_a_tilde, x_oh_a, x_mh_b
    
    x_mh_a_hat = x_mh_a_hat.transpose('member','time','latitude','longitude')
        
    return x_mh_a_hat



def bias_correct_monthly_future(variable = 'tas', mode = 'additive'):
    
    # 1.0: Load/ Generate monthly data:
    print(f'{variable}: {mode}')
    sys.stdout.flush()
    x_mh_b = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                             'wildfires_theo_keeping/live/gpp_ensemble/'+
                             f'{variable}_200001_200912_LENTIS.nc')[variable]
    x_mh_b = x_mh_b.transpose('member','time','latitude','longitude')
    x_mp_b = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                             'wildfires_theo_keeping/live/gpp_ensemble/'+
                             f'{variable}_207501_208412_LENTIS.nc')[variable]
    x_mp_b = x_mp_b.transpose('member','time','latitude','longitude')
    path = glob.glob('/rds/general/user/tk22/projects/leverhulme_'+
                     'wildfires_theo_keeping/live/gpp_ensemble/'+
                     f'{variable}_199001_201912_*.nc')[0]
    x_oh_a = xr.open_dataset(path)[variable]
    if variable == 'mrso':
        x_oh_a = x_oh_a.fillna(1.)
    #x_oh_a = x_oh_a.resample(time = '1M').mean()
    #x_oh_a['time'] = pd.date_range("1990-01-15T00:00:00.000000000",
    #                               freq = pd.DateOffset(months = 1),
    #                               periods = 360)
    
    if mode == 'multiplicative':
        x_mh_b = x_mh_b.clip(min = 0)
        x_mp_b = x_mp_b.clip(min = 0)
        x_oh_a = x_oh_a.clip(min = 0)

    # 1.1: Temporarily smooth x_oh_a:
    print('Step 1.1')
    sys.stdout.flush()
    x_oh_a_tilde = smooth_x_oh_a(x_oh_a)
    x_oh_a_tilde['time'] = x_mp_b['time']
    x_oh_a.close()

    # 1.2: Find x_mh_b_base from x_mh_b:
    print('Step 1.2')
    sys.stdout.flush()
    x_mh_b['time'] = x_mp_b['time']
    x_mh_b['member'] = x_mp_b['member']
    x_mh_b_base = similar_baseline_model(x_mh_b)

    x_mh_b.close()

    # 1.3: Find d_mh_b from x_mh_b and x_mh_b_base:
    print('Step 1.3')
    sys.stdout.flush()
    if mode == 'additive':
        d_mp_b = x_mp_b - x_mh_b_base
        d_mp_b.load()

    elif mode == 'multiplicative':
        q_mp_b = x_mp_b / x_mh_b_base
        q_mp_b.load()

    x_mp_b.close()
    x_mh_b_base.close()
    del x_mh_b_base

    # 1.4: Find d_mh_a from d_mh_b:
    print('Step 1.4')
    sys.stdout.flush()
    if mode == 'additive':
        d_mp_a = xr.Dataset(data_vars = {variable: (['member','time','latitude','longitude'],
                                                    np.zeros((160,120,251,581)))},
                            coords = {'member': x_mp_b.member.to_numpy(),
                                      'time': x_mp_b.time.to_numpy(),
                                      'latitude': x_oh_a.latitude.to_numpy(),
                                      'longitude': x_oh_a.longitude.to_numpy()})[variable]
        for i in range(len(d_mp_a.member)):
            d_mp_a[i,:,:,:] = d_mp_b[i,:,:,:].interp(latitude = x_oh_a.latitude.data,
                                                     longitude = x_oh_a.longitude.data)

    if mode == 'multiplicative':
        q_mp_a = xr.Dataset(data_vars = {variable: (['member','time','latitude','longitude'],
                                                    np.zeros((160,120,251,581)))},
                            coords = {'member': x_mp_b.member.to_numpy(),
                                      'time': x_mp_b.time.to_numpy(),
                                      'latitude': x_oh_a.latitude.to_numpy(),
                                      'longitude': x_oh_a.longitude.to_numpy()})[variable]
        for i in range(len(q_mp_a.member)):
            q_mp_a[i,:,:,:] = q_mp_b[i,:,:,:].interp(latitude = x_oh_a.latitude.data,
                                                     longitude = x_oh_a.longitude.data)

    # 1.5: Find x_mh_a_tilde:
    print('Step 1.5')
    sys.stdout.flush()
    if mode == 'additive':
        x_mp_a_hat = x_oh_a_tilde + d_mp_a
        d_mp_a.close()
        del d_mp_a
    if mode == 'multiplicative':
        x_mp_a_hat = x_oh_a_tilde * q_mp_a
        q_mp_a.close()
        del q_mp_a
    
    x_mp_a_hat.load()
    
    x_oh_a_tilde.close()
    x_oh_a.close()
    x_mp_b.close()
    x_mh_b.close()
    del x_oh_a_tilde, x_oh_a, x_mp_b, x_mh_b
    
    x_mp_a_hat = x_mp_a_hat.transpose('member','time','latitude','longitude')
    
    return x_mp_a_hat



def rel_soilm_runner(period = 'hist'):
    if period == 'hist':
        da = bias_correct_monthly_present(variable = 'mrso', mode = 'multiplicative')
    if period == 'proj':
        da = bias_correct_monthly_future(variable = 'mrso', mode = 'multiplicative')
    da = da / da.max(dim = ['member','time'])
    da = da.to_dataset()
    da = da.rename({'mrso':'rel_soilm'})
    if period == 'hist':
        da.to_netcdf('/rds/general/user/tk22/ephemeral/'+
                     'bias_corrected/rel_soilm_historical_monthly.nc')
    if period == 'proj':
        da.to_netcdf('/rds/general/user/tk22/ephemeral/'+
                     'bias_corrected/rel_soilm_projected_monthly.nc')
    return


def alpha_runner(period = 'hist'):
    if period == 'hist':
        da1 = bias_correct_monthly_present(variable = 'et', mode = 'additive')
    if period == 'proj':
        da1 = bias_correct_monthly_future(variable = 'et', mode = 'additive')
    member = da1.member.to_numpy()
    time = da1.time.to_numpy()
    latitude = da1.latitude.to_numpy()
    longitude = da1.longitude.to_numpy()
    da1 = np.repeat(da1.resample(time = '1Y').mean().to_numpy(), 12, axis = 1)
    if period == 'hist':
        da2 = bias_correct_monthly_present(variable = 'pet', mode = 'additive')
    if period == 'proj':
        da2 = bias_correct_monthly_future(variable = 'pet', mode = 'additive')
    da2 = np.repeat(da2.resample(time = '1Y').mean().to_numpy(), 12, axis = 1)
    da = da1 / da2
    del da1, da2
    da = xr.Dataset(data_vars = {'alpha': (['member','time','latitude','longitude'], da)},
                    coords = {'member': member, 'time': time,
                              'latitude': latitude, 'longitude': longitude})
    if period == 'hist':
        da.to_netcdf('/rds/general/user/tk22/ephemeral/'+
                     'bias_corrected/alpha_historical_monthly.nc')
    if period == 'proj':
        da.to_netcdf('/rds/general/user/tk22/ephemeral/'+
                     'bias_corrected/alpha_projected_monthly.nc')
    return


    

if __name__ == '__main__':
    variable = str(sys.argv[1])
    
    if variable == 'tas':
        tas_runner(period = 'hist')
        tas_runner(period = 'proj')
        
    if variable == 'patm':
        ps_runner(period = 'hist')
        ps_runner(period = 'proj')
        
    if variable == 'vpd':
        vpd_runner(period = 'hist')
        vpd_runner(period = 'proj')
        
    if variable == 'ppfd':
        ppfd_runner(period = 'hist')
        ppfd_runner(period = 'proj')
        
    if variable == 'pr':
        pr_runner(period = 'hist')
        pr_runner(period = 'proj')
        
    if variable == 'rel_soilm':
        rel_soilm_runner(period = 'hist')
        rel_soilm_runner(period = 'proj')
        
    if variable == 'alpha':
        alpha_runner(period = 'hist')
        alpha_runner(period = 'proj')