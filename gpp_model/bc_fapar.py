from bias_correction_functions import *


def fapar_runner(period = 'hist'):
    variable = 'fAPAR'
    mode = 'mult'
    
    if period == 'hist':
        members = ['h'+str(x).zfill(3) for x in range(10,170)]
    if period == 'proj':
        members = ['s'+str(x).zfill(3) for x in range(10,170)]
    
    # 1: Load observation and ensemble data for only that time of day.
    x_oh_a = xr.open_dataset('/rds/general/user/tk22/ephemeral/gpp_inputs_daily/'+
                             'fAPAR_19820101_20211231.nc')[variable].astype('float32')
    x_oh_a = x_oh_a.sel(time = x_oh_a.time.dt.year >= 1990)
    x_oh_a = x_oh_a.sel(time = x_oh_a.time.dt.year <= 2019)
    x_oh_a['time'] = pd.date_range("1990-01-01T12:00:00.000000000",
                                   freq = pd.DateOffset(days = 1),
                                   periods = 10957)
    
    x_mh_b = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/gpp_outputs/'+
                               'fAPAR_historical_h???_daily.nc')[variable].astype('float32')
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
        x_mp_b = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/gpp_outputs/'+
                                   'fAPAR_projected_s???_daily.nc')[variable].astype('float32')
        x_mp_b['time'] = pd.date_range("2075-01-01T12:00:00.000000000",
                                       freq = pd.DateOffset(days = 1),
                                       periods = 3653)
        if mode == 'add':
            future_additive_run(x_oh_a, x_mh_b, x_mp_b,
                                variable = variable, ttime = 12)
        if mode == 'mult':
            x_oh_a = x_oh_a.clip(min = 0)
            x_mh_b = x_mh_b.clip(min = 0)
            x_mp_b = x_mp_b.clip(min = 0)
            future_multiplicative_run(x_oh_a, x_mh_b, x_mp_b,
                                      variable = variable, ttime = 12)
    return
    
    
if __name__ == '__main__':
    #os.environ["OMP_NUM_THREADS"] = "2"
    fapar_runner(period = sys.argv[1])