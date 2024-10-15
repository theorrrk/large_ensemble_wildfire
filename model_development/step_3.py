from build_dataset import *
from stretch_dataset import *
from dataset_benchmarks import *


def main(index, fire_class = 'B'):
    # 0: Get variables and thresholds:
    directory = '/rds/general/user/tk22/home/paper_2/model/step_3/final_thresholds/'
    path = directory + f'threshold_summary_{index}.csv'
    thresh_row = pd.read_csv(path, index_col = 0)
    #variables = [x[3:] for x in list(thresh_row.index) if x[:3] == 'lo_']
    #print('\nStep 0 Complete.\n')
    #sys.stdout.flush()
    ## 1: Build dataset based on step 2:
    #ds, aic, auc, max_vif = build_dataset(variables, thresh_row,
    #                                      fire_class = fire_class,
    #                                      index = index)
    #ds = ds.to_dataset()
    #print('\nStep 1 Complete.\n')
    #sys.stdout.flush()
    ## 2: Find best scaling of dataset via RSS: (~8 hours)
    #path = ('/rds/general/user/tk22/projects/leverhulme_wildfires_theo_keeping/'+
    #        f'live/ensemble_data/counts_{fire_class}_FPAFOD_19920101_20201231.nc')
    #ds['counts'] = (['time','latitude','longitude'],
    #                xr.open_dataset(path).counts.to_numpy())
    ds = xr.open_dataset(f'/rds/general/user/tk22/ephemeral/step_3/tmp_{index}.nc').load()
    mask = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                           'wildfires_theo_keeping/live/ensemble_data/'+
                           'mask_conus.nc').mask.to_numpy()
    print('\nStep 1 Complete.\n')
    sys.stdout.flush()
    df = stretch_dataset(ds, mask, decimals = 5)
    a = float(df[df.rss == df.rss.min()]['a'].iloc[0])
    b = float(df[df.rss == df.rss.min()]['b'].iloc[0])
    print('\nStep 2 Complete.\n')
    sys.stdout.flush()
    # 3: Build stretched dataset:
    ds['p'] = (a * ds.p ** b) * mask
    print(ds)
    print('\nStep 3 Complete.\n')
    sys.stdout.flush()
    # 4: Benchmarks of stretched dataset:
    #path = ('/rds/general/user/tk22/projects/leverhulme_wildfires_theo_keeping/'+
    #        f'live/ensemble_data/cell_area_6min_19920101_20201231.nc')
    #time.sleep(60)
    #ds['cell_area'] = xr.open_dataset(path).cell_area.load()
    # Building data (monthly, 0.5-degree interpolation to match FireMIP):
    ds1 = ds.copy(deep = True)
    ds1 = ds1.coarsen({'latitude':5,'longitude':5}, boundary = 'trim').mean()
    ds1 = ds1.resample(time = '1M').mean()
    # Annual data:
    ds2 = ds.copy(deep = True).resample(time = '1Y').sum()
    del ds
    print('\nStep 4 input datasets built.\n')
    sys.stdout.flush()
    # Getting stats:
    # Trimming as less than 2019 due to NaNs:
    ds1 = ds1.sel(time = ds1.time.dt.year <= 2019)
    ds2 = ds2.sel(time = ds2.time.dt.year <= 2019)
    
    nme_geospatial, mpd_seasonal, nme_seasonal, nme_interannual = benchmark_stats(ds1, ds2)
    print('\nStep 4 Complete.\n')
    sys.stdout.flush()
    # 5: Save key variables (thresholds, stretch coef, benchmarks) to a dataframe:
    output = thresh_row
    output = output.rename(columns = {'threshold': 'value'})
    stats = pd.DataFrame({'index': ['a', 'b', 'NME_geospatial',
                                    'NME_interannual', 'NME_seasonal', 'MPD_seasonal'],
                          'value': [a, b, nme_geospatial,
                                    nme_interannual, nme_seasonal, mpd_seasonal]})
    stats = stats.set_index('index')
    output = pd.concat([stats, output])
    directory = '/rds/general/user/tk22/home/paper_2/model/step_3/output/'
    path = directory + f'output_summary_{index}.csv'
    output.to_csv(path)
    print('\nStep 5 Complete.\n')
    sys.stdout.flush()
    try:
        os.remove(f'/rds/general/user/tk22/ephemeral/step_3/tmp_{index}.nc')
    except:
        pass
    print('\nStep 6 Complete.\n')
    sys.stdout.flush()
    return


if __name__ == '__main__':
    try:
        member = int(os.getenv('PBS_ARRAY_INDEX'))
        time.sleep(member*3*60)
        
        indices = list(np.arange(1,1001))
        random.shuffle(indices)
        for index in indices:
            print(f'Index: {index}')
            sys.stdout.flush()
            array_built = os.path.exists('/rds/general/user/tk22/ephemeral/'+
                                         f'step_3/tmp_{index}.nc')
            array_used = os.path.exists('/rds/general/user/tk22/home/'+
                                        'paper_2/model/step_3/output/'+
                                        f'output_summary_{index}.csv')
            if array_built:
                if not array_used:
                    main(index, fire_class = 'B')
        
    except:
        for loop in range(100):
            sys.stdout.flush()
            indices = list(np.arange(1,1001))
            random.shuffle(indices)

            for index in indices:
                print(f'\tIndex: {index}')
                sys.stdout.flush()
                array_built = os.path.exists('/rds/general/user/tk22/ephemeral/'+
                                             f'step_3/tmp_{index}.nc')
                array_used = os.path.exists('/rds/general/user/tk22/home/'+
                                            'paper_2/model/step_3/output/'+
                                            f'output_summary_{index}.csv')
                if array_built:
                    if not array_used:
                        main(index, fire_class = 'B')
            time.sleep(60*60)
