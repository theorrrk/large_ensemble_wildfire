from build_dataset import *

def reorganise_inputs(start = 1, stop = 200):
    in_directory = '/rds/general/user/tk22/home/paper_2/model/step_2/output/class_B/'
    out_directory = '/rds/general/user/tk22/home/paper_2/model/step_3/final_thresholds/'
    
    for i in range(start,stop + 1):
        path = in_directory + f'threshold_summary_{i}.csv'
        path_exists = os.path.exists(path)
        
        run_paths = glob.glob('/rds/general/user/tk22/home/paper_2/model/step_2/'+
                              f'runners/step_2_???-???.o*.{i}')
        run_finished = (len(run_paths) == 1)
        if path_exists and run_finished:
            series = pd.read_csv(path, index_col = 0).iloc[-1]
            series.to_csv(out_directory + path.split('/output/class_B/')[1],
                          index = True, header = ['threshold'])
            print(path)
            sys.stdout.flush()
    return


def build_arrays(fire_class = 'B', from_ind = 1, to_ind = 1001):
    # Load inputs:
    variables = ['CROP','elevation','GPP_50d','GPP_100d','GPP_150d','GPP_1yr',
                 'GPP_2yr','GPP_5yr','GPP_10yr','HERB','NEEDLELEAF','PopDens',
                 'PopDens_rural','RoadDens','SHRUB','TREE','VRM','dtr','mrsos',
                 'pr_5d','pr','rlds','rsds','sfcwind','sfcwind_max','snc','tas',
                 'tas_max','vpd_10d','vpd','vpd_max','vpd_night']
    directory = ('/rds/general/user/tk22/projects/'+
                 'leverhulme_wildfires_theo_keeping/'+
                 'live/ensemble_data/')
    reanalysis_paths = glob.glob(directory + 'reanalysis_*19900101_20211231.nc')
    landscape_paths  = glob.glob(directory + 'landscape_*_19920101_20201231.nc')
    reanalysis_paths = [r_path for r_path in reanalysis_paths if True in 
                        [v in r_path for v in variables]]
    landscape_paths  = [l_path for l_path in landscape_paths if True in 
                        [v in l_path for v in variables]]
    fire_path = glob.glob(directory + f'counts_{fire_class}*_19920101_20201231.nc')
    size_path = glob.glob(directory + f'cell_area*_19920101_20201231.nc')
    ds_paths = reanalysis_paths + landscape_paths + fire_path + size_path

    ds_dict = {'None': None}
    print('Building full dataset:')
    sys.stdout.flush()
    for path in ds_paths:
        da = xr.open_dataset(path)
        da = da.sel(time = da.time.dt.year <= 2020)
        da = da.sel(time = da.time.dt.year >= 1992)
        var = [v for v in list(da.variables) 
               if v not in ['latitude', 'longitude', 'time']][0]
        print(f'\t{var} added')
        sys.stdout.flush()
        ds_dict[var] = da[var].astype('float32').load()
    
    print('Data concatenated.')
    sys.stdout.flush()
    
    runs = list(np.arange(from_ind, to_ind))
    random.shuffle(runs)
    for loop in range(100):
        print(f'\n\nLoop: {loop}.\n\n')
        sys.stdout.flush()
        
        for index in runs:
            array_built = os.path.exists('/rds/general/user/tk22/ephemeral/'+
                                         f'step_3/tmp_{index}.nc')
            array_used = os.path.exists('/rds/general/user/tk22/home/'+
                                        'paper_2/model/step_3/output/'+
                                        f'output_summary_{index}.csv')
            step_2_complete = os.path.exists('/rds/general/user/tk22/home/paper_2/'+
                                             f'model/step_3/final_thresholds/threshold_summary_{index}.csv')
            if array_built:
                print(f'\t{index} of {from_ind} - {to_ind}: Temporary array already built!')
                sys.stdout.flush()
            elif array_used:
                print(f'\t{index} of {from_ind} - {to_ind}: Temporary array already used!')
                sys.stdout.flush()
            elif not step_2_complete:
                print(f'\t{index} of {from_ind} - {to_ind}: Input data not ready!')
                sys.stdout.flush()
            else:
                print(f'\tBuilding array for {index}.')
                sys.stdout.flush()
                directory = '/rds/general/user/tk22/home/paper_2/model/step_3/final_thresholds/'
                path = directory + f'threshold_summary_{index}.csv'
                thresh_row = pd.read_csv(path, index_col = 0)
                variables = [x[3:] for x in list(thresh_row.index) if x[:3] == 'lo_']

                train = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                                     f'genesis_inputs_train_{index}.csv'))
                test = pd.read_csv(('/rds/general/user/tk22/ephemeral/model_inputs/'+
                                    f'genesis_inputs_test_{index}.csv'))
                #print(f'\tTrain dataframe:\n{train}\n\n')
                #print(f'\tTest dataframe:\n{test}\n\n')
                #sys.stdout.flush()

                # Applying thresholds:
                test, train = apply_thresh(test, train, thresh_row, variables)
                #print('Thresholds applied.')
                #sys.stdout.flush()

                formula = 'counts ~ ' + ' + '.join(variables)

                params, aic, auc, max_vif = model_fit(test, train, formula, summary = True)
                del test, train

                logit_p = 0 * ds_dict['cell_area'].to_numpy()

                #time = ds_dict['cell_area'].time.to_numpy()
                #latitude = ds_dict['cell_area'].latitude.to_numpy()
                #longitude = ds_dict['cell_area'].longitude.to_numpy()

                ### TRY REWRITING WITH A NUMPY ARRAY!
                for param in params.keys():
                    if param == 'Intercept':
                        logit_p = logit_p + params['Intercept']
                    else:
                        #print(f'\tAdding {param} to model.')
                        #sys.stdout.flush()
                        temp = np.clip(ds_dict[param].to_numpy(),
                                       float(thresh_row.loc['lo_' + param]),
                                       float(thresh_row.loc['hi_' + param]))
                        logit_p = logit_p + params[param] * temp
                        del temp

                logit_p = logit_p + np.log(ds_dict['cell_area'].to_numpy())
                p = np.exp(logit_p)/(1 + np.exp(logit_p))
                ds = ds_dict['counts'].to_dataset()
                ds['cell_area'] = ds_dict['cell_area']
                ds['p'] = (['time','latitude','longitude'], p)
                ds[['p','counts','cell_area']].to_netcdf('/rds/general/user/tk22/ephemeral/'+
                                                         f'step_3/tmp_{index}.nc')
                print('\t\tArray built and saved!')
                sys.stdout.flush()
        print(f'\n\nLoop finished.\n\n')
        time.sleep(60*60)
        sys.stdout.flush()
    return


if __name__ == '__main__':
    reorganise_inputs(start = int(sys.argv[1]), stop = int(sys.argv[2]))
    build_arrays(fire_class = 'B', from_ind = int(sys.argv[1]), to_ind = int(sys.argv[2])+1)
    
