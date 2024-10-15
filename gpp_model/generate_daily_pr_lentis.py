import xarray as xr
import numpy as np
import pandas as pd
import sys
import os


def main():
    ds = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/pr_20000101_20091231_lentis.nc')['pr'].load()

    ds_list = []
    for i in range(160):
        print(i)
        sys.stdout.flush()
        ds_list.append(ds[i,:,:,:].resample(time = 'D').mean())
    ds_list = xr.concat(ds_list, dim = 'member')
    del ds
    ds_list['time'] = pd.date_range("2000-01-01T12:00:00.000000000",
                                    freq = pd.DateOffset(days = 1),
                                    periods = 3653)
    ds_list.to_dataset()
    ds_list.load()
    os.remove('/rds/general/user/tk22/projects/leverhulme_'+
              'wildfires_theo_keeping/live/ensemble_data/'+
              'processing/pr_20000101_20091231_lentis.nc')
    ds_list.to_netcdf('/rds/general/user/tk22/projects/leverhulme_'+
                      'wildfires_theo_keeping/live/ensemble_data/'+
                      'processing/pr_20000101_20091231_lentis.nc')
    del ds_list
    
    
    ds = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         'processing/pr_20750101_20841231_lentis.nc')['pr'].load()

    ds_list = []
    for i in range(160):
        print(i)
        sys.stdout.flush()
        ds_list.append(ds[i,:,:,:].resample(time = 'D').mean())
    ds_list = xr.concat(ds_list, dim = 'member')
    del ds
    ds_list['time'] = pd.date_range("2075-01-01T12:00:00.000000000",
                                    freq = pd.DateOffset(days = 1),
                                    periods = 3653)
    ds_list.to_dataset()
    ds_list.load()
    os.remove('/rds/general/user/tk22/projects/leverhulme_'+
              'wildfires_theo_keeping/live/ensemble_data/'+
              'processing/pr_20750101_20841231_lentis.nc')
    ds_list.to_netcdf('/rds/general/user/tk22/projects/leverhulme_'+
                      'wildfires_theo_keeping/live/ensemble_data/'+
                      'processing/pr_20750101_20841231_lentis.nc')
    del ds_list
    return
    
if __name__ == '__main__':
    main()