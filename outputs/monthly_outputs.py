import tempfile
import xarray as xr
import sys


def monthly_output(variable = 'p', period = 'historical'):
    print(f'\n\n\n{period}:\t{variable}\n')
    sys.stdout.flush()
    tmp = tempfile.gettempdir()
    try:
        da = xr.open_mfdataset('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                               f'downscaled/{variable}_{period}_*.nc')[variable].astype('float32').load()
    except:
        da = xr.open_dataset('/rds/general/user/tk22/ephemeral/ensemble_predictors/'+
                             f'downscaled/{variable}_{period}.nc')[variable].astype('float32').load()
    print(f'Dataset loaded.')
    sys.stdout.flush()
    for i in range(160):
        print(f'\t{i}')
        sys.stdout.flush()
        temp = da[i,:,:,:].copy(deep = True).load().expand_dims(dim = 'member')
        temp_mean = temp.resample(time = 'M').mean()
        #temp_std  = temp.resample(time = 'M').std()
        temp_mean.to_netcdf(tmp + f'/mean_{variable}_monthly_{period}_{i}.nc')
        #temp_std.to_netcdf(tmp + f'/std_{variable}_monthly_{period}_{i}.nc')
        temp.close()
        da.close()
        temp_mean.close()
        #temp_std.close()
        del temp, temp_mean#, temp_std
    da.close()
    del da
    ds = xr.open_mfdataset(tmp + f'/mean_{variable}_monthly_{period}_*.nc').load()
    ds[variable] = ds[variable].astype('float32').load()
    ds.to_netcdf('/rds/general/user/tk22/projects/leverhulme_'+
                 'wildfires_theo_keeping/live/ensemble_summaries/monthly/'+
                 f'monthly_mean_{variable}_{period}.nc')
    ds.close()
    del ds
    #ds = xr.open_mfdataset(tmp + f'/std_{variable}_monthly_{period}_*.nc').load()
    #ds[variable] = ds[variable].astype('float32').load()
    #ds.to_netcdf('/rds/general/user/tk22/projects/leverhulme_'+
    #             'wildfires_theo_keeping/live/ensemble_summaries/monthly/'+
    #             f'monthly_std_{variable}_{period}.nc')
    #ds.close()
    #del ds 
    return

if __name__ == '__main__':
    monthly_output(variable = sys.argv[1], period = sys.argv[2])