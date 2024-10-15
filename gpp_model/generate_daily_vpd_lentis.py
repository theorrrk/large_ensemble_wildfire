import xarray as xr
import numpy as np
import sys

def arden_buck(T):
    P = np.zeros_like(T)
    P[T >= 0] = 611.21 * np.exp((18.678 - T[T >= 0]/234.5)*(T[T >= 0]/(257.14 + T[T >= 0])))
    P[T < 0]  = 611.15 * np.exp((23.036 - T[T < 0]/333.7)*(T[T < 0]/(279.82 + T[T < 0])))
    return P


def make_vpd(T, D):
    return arden_buck(T) - arden_buck(D)


def main_historical():
    T = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                        'wildfires_theo_keeping/live/ensemble_data/processing/'+
                        'tas_20000101_20091231_lentis.nc')['tas'].to_numpy() - 273.15
    print('T loaded')
    sys.stdout.flush()
    D = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                        'wildfires_theo_keeping/live/ensemble_data/processing/'+
                        'das_20000101_20091231_lentis.nc')['das'].to_numpy() - 273.15
    print('D loaded')
    sys.stdout.flush()
    V = np.zeros_like(T)
    print('V loaded')
    sys.stdout.flush()
    for i in range(T.shape[0]):
        print(i)
        sys.stdout.flush()
        for j in range(T.shape[1]):
            V[i,j,:,:] = make_vpd(T[i,j,:,:], D[i,j,:,:])
            
    del T, D
    
    ds = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/processing/'+
                         'das_20000101_20091231_lentis.nc')
    
    V = xr.Dataset(data_vars = {'vpd': (['member','time','latitude','longitude'], V)},
                   coords = {'member': ds.member.to_numpy(),
                             'time': ds.time.to_numpy(),
                             'latitude': ds.latitude.to_numpy(),
                             'longitude': ds.longitude.to_numpy()})
    V.to_netcdf('/rds/general/user/tk22/projects/leverhulme_'+
                'wildfires_theo_keeping/live/ensemble_data/processing/'+
                'vpd_20000101_20091231_lentis.nc')
    return


def main_future():
    T = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                        'wildfires_theo_keeping/live/ensemble_data/processing/'+
                        'tas_20750101_20841231_lentis.nc')['tas'].to_numpy() - 273.15
    D = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                        'wildfires_theo_keeping/live/ensemble_data/processing/'+
                        'das_20750101_20841231_lentis.nc')['das'].to_numpy() - 273.15
    V = np.zeros_like(T)
    for i in range(T.shape[0]):
        print(i)
        sys.stdout.flush()
        for j in range(T.shape[1]):
            V[i,j,:,:] = make_vpd(T[i,j,:,:], D[i,j,:,:])
            
    del T, D
    
    ds = xr.open_dataset('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/processing/'+
                         'das_20750101_20841231_lentis.nc')
    
    V = xr.Dataset(data_vars = {'vpd': (['member','time','latitude','longitude'], V)},
                   coords = {'member': ds.member.to_numpy(),
                             'time': ds.time.to_numpy(),
                             'latitude': ds.latitude.to_numpy(),
                             'longitude': ds.longitude.to_numpy()})
    V.to_netcdf('/rds/general/user/tk22/projects/leverhulme_'+
                'wildfires_theo_keeping/live/ensemble_data/processing/'+
                'vpd_20750101_20841231_lentis.nc')
    return

    
if __name__ == '__main__':
    main_historical()
    main_future()