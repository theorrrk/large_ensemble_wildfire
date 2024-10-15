from bc_funcs import *


def generate_moments(var = 'vpd'):
    path = min(glob.glob('/rds/general/user/tk22/projects/leverhulme_'+
                         'wildfires_theo_keeping/live/ensemble_data/'+
                         f'*{var}*.nc'), key = len)
    da = xr.open_dataset(path).load()
    df = get_moments_obs(da)
    df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
               f'final_moments/{variable}.csv'),
              header = True, mode = 'w', index = False)
    da.close()
    del da
    
    da = xr.open_dataset('/rds/general/user/tk22/ephemeral/ensemble_'+
                         f'predictors/downscaled/{var}_historical.nc')[var]
    members = da.member.to_numpy()
    for i in range(160):
        print(f'Adding moments from {members[i]}')
        sys.stdout.flush()
        temp = da.copy(deep = True).load()
        df = get_moments_ens(temp, member = members[i])
        df.to_csv(('/rds/general/user/tk22/home/paper_2/final_bias_correction/'+
                   f'final_moments/{var}.csv'),
                  header = False, mode = 'a', index = False)
        temp.close()
        da.close()
    return


if __name__ == '__main__':
    generate_moments(var = sys.argv[1])