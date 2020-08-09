import pandas as pd
import numpy as np

def load_data(file_name,data_dir='csv', gdrive=True):
    file = data_dir*gdrive +  file_name
    df = pd.read_csv(file)
    return df

def delsys_cleanup(df, column='L PECTORALIS MAJOR: EKG 16'):
    df = df.fillna(0.0)
    
    start = np.where(np.diff(df[column])!=0)[0][0]+1
    end = np.where(np.diff(df[column][-1:0:-1])!=0)[0][0]+1
    ecg = df[column][start:-end].reset_index(drop=True)
    ecg_t = df.iloc[:, df.columns.get_loc(column)-1][start:-end].reset_index(drop=True)
    out = pd.concat([ecg_t.round(5), ecg],axis=1)
    out.columns = ['time','ecg']
    #out['time'] = pd.to_datetime(out['time'], unit = 's')
    return out
    
def resample(all_data, oversampling_period, undersampling_period, time_col = 'time'):
    """
    all_data: df with trajectories
    oversampling_period : increase resolution in time by interpolating within higher res period
    undersampling_period : picking points from the oversampled tajectory 
    tim_col 
    taken from eyas alfaris, theaeros github
    """
    traj_df =  all_data.copy()
    traj_df.set_index('time',inplace =True)
    
    tmp = traj_df.resample(oversampling_period).mean().interpolate() # oversample
    resampled = tmp.resample(undersampling_period).ffill()           # then undersample
    resampled.reset_index(inplace =True)
    #resampled = resampled.drop(['time'],axis= 1)
      
    return resampled

def cutting(input,dt):
    pos = next(x for x, val in enumerate(input['time']) if val >= dt) 
    out = input[pos::]
    out.reset_index(inplace =True)
    out = out.drop(columns=['index'])
    out['time'] = out['time']-dt
    return out