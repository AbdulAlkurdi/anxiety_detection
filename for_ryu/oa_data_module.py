

import warnings
warnings.filterwarnings('ignore')

# ML libraries 
from sklearn.cluster import KMeans
from sklearn.neighbors.kde import KernelDensity


# Computation & Signal Processing
from scipy import signal
import numpy as np
import pandas as pd
import pylab as pl
import pickle

#biosppy package for ecg signal analysis
from biosppy import storage
from biosppy.signals import ecg

from utils import * #import data import, clean up and sampling functions. 
# import time
import time
import os
import glob
import time

from ECG_feature_extractor_1000 import *
from utils import * #import data import, clean up and sampling functions. 

emg_ecg_columns = ['X[s]', 'R RECTUS FEMORIS: EMG 1', 'L RECTUS FEMORIS: EMG 2', 'Trigno sensor 3: EMG 3', 
                 'L BICEPS FEMORIS: EMG 4', 'Trigno Mini sensor 5: EMG 5', 'R BICEPS FEMORIS: EMG 6',
                 'R TIBIALIS ANTERIOR: EMG 7','L TIBIALIS ANTERIOR: EMG 8', 'L GASTROCNEMIUS LATERAL HEAD: EMG 11',
                 'L GASTROCNEMIUS MEDIAL HEAD: EMG 12', 'R GASTROCNEMIUS LATERAL HEAD: EMG 13',
                 'R GASTROCNEMIUS MEDIAL HEAD: EMG 14', 'L PECTORALIS MAJOR: EKG 16']
acc_columns = ['X[s].1', 'R RECTUS FEMORIS: Acc 1.X', 'R RECTUS FEMORIS: Acc 1.Y',
               'R RECTUS FEMORIS: Acc 1.Z', 'L RECTUS FEMORIS: Acc 2.X',
               'L RECTUS FEMORIS: Acc 2.Y', 'L RECTUS FEMORIS: Acc 2.Z',
               'Trigno sensor 3: Acc 3.X', 'Trigno sensor 3: Acc 3.Y',
               'Trigno sensor 3: Acc 3.Z', 'L BICEPS FEMORIS: Acc 4.X', 
               'L BICEPS FEMORIS: Acc 4.Y', 'L BICEPS FEMORIS: Acc 4.Z',
               'Trigno Mini sensor 5: Acc 5.X', 'Trigno Mini sensor 5: Acc 5.Y',
               'Trigno Mini sensor 5: Acc 5.Z', 'R BICEPS FEMORIS: Acc 6.X', 
               'R BICEPS FEMORIS: Acc 6.Y', 'R BICEPS FEMORIS: Acc 6.Z',
               'R TIBIALIS ANTERIOR: Acc 7.X', 'R TIBIALIS ANTERIOR: Acc 7.Y',
               'R TIBIALIS ANTERIOR: Acc 7.Z', 'L TIBIALIS ANTERIOR: Acc 8.X',
               'L TIBIALIS ANTERIOR: Acc 8.Y', 'L TIBIALIS ANTERIOR: Acc 8.Z',
               'R TENSOR FASCIAE LATAE: Acc 9.X', 'R TENSOR FASCIAE LATAE: Acc 9.Y',
               'R TENSOR FASCIAE LATAE: Acc 9.Z', 'R VASTUS LATERALIS: Acc 10.X',
               'R VASTUS LATERALIS: Acc 10.Y', 'R VASTUS LATERALIS: Acc 10.Z',
               'L GASTROCNEMIUS LATERAL HEAD: Acc 11.X',
          'L GASTROCNEMIUS LATERAL HEAD: Acc 11.Y', 'L GASTROCNEMIUS LATERAL HEAD: Acc 11.Z', 
          'L GASTROCNEMIUS MEDIAL HEAD: Acc 12.X', 'L GASTROCNEMIUS MEDIAL HEAD: Acc 12.Y', 
          'L GASTROCNEMIUS MEDIAL HEAD: Acc 12.Z', 'R GASTROCNEMIUS LATERAL HEAD: Acc 13.X', 
          'R GASTROCNEMIUS LATERAL HEAD: Acc 13.Y', 'R GASTROCNEMIUS LATERAL HEAD: Acc 13.Z',
          'R GASTROCNEMIUS MEDIAL HEAD: Acc 14.X', 'R GASTROCNEMIUS MEDIAL HEAD: Acc 14.Y', 
          'R GASTROCNEMIUS MEDIAL HEAD: Acc 14.Z', 'Trigno Trigger sensor 15: Trig 15',
          'L PECTORALIS MAJOR: Acc 16.X', 'L PECTORALIS MAJOR: Acc 16.Y', 'L PECTORALIS MAJOR: Acc 16.Z']

class OA_data:
    def __init__(self, number):
        self.data_array = []
        self.number = number
        
        
    def fit_emg_acc(self, data_dir = 'csv/'):
        trial_name_1 = self.number + "_NW1"
        trial_name_2 = self.number + "_NW2"
        trial_name_3 = self.number + "_P1"
        trial_name_4 = self.number + "_P2"
        
        trial_data = glob.glob(data_dir + trial_name_1 + "*")
        try:
            #read raw file data
            trial_name = trial_data[-1]
            df_emg_acc1 = load_data(trial_name, data_dir = '')
            #extract emg and ekg and do cleanup
            self.df_emg1 = df_emg_acc1[emg_ecg_columns]
            self.df_emg1 = delsys_cleanup(self.df_emg1, column='all')
            self.df_emg1 = self.df_emg1.rename(columns={'X[s]': 'time'})
            #extract acc and do cleanup
            self.df_acc1 = df_emg_acc1[acc_columns]
            self.df_acc1 = delsys_cleanup(self.df_acc1, column='all')
            self.df_acc1 = self.df_acc1.rename(columns={'X[s].1': 'time'})
            
        except:
            print("EMG and ACC file for " + trial_name_1 + " does not exist")
            
        trial_data = glob.glob(data_dir + trial_name_2 + "*")
        try:
            #read raw file data
            trial_name = trial_data[-1]
            df_emg_acc2 = load_data(trial_name, data_dir = '')
            #extract emg and ekg and do cleanup
            self.df_emg2 = df_emg_acc2[emg_ecg_columns]
            self.df_emg2 = delsys_cleanup(self.df_emg2, column='all')
            self.df_emg2 = self.df_emg2.rename(columns={'X[s]': 'time'})
            #extract acc and do cleanup
            self.df_acc2 = df_emg_acc2[acc_columns]
            self.df_acc2 = delsys_cleanup(self.df_acc2, column='all')
            self.df_acc2 = self.df_acc2.rename(columns={'X[s].1': 'time'})
            
        except:
            print("EMG file for " + trial_name_2 + " does not exist")
            
        trial_data = glob.glob(data_dir + trial_name_3 + "*")
        try:
            #read raw file data
            trial_name = trial_data[-1]
            df_emg_acc3 = load_data(trial_name, data_dir = '')
            #extract emg and ekg and do cleanup
            self.df_emg3 = df_emg_acc3[emg_ecg_columns]
            self.df_emg3 = delsys_cleanup(self.df_emg3, column='all')
            self.df_emg3 = self.df_emg3.rename(columns={'X[s]': 'time'})
            #extract acc and do cleanup
            self.df_acc3 = df_emg_acc3[acc_columns]
            self.df_acc3 = delsys_cleanup(self.df_acc3, column='all')
            self.df_acc3 = self.df_acc3.rename(columns={'X[s].1': 'time'})
        except:
            print("EMG file for " + trial_name_3 + " does not exist")
            
        trial_data = glob.glob(data_dir + trial_name_4 + "*")
        try:
            #read raw file data
            trial_name = trial_data[-1]
            df_emg_acc4 = load_data(trial_name, data_dir = '')
            #extract emg and ekg and do cleanup
            self.df_emg4 = df_emg_acc4[emg_ecg_columns]
            self.df_emg4 = delsys_cleanup(self.df_emg4, column='all')
            self.df_emg4 = self.df_emg4.rename(columns={'X[s]': 'time'})
            #extract acc and do cleanup
            self.df_acc4 = df_emg_acc4[acc_columns]
            self.df_acc4 = delsys_cleanup(self.df_acc4, column='all')
            self.df_acc4 = self.df_acc4.rename(columns={'X[s].1': 'time'})
        except:
            print("EMG file for " + trial_name_4 + " does not exist")
        
    
    def fit_fNIRS(self, data_dir = 'fNIRS_Walking data'):
        file_name = '/OA_FNIRS_2019_WALK_' + self.number + '_oxydata.txt'
        df_fNIRS = pd.read_csv(data_dir+file_name,sep='\t')
        df_fNIRS = df_fNIRS.rename(columns={'1':'subject_ID', '2':'cohort', '3':'block','4':'trial_i','5':'HbO1','6':'HbR1','7':'HbO2','7':'HbR2','8':'HbO3','9':'HbR3','10':'HbO4','11':'HbR4','12':'HbO5','13':'HbR5','14':'HbO6','15':'HbR6','16':'HbO7','17':'HbR7','18':'HbO8','19':'HbR8','20':'HbO9','21':'HbR9','23':'HbO10','24':'HbR10','25':'HbO11','26':'HbR11','27':'HbO12','28':'HbR12','29':'HbO13','30':'HbR13','31':'HbO14','32':'HbR14','33':'HbO15','34':'HbR15','35':'HbO16','36':'HbR16'})
        df_fNIRS['time'] = [i*0.5 for i in range(len(df_fNIRS))]
        df_fNIRS = df_fNIRS[['time','subject_ID', 'cohort', 'block', 'trial_i', 'HbO1', 'HbR1', 'HbR2',
                            'HbO3', 'HbR3', 'HbO4', 'HbR4', 'HbO5', 'HbR5', 'HbO6', 'HbR6', 'HbO7',
                            'HbR7', 'HbO8', 'HbR8', 'HbO9', 'HbR9', '22', 'HbO10', 'HbR10', 'HbO11',
                            'HbR11', 'HbO12', 'HbR12', 'HbO13', 'HbR13', 'HbO14', 'HbR14', 'HbO15',
                            'HbR15', 'HbO16', 'HbR16']]
        self.df_fNIRS_N1 = df_fNIRS[df_fNIRS['trial_i']==1]
        self.df_fNIRS_N2 = df_fNIRS[df_fNIRS['trial_i']==2]
        self.df_fNIRS_P1 = df_fNIRS[df_fNIRS['trial_i']==3]
        self.df_fNIRS_P2 = df_fNIRS[df_fNIRS['trial_i']==4]
        
    
    def fit_TMdata(self, data_dir = 'Treadmill_data'):
        trial_name_1 = self.number + "_N1"
        trial_name_2 = self.number + "_N2"
        trial_name_3 = self.number + "_P1"
        trial_name_4 = self.number + "_P2"
        
        file_name = '/OA_' + trial_name_1 + '_RAWDATA.csv'
        self.df_tm1 = load_data(file_name, data_dir, header=1)
        self.df_tm1 = self.df_tm1.rename(columns={'Time': 'time'})
        
        file_name = '/OA_' + trial_name_2 + '_RAWDATA.csv'
        self.df_tm2 = load_data(file_name, data_dir, header=1)
        self.df_tm2 = self.df_tm2.rename(columns={'Time': 'time'})
        
        file_name = '/OA_' + trial_name_3 + '_RAWDATA.csv'
        self.df_tm3 = load_data(file_name, data_dir, header=1)
        self.df_tm3 = self.df_tm3.rename(columns={'Time': 'time'})
        
        file_name = '/OA_' + trial_name_4 + '_RAWDATA.csv'
        self.df_tm4 = load_data(file_name, data_dir, header=1)
        self.df_tm4 = self.df_tm4.rename(columns={'Time': 'time'})
        
    def time_synch(self, data_dir_sync = 'Sync_txt_file/OA_302_B_WMP/'):
        trials = ['N1','N2','P1','P2']
        for i in trials:
            emg, acc, fNIRS, treadmill = self.cut_plz(i)
            if i == "N1":
                self.synched_emg1 = emg
                self.synched_acc1 = acc
                self.synched_fNIRS_N1 = fNIRS
                self.synched_tm1 = treadmill
            elif i == "N2":
                self.synched_emg2 = emg
                self.synched_acc2 = acc
                self.synched_fNIRS_N2 = fNIRS
                self.synched_tm2 = treadmill
            elif i == "P1":
                self.synched_emg3 = emg
                self.synched_acc3 = acc
                self.synched_fNIRS_P1 = fNIRS
                self.synched_tm3 = treadmill
            elif i == "P2":
                self.synched_emg4 = emg
                self.synched_acc4 = acc
                self.synched_fNIRS_P2 = fNIRS
                self.synched_tm4 = treadmill
            
    def cut_plz(self, trial, data_dir_sync = 'Sync_txt_file/OA_302_B_WMP/'):
        data_dir_sync = data_dir_sync + trial
        sync_emg = pd.read_csv(data_dir_sync+'/EMGtime.txt', sep='\s+')
        sync_fnirs = pd.read_csv(data_dir_sync+'/fNIRstime.txt', sep='\s+')
        sync_tm = pd.read_csv(data_dir_sync+'/treadmill_v.txt', sep='\s+')
        
        res = next(x for x, val in enumerate(sync_tm['TreadmillV']) if val > 4.98) 
    
        t_tm = sync_tm['Time'][res]
        t_fnirs = sync_fnirs['RStime'][0]
        t_emg = sync_emg['begintime'][0]
            
        if trial == 'N1':
            emg = self.df_emg1
            acc = self.df_acc1
            fNIRS = self.df_fNIRS_N1
            treadmill = self.df_tm1
        elif trial == 'N2':
            emg = self.df_emg2
            acc = self.df_acc2
            fNIRS = self.df_fNIRS_N2
            treadmill = self.df_tm2
        elif trial == 'P1':
            emg = self.df_emg3
            acc = self.df_acc3
            fNIRS = self.df_fNIRS_P1
            treadmill = self.df_tm3
        elif trial == 'P2':
            emg = self.df_emg4
            acc = self.df_acc4
            fNIRS = self.df_fNIRS_P2
            treadmill = self.df_tm4
        
        if t_emg > t_tm and t_emg > t_fnirs: #, cut tm & fnirs

            #treadmill data
            dt = t_emg - t_tm 
            treadmill = cutting(treadmill, dt)

            #fNIRS
            dt = t_emg - t_fnirs
            fNIRS = cutting(fNIRS, dt)
            return emg, acc, treadmill, fNIRS
        
        elif t_fnirs > t_tm and t_fnirs > t_emg:

            #treadmill data
            dt = t_fnirs - t_tm 
            treadmill = cutting(treadmill, dt)

            #emg, acc & ecg  
            dt = t_fnirs - t_emg
            emg = cutting(emg, dt)
            acc = cutting(acc, dt)
            return emg, acc, treadmill, fNIRS
        
        elif t_tm > t_emg and t_tm > t_fnirs:
            #emg, acc & ecg  
            dt = t_tm - t_emg 
            emg = cutting(emg, dt)
            acc = cutting(acc, dt)

            #fNIRS
            dt = t_tm - t_fnirs
            fNIRS = cutting(fNIRS, dt)
            return emg, acc, treadmill, fNIRS
    
    def synched_to_csv(self, trial):
        if trial == 1:
            self.synched_emg1.to_csv("synched_" + str(self.number) + "_emg_N1.csv")
            self.synched_acc1.to_csv("synched_" + str(self.number) + "_acc_N1.csv")
            self.synched_fNIRS_N1.to_csv("synched" + str(self.number) + "_fNIRS_N1.csv")
            self.synched_tm1.to_csv("synched" + str(self.number) + "_treadmill_N1.csv")
        elif trial == 2:
            self.synched_emg2.to_csv("synched_" + str(self.number) + "_emg_N2.csv")
            self.synched_acc2.to_csv("synched_" + str(self.number) + "_acc_N2.csv")
            self.synched_fNIRS_N2.to_csv("synched" + str(self.number) + "_fNIRS_N2.csv")
            self.synched_tm2.to_csv("synched" + str(self.number) + "_treadmill_N2.csv")
        elif trial == 3:
            self.synched_emg3.to_csv("synched_" + str(self.number) + "_emg_P1.csv")
            self.synched_acc3.to_csv("synched_" + str(self.number) + "_acc_P1.csv")
            self.synched_fNIRS_P1.to_csv("synched" + str(self.number) + "_fNIRS_p1.csv")
            self.synched_tm3.to_csv("synched" + str(self.number) + "_treadmill_P1.csv")
        elif trial == 4:
            self.synched_emg4.to_csv("synched_" + str(self.number) + "_emg_P2.csv")
            self.synched_acc4.to_csv("synched_" + str(self.number) + "_acc_P2.csv")
            self.synched_fNIRS_P2.to_csv("synched" + str(self.number) + "_fNIRS_P2.csv")
            self.synched_tm4.to_csv("synched" + str(self.number) + "_treadmill_P2.csv")    
        
        
    def save(self):
        hdf = pd.HDFStore(self.number +'.h5')
        hdf.put('emg_N1', self.df_emg1, format='table', data_columns=True)
        hdf.put('emg_N2', self.df_emg2, format='table', data_columns=True)
        hdf.put('emg_P1', self.df_emg3, format='table', data_columns=True)
        hdf.put('emg_P2', self.df_emg4, format='table', data_columns=True)
        hdf.put('acc_N1', self.df_acc1, format='table', data_columns=True)
        hdf.put('acc_N2', self.df_acc2, format='table', data_columns=True)
        hdf.put('acc_P1', self.df_acc3, format='table', data_columns=True)
        hdf.put('acc_P2', self.df_acc4, format='table', data_columns=True)        
        hdf.put('fNIRS_N1', self.df_fNIRS_N1, format='table', data_columns=True)
        hdf.put('fNIRS_N2', self.df_fNIRS_N2, format='table', data_columns=True)
        hdf.put('fNIRS_P1', self.df_fNIRS_P1, format='table', data_columns=True)
        hdf.put('fNIRS_P2', self.df_fNIRS_P2, format='table', data_columns=True)
        hdf.put('TM_N1', self.df_tm1, format='table', data_columns=True)
        hdf.put('TM_N2', self.df_tm2, format='table', data_columns=True)
        hdf.put('TM_P1', self.df_tm3, format='table', data_columns=True)
        hdf.put('TM_P2', self.df_tm4, format='table', data_columns=True)
        hdf.close()