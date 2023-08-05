import mne
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
#mne.viz.set_3d_backend("notebook")

debug_asd_file_path=sorted(glob('data/asd/*.set'))
debug_td_file_path=sorted(glob('data/td/*.set'))

def get_filtered_data(file_path, l_freq = 0.5, h_freq = 40):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw.plot(block=True)

get_filtered_data(debug_asd_file_path[0])
#%%

#%%
