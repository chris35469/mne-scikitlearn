import mne
from mne.preprocessing import annotate_muscle_zscore
import math
import numpy as np
import EntropyHub as EH
import matplotlib.pyplot as plt

mne.set_log_level("critical")

class Subject():
    def __init__(self, file_path, bad_channels=[], epoch_duration=1, epoch_overlap=0, l_freq=0.5, h_freq=40, muscle_thresh=5, plot_bads=False, drop_log=False, interpolate={}, fmin=0, fmax=40):
        # Load raw data fronm file
        self.raw = mne.io.read_raw_eeglab(file_path, preload=True)
        self.file_name = file_path
        self.ch_names = self.raw.ch_names
        self.raw.set_eeg_reference()
        self.raw_shape = self.raw.get_data().shape
        self.file_length_seconds = math.floor(int(self.raw_shape[1]) / int(self.raw.info['sfreq']))
        self.sfreq =  int(self.raw.info['sfreq'])
        self.fmin = fmin
        self.fmax = fmax       

        # Mark bad data
        self.markMuscleArtifacts(muscle_thresh)

        self.bad_channels = bad_channels
        self.post_interpolate_check = None

        # Handle Interpolation
        fileID = self.getFileID(file_path)

        if(fileID in interpolate):
            self.post_interpolate_check = self.fix_bad_channels(interpolate[fileID], plot_bads=plot_bads)
        else:
            self.post_interpolate_check = self.raw

        # Apply filter
        self.filtered = self.post_interpolate_check.copy().filter(l_freq=l_freq, h_freq=h_freq)

        # Create even length epochs
        self.epochs = mne.make_fixed_length_epochs(self.filtered, duration=epoch_duration, overlap=epoch_overlap, preload=True)

        print("========")
        print("Epochs (Before Drops): ", self.epochs)

        # Drop Bad Epochs
        self.dropBadEpochs(plotLog=drop_log)

        self.epochs.info['bads'] = [] # remove bads after handling interpolation and epoch drops

        self.psd = self.getPSD()

        print("File: ", self.file_name)
        print("Raw Freq: ",  self.sfreq )
        print("Raw Shape: ", self.raw_shape)
        print("Raw Seconds: ", self.file_length_seconds)
        print("Epochs (After Drops): ", self.epochs)

        #self.main()

    def main(self):
        self.getRegionFeatures(self.psd)
        self.getRegionFeatures(self.epochs)
        #self.getRegionRaw()

    def setEpochs(self, epochs):
        self.epochs = epochs

    def getEpochs(self):
        return self.epochs

    def setFilteredData(self, data):
        self.filtered = data

    def getFilteredData(self):
        return self.filtered

    def dropBadEpochs(self, plotLog=False):
        reject_criteria = dict(eeg=150e-6) # 150 µV
        flat_criteria = dict(eeg=1e-6) # 1 µV
        self.epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)
        if plotLog: self.epochs.plot_drop_log()

    def fix_bad_channels(self, bad_channels, plot_bads=True):
        self.raw.info["bads"] = bad_channels
        interpolated_data = self.raw.copy().interpolate_bads(reset_bads=False)
        #print("Bads: ", interpolated_data.info['bads'])
        if(plot_bads):
            for title, data in zip(["orig.", "interp."], [self.raw, interpolated_data]):
                print(title, "bands: ", bad_channels)
                with mne.viz.use_browser_backend("matplotlib"):
                    fig = data.plot()
                fig.subplots_adjust(top=0.9)
                fig.suptitle(title, size="xx-large", weight="bold")
        return interpolated_data
    
    # Find bad spans of data using mne.preprocessing.annotate_muscle_zscore
    def markMuscleArtifacts(self, threshold, plot=False):
        threshold_muscle = threshold  # z-score
        annot_muscle, scores_muscle = annotate_muscle_zscore(
        self.raw, ch_type="eeg", threshold=threshold_muscle, min_length_good=0.2,
        filter_freq=[0, 60])
        self.raw.set_annotations(annot_muscle)
    
    def checkLeftNumbers(self, channel):
        return (("1" in channel) or ("3" in channel) or ("5" in channel) or ("7" in channel) or ("9" in channel)) and ("10" not in channel)

    def checkRightNumbers(self, channel):
        return ("2" in channel) or ("4" in channel) or ("6" in channel) or ("8" in channel) or ("10" in channel)

    def getFrontalChannels(self):
        _channels = [self.getChannelIndex(i) for i in self.ch_names if "F" in i]
        return _channels

    def getCentralChannels(self):
        _channels = [self.getChannelIndex(i)  for i in self.ch_names if "C" in i and "F" not in i]
        return _channels

    def getPosteriorChannels(self):
        _channels = [self.getChannelIndex(i)  for i in self.ch_names if ("P" in i or "O" in i) and "C" not in i]
        return _channels

    def getLeftChannels(self):
        _channels = [self.getChannelIndex(i)  for i in self.ch_names if self.checkLeftNumbers(i)]
        return _channels 

    def getRightChannels(self):
        _channels = [self.getChannelIndex(i) for i in self.ch_names if self.checkRightNumbers(i)]
        return _channels

    # https://mne.discourse.group/t/psd-multitaper-output-conversion-to-db-welch/4445
    def scaleEEGPower(self, powerArray):
        powerArray = powerArray * 1e6**2 
        powerArray = (10 * np.log10(powerArray))
        return powerArray
    
    def getPSD(self):
        self.psd = self.epochs.compute_psd(method="welch", fmin=self.fmin, fmax=self.fmax, n_fft=self.sfreq)
        return self.psd

    def getChannelIndex(self, channel):
        return self.ch_names.index(channel)

    def _getRegionAvg(self, epoch, regions, scale=True):
        feature_array = []
        for region in regions:
            region_epoch_psd = epoch[region()]
            if(scale):
                scaled_psd = self.scaleEEGPower(region_epoch_psd)
            else:
                scaled_psd = region_epoch_psd
            region_mean_psd = np.mean(scaled_psd, axis=0)
            feature_array.append(region_mean_psd)
        return feature_array
    
    def getRegionFeatures(self, in_featues, scale=True):
        regions = [self.getFrontalChannels, self.getPosteriorChannels, self.getLeftChannels, self.getRightChannels]
        out = [self._getRegionAvg(epoch, regions, scale=scale) for epoch in in_featues]
        return np.array(out)

    def getExperimentalFeatures(self):
        regions = [self.getFrontalChannels, self.getPosteriorChannels]
        out = [self._getRegionAvg(epoch, regions, scale=True) for epoch in self.psd]
        return np.array(out)

    def linePlot(self, series):
        time = np.arange(series.shape[0])
        plt.figure(figsize=(10, 4))
        plt.plot(time, series, linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Entropy")
        plt.title("Entropy")
        plt.grid(True)
        plt.show()
        plt.savefig('fig.png')


    def getMSE(self):
        #print(self.epochs.get_data().shape)
        Mobj = EH.MSobject('FuzzEn', m = 5, tau = 2, Fx = 'sigmoid', r = (3, 1.2))
        _x =  self.epochs.get_data()[0][0]
        MSx = EH.MSEn(_x, Mobj, 20, Plotx=False)[0]
        #print(MSx.shape)
        #self.linePlot(MSx)
        #print(MSx.shape, CI)
        mse_array = []
        for epoch in self.epochs:
            channel_array = [EH.MSEn(channel, Mobj, 20, Plotx=False)[0] for channel in epoch]
            mse_array.append(channel_array)
        #print(np.array(mse_array).shape)
        return np.array(mse_array)
         
    def getRegionPSDFeatures(self):
        return self.getRegionFeatures(self.psd)
    
    def getRegionRawFeatures(self):
        return self.getRegionFeatures(self.epochs, scale=False)

    def getFileID(self, fileName):
        return fileName.split('Abby')[0].split('/')[-1] #specific to test set 

#import torch
#import torch.nn as nn
#from dataset import load_files


#files = load_files('data_debug/td/', ".set")
#subj1 = Subject(files[0], bad_channels=["CP3"], muscle_thresh=5)
#subj1.getMSE()
#epochs = subj1.getRegionFeatures(subj1.psd)
#flatten_model = nn.Flatten(start_dim=0)
#x = torch.from_numpy(epochs[0])
#x = np.expand_dims(epochs[0], axis=0)
#x = torch.randn(32, 1, 5, 5)
#print("x.shape", x.shape)
# Flatten the sample
#output = flatten_model(x) # perform forward pass
#print("output shape", output.shape)

