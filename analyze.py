from numpy import floor
from numpy.core.fromnumeric import std
from numpy.fft import fft
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftshift
from matplotlib import pyplot as plt
from matplotlib import lines
import numpy as np
import math

class WavDataDual: 
    def __init__(self, data, rate, target_rate) -> None:
        self.data = data
        self.rate = rate
        self.target_rate = target_rate
        self.size = int(math.ceil( audioData.shape[0] * target_rate / rate))
        self.shape = [self.size]


    def size(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[int(index * self.rate / self.target_rate), 0]
    
    def __len__(self):
        return len(self.data)

filename = "Recorded_Data_1.wav"

rate, audioData = wavfile.read(filename)
# rate, audioData = wavfile.read("StarWars60.wav")
# rate, audioData = wavfile.read("WanShow_November_19_2021.wav")

print("Original Rate:", rate)
print("Downsampled Rate:", rate)

target_rate = 2000
duration = 60 * 50

if len(audioData.shape) == 2 and audioData.shape[1] == 2:
    wavData = WavDataDual(audioData, rate, target_rate)
else:
    wavData = audioData
    target_rate = rate

segSize = 0.05
nperseg = int(target_rate/segSize)
nfft = nperseg

f_ax, t_ax, Sxx = signal.stft(wavData, target_rate, nperseg=nperseg)
max_f = target_rate 

def get_f_index(f):
    return int(f * nfft / (max_f))

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return math.sqrt(max(variance,0))

base = 50
peak_fs = []
snrs = []

print(f_ax[get_f_index(base * (1 - 0.02))])
print(f_ax[get_f_index(base * (1 + 0.02))])

fig, (ax1, ax2) = plt.subplots(1,2)


modes = [1,3,5,7,9]
for mode in modes:
    lower = base * (1 - 0.02) * mode 
    upper = base * (1 + 0.02) * mode
    low_i = get_f_index(lower)
    upper_i = get_f_index(upper)

    mean = np.array( [ np.average(f_ax[low_i:upper_i], weights=Sxx_t[low_i:upper_i]) / mode for Sxx_t in np.transpose(Sxx) ] ) 
    stddev =  np.array( [ weighted_avg_and_std(f_ax[low_i:upper_i], weights=Sxx_t[low_i:upper_i]) / mode for Sxx_t in np.transpose(Sxx) ] )
    snr = np.where(stddev==0, 0, mean / stddev)        

    print()
    l = np.argmax(Sxx[low_i:upper_i,:],axis=0) + int(low_i)
    peak_f = np.array([ f_ax[i]  / mode for i in l])

    peak_fs.append(peak_f)
    snrs.append(snr)

    # print(snr)
    print("Mode:", mode)
    print("std dev:", np.median(stddev))
    print("snr", np.mean(snr))
    ax1.errorbar(t_ax, peak_f,yerr=None, fmt='.')

ax1.legend(modes, title='Mode')
ax1.set(xlabel="Time (sec)", ylabel="Frequency (Hz)", ylim=(48,52), title="STFT for each mode")
ax2.set(xlabel="Time (sec)", ylim=(48,52), title='Averaged STFT')

total_mean = np.average(peak_fs, weights=None, axis=0)
total_std = np.std(peak_fs,axis=0)
ax2.errorbar(t_ax,total_mean, fmt='.')
w = 1/ total_std
fig.suptitle('Electrical Network Frequency Analysis')
plt.savefig("./images/" + filename[:-4] + ".png")
plt.show()
