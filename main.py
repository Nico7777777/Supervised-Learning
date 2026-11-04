import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (GroupKFold, KFold, LeaveOneOut, LeaveOneGroupOut, LeavePOut,
                                    GridSearchCV, RandomizedSearchCV, ParameterSampler)
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from typing import Dict, List
import importlib
import numpy as np
import IPython.display as ipd
import pandas as pd
import random
import os

classes = ['Cough', 'DoorOpenClose', 'DrawerOpenClose', 'KeyboardTyping',
           'KeyJingle', 'Knock', 'MouseButtonClick', 'PaperTear', 'Pour',
           'SwitchToggle', 'Walk', 'ZipperZip']
librosa = importlib.import_module("librosa")

'''
AUDIO info:
WAV, PCM, 16 bits, 
mono - 16kHz
'''

VISUAL_FILE = lambda x : os.path.join('visuals', x)
SPECTOGRAM_FILE = lambda x : os.path.join('visuals', 'spectograms', x)
CLASS_FILES = lambda x : os.listdir(os.path.join('audio', x))
GET_ENTRY = lambda class_, file_: os.path.join('audio', class_, file_)
train, test, val = list(), list(), list()

def extract_level_features(clip: np.ndarray) -> np.ndarray:
    """Return level-related features for one 32,000-sample audio clip.

    The input is assumed to be normalized to the floating-point range
    ``[-1, 1]``, so 0 dBFS corresponds to a full-scale sample.

    The returned features are, in order: global RMS in dBFS, crest factor
    in dB, estimated noise floor in dBFS, estimated signal level in dBFS,
    and approximate SNR in dB.
    """
    samples = np.asarray(clip, dtype=np.float32).reshape(-1)
    if samples.size != 32_000:
        raise ValueError(
            f"clip must contain exactly 32000 samples, got {samples.size}"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("clip must contain only finite samples")

    epsilon = np.finfo(np.float32).tiny
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    rms_dbfs = 20.0 * np.log10(max(rms, epsilon))
    crest_db = 20.0 * np.log10(max(peak, epsilon) / max(rms, epsilon))

    frame_rms = librosa.feature.rms(
        y=samples,
        frame_length=512,
        hop_length=256,
        center=False,
    )[0]
    frame_levels_dbfs = 20.0 * np.log10(np.maximum(frame_rms, epsilon))
    noise_floor_dbfs = float(np.percentile(frame_levels_dbfs, 10))
    signal_level_dbfs = float(np.percentile(frame_levels_dbfs, 95))
    snr_db = signal_level_dbfs - noise_floor_dbfs

    return np.array(
        [rms_dbfs, crest_db, noise_floor_dbfs, signal_level_dbfs, snr_db],
        dtype=np.float32,
    )


def plot_spectrogram_1(signal, name: str):
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(signal)))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency Power Spectrogram for {name}")
    plt.xlabel("Time")
    plt.savefig(SPECTOGRAM_FILE(f'spectogram_{name}_v1.png'))

def plot_spectogram_2(signal, name: str):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X) # aici X-ul este liniar, deci axa orizontala e echidstanta
    f = np.linspace(0, _, len(X_mag))
    plt.figure(figsize=(18, 10))
    plt.plot(f, X_mag) # magnitude spectrum
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.title(f'Linear Power Spectogram for {name}')
    plt.savefig(SPECTOGRAM_FILE(f'spectogram_{name}_v2.png'))

train_test_val = {
    'Cough': (120, 15, 15),
    'DoorOpenClose': (120, 15, 15),
    'DrawerOpenClose': (94, 12, 11),
    'KeyJingle': (83, 10, 11),
    'KeyboardTyping': (67, 8, 9),
    'Knock': (100, 12, 13),
    'MouseButtonClick': (60, 8, 7,),
    'PaperTear': (105, 14, 13),
    'Pour': (120, 15, 15),
    'SwitchToggle': (120, 15, 15),
    'Walk': (120, 15, 15),
    'ZipperZip': (120, 15, 15)
}

if __name__ == "__main__":

    count = {class_: len(os.listdir(os.path.join('audio', class_))) for class_ in os.listdir('audio')}
    print(count.keys())
    plt.title('Multinomial Distribution Sounds')
    plt.pie(x=count.values(), labels=list(count.keys()), rotatelabels=True)
    plt.savefig(VISUAL_FILE('PIE_CHART.png'))
    for class_ in os.listdir('audio'):
        class_entries = CLASS_FILES(class_)
        a, b, c = train_test_val[class_]
        print(f'class={class_}; a={a}, b={b}, c={c}; total_entries={len(class_entries)}')
        train.extend(class_entries[:a])
        test.extend(class_entries[a:a+b])
        val.extend(class_entries[a+b:])
    print(f'train size: {len(train)}')
    print(f'test size: {len(test)}')
    print(f'val size: {len(val)}')




    examples = [('Cough', '01167.wav'), ('DrawerOpenClose', '04715.wav'), ('PaperTear', '08715.wav'),
                ('ZipperZip', '04039.wav'), ('MouseButtonClick', '01750.wav'), ('Knock', '00098.wav')]
    for (c, e) in examples:
        c1, _ = librosa.load(GET_ENTRY(c, e))
        plot_spectrogram_1(c1, f'{c}_{e}.png')

    # y, _ = librosa.loadx(GET_ENTRY(examples[0], examples[1]))
        S, phase = librosa.magphase(librosa.stft(c1))
        rms = librosa.feature.rms(S=S)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(rms)

        ax[0].semilogy(times, rms[0], label='RMS Energy')
        ax[0].set(xticks=[])
        ax[0].legend()
        ax[0].label_outer()
        librosa.display.specshow(S, vscale='dBFS',
                            y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set(title='log Power spectrogram')
        plt.savefig(f'spectogram{c}_v2.png')
    # df = pd.DataFrame(columns=[''])