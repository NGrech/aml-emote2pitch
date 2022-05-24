# General imports
from traceback import print_tb
from unicodedata import name
import numpy as np

# Music preprocessing imports
import cqhc
import librosa

def sample_song(song_pth:str, sample_rate:int=22050, n_samples:int=3) -> list:
    """Loads a song from a given path, and divides the song into
    n_samples of equal length.
    """
    song, _ = librosa.load(song_pth, sr=sample_rate)
    n_frames = len(song)
    fames_per_sample = n_frames//3

    return [song[fames_per_sample*i:fames_per_sample*(i+1)] for i in range(3)]

def convert_to_QC_spectrogram(sample_name, sub_sample_time=True):
    
    """
    Converts the music category wav-files to constant-Q harmonic coefficients (CQTHCs), using
    the CQHC-Python package: https://github.com/zafarrafii/CQHC-Python
    
    Args:
        sample_name      (str):   Full path to the WAV-file 
        sub_sample_time  (bool):  Switch for subsampling in time
    """

    # takes wav sample and returns spectrogram img.

    # Load the audio file
    audio_signal, sampling_frequency = librosa.load(sample_name, sr=None, mono=True)

    # Define the parameters and compute the CQT spectrogram
    step_length = int(pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency)))) / 2)
    minimum_frequency = 32.70
    octave_resolution = 31  # 51, 12
    cqt_spectrogram = cqhc.cqtspectrogram(audio_signal,
                                          sampling_frequency,
                                          step_length,
                                          minimum_frequency,
                                          octave_resolution)

    w1 = h1 = 256

    if sub_sample_time:

        # Subsampling the time-dimension
        cqts_2 = cqt_spectrogram.copy()
        cqts_2 = cqts_2[:, ::5]

        # Post-subsampling truncate in both time and freq dims to 256x256
        h0 = cqts_2.shape[0]
        w0 = cqts_2.shape[1]
        dh = np.int(np.ceil((h0 - h1) / 2))
        dw = np.int(np.floor((w0 - w1) / 2)) + 1
        cqts_3 = cqts_2.copy()
        cqts_3 = cqts_3[dh:h0-dh, 1:w0-dw]

    else:

        # Truncate in freq dims to 256
        cqts_2 = cqt_spectrogram.copy()
        h0 = cqts_2.shape[0]
        dh = np.int(np.ceil((h0 - h1) / 2))
        cqts_3 = cqts_2.copy()
        cqts_3 = cqts_3[dh:h0-dh, :]

    return cqts_3

def batch_process():
    # process all songs.
    pass

if __name__=='main':
    batch_process()