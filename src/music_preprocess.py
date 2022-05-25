# General imports
import os
from tqdm import tqdm
from logging import root

import librosa
import matplotlib
import numpy as np

matplotlib.use('Agg')

import cqhc
import librosa
import matplotlib.pyplot as plt


def sample_song(song_pth:str, sample_rate:int=22050, n_samples:int=3) -> list:
    """Loads a song from a given path, and divides the song into
    n_samples of equal length.
    """
    song, _ = librosa.load(song_pth, sr=sample_rate)
    if n_samples == 1:
        return [song]
    n_frames = len(song)
    fames_per_sample = n_frames//int(n_samples)

    return [song[fames_per_sample*i:fames_per_sample*(i+1)] for i in range(n_samples)]

def convert_to_QC_spectrogram(audio_signal, sampling_frequency, sub_sample_time=True):
    
    """
    Converts the music category wav-files to constant-Q harmonic coefficients (CQTHCs), using
    the CQHC-Python package: https://github.com/zafarrafii/CQHC-Python
    
    Args:
        audio_signal      (np.array):   Loaded wave file as an np array
        sampling frequency (int):       HZ at with the audio file was sampled   
        sub_sample_time   (bool):       Switch for subsampling in time
    """

    # takes wav sample and returns spectrogram img.

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
        dh = int(np.ceil((h0 - h1) / 2))
        cqts_3 = cqts_2.copy()
        cqts_3 = cqts_3[dh:h0-dh, :]

    return cqts_3

def convert_to_chromagram(sample, sample_rate):
    return librosa.feature.chroma_stft(y=sample, sr=sample_rate)

def batch_process_songs_to_spectrograms(root_dir:str, output_dir:str, genres:list=None, sample_rate:int=22050, splits=3, mode:str='chromagram'):
    """Process all songs in root_dir to 3 even length sample spectrograms"""

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not genres:
        genres = os.listdir(root_dir)

    for genre in genres:
        pth = os.path.join(root_dir, genre)
        song_paths = [os.path.join(pth, s) for s in os.listdir(pth)]
        process_songs(
            song_paths,
            output_dir,
            genre,
            sample_rate,
            splits,
            mode
        )

def process_songs(song_paths:list, out_path:str, genre:str, sample_rate:int=22050, splits=3, mode:str='chromagram'):
    chroma_dir = os.path.join(out_path, 'chromagram', genre, f'{sample_rate}Hz', f'{splits}-splits')
    constant_q_dir = os.path.join(out_path, 'constant-q', genre, f'{sample_rate}Hz', f'{splits}-splits')
    for i, song_path in enumerate(tqdm(song_paths, desc=genre)):
        for j, sample in enumerate(sample_song(song_path, n_samples=splits)):
            if mode == 'chromagram' or mode == 'both':
                if not os.path.isdir(chroma_dir):
                    os.makedirs(chroma_dir)
                spect = convert_to_chromagram(sample, sample_rate)
                pth =  os.path.join(chroma_dir, f'{i:003d}-{j:002d}.png')
                save_spectrogram(spect, pth)
            if mode == 'constant-q' or mode == 'both':
                if not os.path.isdir(constant_q_dir):
                    os.makedirs(constant_q_dir)
                pth =  os.path.join(constant_q_dir, f'{i:003d}-{j:002d}.png')
                spect = convert_to_QC_spectrogram(sample, sample_rate, sub_sample_time=False)
                save_spectrogram(spect, pth)
                pass

def save_spectrogram(spect, pth:str):
    fig = plt.figure()
    plt.figure(figsize=(5, 5))
    plt.pcolormesh(spect, cmap='binary')
    plt.axis('off')
    plt.savefig(pth, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    plt.close('all')

if __name__=='__main__':
    root = os.path.join('data','GTZAN','genres_original')
    output = os.path.join('data', 'Spectrograms')
    genres = ['blues', 'jazz', 'pop', 'metal']
    batch_process_songs_to_spectrograms(root, output, genres,sample_rate=44100, mode='constant-q')

