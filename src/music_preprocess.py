# General imports
import os
from tqdm import tqdm
from logging import root
import importlib

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

def wav2cqt(y, sr, hl, bpo, do_cqt):
    
    """
    Converts wav-files to the constant-Q harmonic coefficients (CQTHCs) using the
    librosa-package: https://librosa.org/doc/latest/generated/librosa.cqt.html
    
    Optional package: https://github.com/zafarrafii/CQHC-Python
    """

    importlib.reload(cqhc)

    if do_cqt:
        cqt = librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hl,
            bins_per_octave=bpo
        )
    else:
        cqt = cqhc.cqtspectrogram(
            y,
            sr,
            hl,
            bpo
        )
    
    return cqt

def cqt2wav(C, sr, hl, bpo):
    
    """
    Converts the constant-Q harmonic coefficients (CQTHCs) to wav-files using the
    librosa-package: https://librosa.org/doc/main/generated/librosa.icqt.html
    """

    wav = librosa.icqt(C=C,
        sr=sr,
        hop_length=hl,
        bins_per_octave=bpo
    )

    return wav

def wav2mel(y, sr, hl, n_mels):

    """
    Converts wav-files to mel-sepctrogram, using the librosa package:
    https://librosa.org/doc/0.9.1/generated/librosa.feature.melspectrogram.html
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        hop_length=hl,
        n_mels=n_mels,
    )

    return mel

def mel2wav(M, sr, hl):

    """
    Converts mel-sepctrogram to wav-files, using the librosa package:
    https://librosa.org/doc/0.9.1/generated/librosa.feature.inverse.mel_to_audio.html

    """
    waw = librosa.feature.inverse.mel_to_audio(
        M=M,
        sr=sr,
        hop_length=hl
    )

    return waw

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
    constant_q_dir = os.path.join(out_path, 'constant-qv2', genre, f'{sample_rate}Hz', f'{splits}-splits')
    mel_dir = os.path.join(out_path, 'mel', genre, f'{sample_rate}Hz', f'{splits}-splits')
    for i, song_path in enumerate(tqdm(song_paths, desc=genre, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
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
                spect = np.real(wav2cqt(sample, sample_rate, 32, 12, 1))
                save_spectrogram(spect, pth)
            if mode == 'mel' or mode == 'both':
                if not os.path.isdir(mel_dir):
                    os.makedirs(mel_dir)
                pth =  os.path.join(mel_dir, f'{i:003d}-{j:002d}.png')
                spect = wav2mel(sample, sample_rate, 512, 400)
                save_spectrogram(spect, pth)

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

    batch_process_songs_to_spectrograms(root, output, genres,sample_rate=22050, mode='constant-q')
    batch_process_songs_to_spectrograms(root, output, genres,sample_rate=44100, mode='constant-q')

