import librosa


from unicodedata import name


def sample_song(song_pth:str, sample_rate:int=22050, n_samples:int=3) -> list:
    """Loads a song from a given path, and divides the song into
    n_samples of equal length.
    """
    song, _ = librosa.load(song_pth, sr=sample_rate)
    n_frames = len(song)
    fames_per_sample = n_frames//3

    return [song[fames_per_sample*i:fames_per_sample*(i+1)] for i in range(3)]

def convert_to_QC_spectrogram(sample):
    # takes wav sample and returns spectrogram img.
    pass

def batch_process():
    # process all songs.
    pass

if __name__=='main':
    batch_process()