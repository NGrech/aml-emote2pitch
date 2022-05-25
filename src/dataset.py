import csv
import os

from torch.utils.data import Dataset
from tqdm import tqdm


def generate_pairings_csv(output_path:str, img_src:str, spectrograms_data:tuple, pairings:list, train_split:float):
    """Generates a pairing file with paired images to spectrograms which are outputted as a csv in the format: <emote img path>,<spectrogram pth>,<set>
    Args:
        output_path:str         Path to csv file to output 
        img_src:str             Base path to images to pair
        spectrogram_data:tuple   Tuple of spectrogram data, Base path of spectrogram images, sample rate, sample split
        pairings:list           List of tuples to pair imgs nd spectrogram categories
        train_split:float       Percent to split data to train and test
    """
    spectrogram_src, sample_rate, sample_splits = spectrograms_data
    all_pairings = []
    for emote, genre in tqdm(pairings):
        # Get spectrogram pths
        spect_base_pth = os.path.join(spectrogram_src, genre, sample_rate, sample_splits)
        spect_paths = [os.path.join(spect_base_pth, fn) for fn in os.listdir(spect_base_pth)]

        # Get emot image paths
        emote_base_pth = os.path.join(img_src, emote)
        emote_paths = [os.path.join(emote_base_pth, fn) for fn in os.listdir(emote_base_pth)]

        # test train split 
        split_idx = int(len(spect_paths)*train_split)
        d_set = ['train']*len(spect_paths)
        d_set[split_idx:] = ['test']*len(d_set[split_idx:])

        # pairing
        c_pairing = zip(emote_paths[:len(spect_paths)], spect_paths, d_set)
        all_pairings += [(e, spec, _set) for e, spec, _set in c_pairing]

    with open(output_path, 'w', newline='') as fw:
        csv_out=csv.writer(fw)
        csv_out.writerows(all_pairings)

class EmotePairingDataset(Dataset):

    def __init__(self, parings_pth, set:str='train', transform=None, target_transform=None):
        pass

if __name__=='__main__':

    img_src = os.path.join('data','FER','train')
    spectrogram_type = 'constant-q'
    sample_rate = '22050Hz'
    sample_splits = '3-splits'
    spectrogram_data = (
        os.path.join('data','Spectrograms',spectrogram_type),
        sample_rate,
        sample_splits
    )
    pairings = [
        ('happy', 'pop'),
        ('angry', 'metal'),
        ('sad', 'blues'),
        ('surprise', 'jazz')
    ]
    train_split = 0.9

    output_path = os.path.join('data', 'pairings', f'FER2{spectrogram_type}-{sample_splits}-{sample_rate}.csv')

    generate_pairings_csv(output_path, img_src, spectrogram_data, pairings, train_split)
