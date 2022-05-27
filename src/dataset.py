import csv
import os

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
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

    def __init__(self, pairings_csv_pth, set: str='train', transform=None, target_transform=None, **kwargs) -> None:

        """
        Creates a Pytorch dataset that a Pytorch dataloader can load in a train loop
        
        Args:
            pairings_csv_pth:   Path to csv-file containing paring paired img paths.
            set (str):          Switch for creating a train or test dataset.
            transform:          Transformations for emotion face images (GAN conditional)
            target_transform:   Transformations for spectrograms
        """

        super().__init__()

        self.set = set

        self.transform = transform
        self.target_transform = target_transform

        self.csv_path = pairings_csv_pth

        self.csv_data_train = []
        self.csv_data_test = []

        with open(self.csv_path, 'r', newline='') as fr:
            
            csv_in = csv.reader(fr)

            cnt = 0
            for emo_p, sptg_p, train_test in csv_in:
                if train_test == 'train' and self.set == 'train':
                    self.csv_data_train.append([emo_p, sptg_p])
                    cnt += 1
                elif train_test == 'test' and self.set == 'test':
                    self.csv_data_test.append([emo_p, sptg_p])
                    cnt += 1

        self.csv_data_len = cnt

    def __getitem__(self, idx):

        if self.set == 'train':
            emo_p = self.csv_data_train[idx][0]
            sptg_p = self.csv_data_train[idx][1]
        elif self.set == 'test':
            emo_p = self.csv_data_test[idx][0]
            sptg_p = self.csv_data_test[idx][1]

        x = Image.open(emo_p)
        y = Image.open(sptg_p)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.csv_data_len

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
    train_split = 1

    output_path = os.path.join('data', 'pairings', f'FER2{spectrogram_type}-{sample_splits}-{sample_rate}.csv')

    generate_pairings_csv(output_path, img_src, spectrogram_data, pairings, train_split)
