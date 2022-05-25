from torch.utils.data import Dataset

def generate_pairings_csv(output_path:str):
    # generate a csv file for pairing to output path
    # <emote img path>,<spectrogram pth>,<set>
    pass

class EmotePairingDataset(Dataset):

    def __init__(self, parings_pth, set:str='train', transform=None, target_transform=None):
        pass

if __name__=='main':
    generate_pairings_csv()