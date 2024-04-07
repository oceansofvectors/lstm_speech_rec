from torch.utils.data import Dataset

from feature_extractor import Mfcc


class MFCCDataset(Dataset):
    def __init__(self, txt_file, directory, transform=None):
        """
        Args:
            txt_file (string): Path to the text file with file paths and labels.
            directory (string): Base directory with all the label directories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        # Manually define the labels based on the list provided
        self.labels = [
            'sheila', 'seven', 'right', 'one', 'house', 'down', 'zero', 'go',
            'yes', 'wow', 'six', 'no', 'three', 'happy', 'bird', 'stop', 'marvin',
            'two', 'five', 'on', 'off', 'four', 'dog', 'up', 'tree', 'cat', 'bed',
            'nine', 'eight', 'left'
        ]
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        with open(txt_file, 'r') as f:
            self.filepaths_labels = [line.strip().split('/') for line in f.readlines()]

    def __len__(self):
        return len(self.filepaths_labels)

    def __getitem__(self, idx):
        mfcc_extractor = Mfcc()
        label_name, filename = self.filepaths_labels[idx]
        full_path = f"{self.directory}/{label_name}/{filename}"
        mfccs = mfcc_extractor.get_mfccs(full_path)
        label = self.label_to_index[label_name]
        sample = {'mfccs': mfccs, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
