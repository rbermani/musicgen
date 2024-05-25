# Import libraries
import torch
import os
from torch.utils.data import Dataset, DataLoader
from random import randint, choice
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizers_rbermani.mubin_tokenizer import MuBinTokenizer
from mut.config import config

class MuBinDataset(Dataset):
    def __init__(self,
                 config,
                 mubin_files,
                 binlabels,
                 music_tokenizer=None
                 ):
        """
        @param files: dictionary containing musicbin file names
        """
        super().__init__()
        assert(len(mubin_files) == len(binlabels))
        self.mubin_token_limit = config["mubin_token_limit"]
        self.music_tokenizer = music_tokenizer
        self.mubin_files = mubin_files
        self.binlabels = binlabels

        # print(f"mubin_files:{self.mubin_files}")
        # print(f"binlabels:{self.binlabels}")

    def __len__(self):
        return len(self.mubin_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        music_file = self.mubin_files[idx]
        tokenized_music = self.music_tokenizer.tokenize(
            music_file,
            self.mubin_token_limit
        ).squeeze(0)
        sample = {'data': tokenized_music, 'target': self.binlabels[idx]}
        return self._to_tensor(sample)

    def _to_tensor(self, sample):
        data, target = sample['data'], sample['target']
        return {'data': torch.tensor(data), 'target': torch.tensor(target)}  # Convert target to tensor

def verify_files_exist(file_list):
    for file_name in file_list:
        #full_path = os.path.join(root_directory, file_name)
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")
        elif not os.path.isfile(file_name):
            raise ValueError(f"File '{file_name}' is not a regular file.")
        else:
            continue

def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, val_sample_size=None):
    mubin_dir = "../data/train/mubins"
    target_labels_dir = "../data/train/labels"
    #labels_files = os.listdir(target_labels_dir)
    mubin_input_filenames = [f for f in os.listdir(mubin_dir) if f.endswith('.mbin')]
    target_label_filenames = [f.replace('.mbin', '.txt') for f in mubin_input_filenames]
    # Convert to full paths
    mubin_input_filenames = [Path(mubin_dir, f) for f in mubin_input_filenames]
    target_label_filenames = [Path(target_labels_dir, f) for f in target_label_filenames]
    try:
        # Ensure there is a one-to-one mapping between mubins and label files
        verify_files_exist(target_label_filenames)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    # Use the target_label_filenames to create a consolidated list of all possible labels in the complete dataset
    # Also create a dictionary mapping of mubins to associated labels
    # Initialize a set to collect unique classes
    # unique_classes = set()
    # Read all labels and aggregate unique classes
    all_label_groups = []
    mubin_to_labels = {}
    for mubin_file, target_file in zip(mubin_input_filenames, target_label_filenames):
        with open(target_file, 'r') as f:
            labels = f.read().strip().split()
            all_label_groups.append(labels)
            mubin_to_labels[os.path.basename(mubin_file)] = labels
    # Convert the set of unique classes to a list
    # classes = list(unique_classes)
    # print(f"Unique: {classes}")
    # print(f"All Labels: {all_label_groups}")
    # print(f"Mubin_to_labels: {mubin_to_labels}")

    # Extract file names and label sets from the dictionary items
    file_names = list(mubin_to_labels.keys())
    label_sets = list(mubin_to_labels.values())
    # Fit the MultiLabelBinarizer on the entire set of labels
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(label_sets)
    unique_labels = mlb.classes_
    #num_samples, num_classes = labels.shape
    #print(f"file_names {file_names} unique_labels {unique_labels}")
    # print(f"num_samples {num_samples} num_classes {num_classes}")
    # print(f"labels {labels}")
    # Split the files into training and validation sets
    train_mubins, val_mubins, train_labels, val_labels = train_test_split(file_names, labels, test_size=0.2, random_state=None, shuffle=False)
    # print(f"Train MuBins {train_mubins}")
    # print(f"Train Labels {train_labels}")
    # print(f"Val MuBins {val_mubins}")
    # print(f"Val Labels {val_labels}")
    music_tokenizer = MuBinTokenizer()
    train_data = MuBinDataset(config, train_mubins, train_labels, music_tokenizer=music_tokenizer)
    val_data = MuBinDataset(config, val_mubins, val_labels, music_tokenizer=music_tokenizer)

    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(train_data))[:train_sample_size]
        train_data = torch.utils.data.Subset(train_data, indices)
    if val_sample_size is not None:
        # Randomly sample a subset of the validation set
        indices = torch.randperm(len(val_data))[:val_sample_size]
        val_data = torch.utils.data.Subset(val_data, indices)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, unique_labels