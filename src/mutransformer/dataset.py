
import os
from typing import List, Optional
from random import randint, choice
from pathlib import Path
from sklearn.model_selection import train_test_split
from tokenizers_rbermani.mubin_tokenizer import MuBinTokenizer

from .config import config
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers.utils import PaddingStrategy

class MuBinDataset(Dataset):
    def __init__(self,
                 config,
                 text_prompt_files : List[Path],
                 mubin_files : List[Path],
                 mubin_target_files : List[Path]
                 ):
        """
        @param config: global configuration dictionary
        @param mubin_files: dictionary containing mubin file names
        """
        super(MuBinDataset, self).__init__()
        #assert(len(mubin_files) == len(text_prompt_files))
        assert(len(text_prompt_files) == len(mubin_target_files))
        self.text_prompt_ntoken = config["text_prompt_ntoken"]
        self.mubin_token_limit = config["mubin_ntoken"]
        self.mubin_tokenizer = MuBinTokenizer(config["mubin_ntoken"])
        self.mubin_files = mubin_files
        self.textprompt_files = text_prompt_files
        self.mubin_target_files = mubin_target_files

        self.text_tokenizer = DistilBertTokenizer.from_pretrained(config["distilbert_model"])

        # Initializing a pretrained DistilBERT configuration
        distilbert_config = DistilBertConfig.from_pretrained(config["distilbert_model"], max_position_embeddings=self.text_prompt_ntoken)
        # Initializing a pretrained DistilBERT model from the configuration
        self.distilbert_model = DistilBertModel(distilbert_config)
        # print(f"mubin_files:{self.mubin_files}")
        # print(f"textprompt_files:{self.textprompt_files}")
        # print(f"mubin_target_files:{self.mubin_target_files}")

    def __len__(self):
        return len(self.mubin_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        mubin_input = self.mubin_files[idx]
        text_file = self.textprompt_files[idx]
        mubin_target_file = self.mubin_target_files[idx]
        with open(text_file, 'r') as f:
            text_input = f.read()

        if mubin_input.exists():
            tokenized_mubin_input = self.mubin_tokenizer.tokenize(mubin_input)
        else:
            # Fill tokenized input with padding tokens up to the context limit
            tokenized_mubin_input = [MuBinTokenizer.PADDING_TOKEN for _ in range(self.mubin_token_limit )]

        tokenized_mubin_target = self.mubin_tokenizer.tokenize(mubin_target_file)

        with torch.no_grad():
            tokenized_prompt = self.text_tokenizer(text_input,
                                               return_tensors='pt',
                                               max_length=self.text_prompt_ntoken,
                                               truncation=True,
                                               padding=PaddingStrategy.MAX_LENGTH)
            distil_input_ids = tokenized_prompt['input_ids']
            distil_attention_mask = tokenized_prompt['attention_mask']
            outputs = self.distilbert_model(distil_input_ids, attention_mask=distil_attention_mask)
            # The last_hidden_state is used instead of normal output
            # to apply the custom attention_mask for a truncated token limit
            # squeeze the token dimension 0 to remove the unnecessary batch dimension 1
            encoded_prompt = outputs.last_hidden_state.squeeze(0)
            #print(f"encoded_prompt shape: {encoded_prompt.shape}")

        tensor_mubin_input = torch.tensor(self.mubin_tokenizer.convert_tokens_to_ids(tokenized_mubin_input))
        tensor_mubin_target = torch.tensor(self.mubin_tokenizer.convert_tokens_to_ids(tokenized_mubin_target))

        #print(f"encoded_prompt shape: {encoded_prompt.shape}")
        return {'encoded_prompt': encoded_prompt,
                'input_ids': tensor_mubin_input,
                'target_ids': tensor_mubin_target}

def verify_files_exist(file_list):
    for file_name in file_list:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")
        elif not os.path.isfile(file_name):
            raise ValueError(f"File '{file_name}' is not a regular file.")
        else:
            continue

def collate_fn(samples):
    encoded_prompts = [sample['encoded_prompt'].clone().detach() for sample in samples]
    input_ids = [sample['input_ids'].clone().detach() for sample in samples]
    target_ids = [sample['target_ids'].clone().detach() for sample in samples]

    # Pad sequences to the same length
    prompts_padded = pad_sequence(encoded_prompts, batch_first=True, padding_value=0)
    inputs_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)

    return ((prompts_padded, inputs_ids_padded), target_ids_padded)

def prepare_data(batch_size=4, num_workers=2, train_sample_size: Optional[int] = None, val_sample_size: Optional[int] = None):
    mubin_dir = "../data/train/mubins"
    textprompts_dir = "../data/train/textprompts"
    target_mubins_dir = "../data/train/target_mubins"

    textprompt_filenames =[f for f in os.listdir(textprompts_dir) if f.endswith('.txt')]
    mubin_input_filenames = [f.replace('.mbin', '.txt') for f in textprompt_filenames]
    mubin_target_filenames = [f for f in os.listdir(target_mubins_dir) if f.endswith('.mbin')]
    # Convert to full paths
    mubin_input_filenames = [Path(mubin_dir, f) for f in mubin_input_filenames]
    textprompts_filenames = [Path(textprompts_dir, f) for f in textprompt_filenames]
    mubin_target_filenames = [Path(target_mubins_dir, f) for f in mubin_target_filenames]
    try:
        # Ensure there is a one-to-one mapping between mubins and label files
        verify_files_exist(textprompts_filenames)
        verify_files_exist(mubin_target_filenames)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return None

    # Split the files into training and validation sets
    train_textprompts, val_textprompts, train_mubins, val_mubins, train_target_mubins, val_target_mubins = train_test_split(textprompts_filenames, mubin_input_filenames, mubin_target_filenames, test_size=0.2, random_state=None, shuffle=False)
    # print(f"train_textprompts {train_textprompts}")
    # print(f"val_textprompts {val_textprompts}")
    # print(f"Train MuBins {train_mubins}")
    # print(f"Train Labels {train_target_mubins}")
    # print(f"Val MuBins {val_mubins}")
    # print(f"Val Labels {val_target_mubins}")
    train_data = MuBinDataset(config, train_textprompts, train_mubins, train_target_mubins)
    val_data = MuBinDataset(config, val_textprompts, val_mubins, val_target_mubins)

    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(train_data))[:train_sample_size].tolist()
        train_data = torch.utils.data.Subset(train_data, indices)
    if val_sample_size is not None:
        # Randomly sample a subset of the validation set
        indices = torch.randperm(len(val_data))[:val_sample_size].tolist()
        val_data = torch.utils.data.Subset(val_data, indices)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataloader, val_dataloader