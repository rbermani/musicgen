from pathlib import Path
from random import randint, choice
import PIL
from torch.utils.data import Dataset

class TextMusicDataset(Dataset):
    def __init__(self,
                 folder,
                 max_text_len=256,
                 truncate_captions=False,
                 text_tokenizer=None,
                 music_tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing musicbins and text files matched by their path's respective "stem"
        @param truncate_captions: Captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        music_files = [
            *path.glob('**/*.mbin')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        music_files = {music_file.stem: music_file for music_file in music_files}

        keys = (music_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.music_files = {k: v for k, v in music_files.items() if k in keys}
        self.max_text_len = max_text_len
        self.truncate_captions = truncate_captions
        self.text_tokenizer = text_tokenizer
        self.music_tokenizer = music_tokenizer

    def __len__(self):
        return len(self.keys)

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

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        music_file = self.music_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.text_tokenizer.tokenize(
            description,
            self.max_text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)

        tokenized_music = self.music_tokenizer.tokenize(
            music_file,
            self.max_music_len
        ).squeeze(0)

        # Success
        return tokenized_text, tokenized_music
