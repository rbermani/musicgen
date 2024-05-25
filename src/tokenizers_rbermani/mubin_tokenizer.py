
from pathlib import Path

import struct
import torch
import json
import os

class MuBinTokenizer:
    UNKNOWN_TOKEN = 4356
    PADDING_TOKEN = 0
    """ Convert a MusicBin to tensors
    """
    def __init__(self, max_context_length=4096):
        self.max_context_length = max_context_length
        self.word_to_index = {}
        self.index_to_word = []

    def get_vocab_size(self):
        return len(self.index_to_word)

    def convert_tokens_to_ids(self, tokens):
        """
        Convert a list of unique dword tokens to a list of token IDs

        Args:
            tokens: obj:List(`int`): A list of unique dwords

        Returns:
            obj:List(`int`): A list of token IDs
        """
        return [self._encode(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """
        Convert a list of token ids to a list of unique dwords

        Args:
            tokens: obj:List(`int`): A list of token IDs

        Returns:
            obj:List(`int`): A list of unique dwords
        """
        return [self._decode(id) for id in ids]

    def _decode(self, token_id):
        """
        Convert a token ID back into its unique word from the vocabulary.

        Args:
            token_id (`int`): A token ID in the vocabulary.

        Returns:
            `int`: A unique dword from the vocabulary, or a special control token.
        """
        try:
            # Ensure token_id is an integer
            if not isinstance(token_id, int):
                raise ValueError(f"Token ID must be an integer, got {type(token_id)}")

            # Access the corresponding word from the index_to_word mapping
            return self.index_to_word[token_id]
        except IndexError:
            # Handle the case where token_id is out of bounds
            print(f"Error: Token ID {token_id} is out of bounds.")
            return MuBinTokenizer.UNKNOWN_TOKEN

    def _encode(self, dword):
        """
        Convert an individual unique dword into a token ID, and if it
        doesn't exist in the vocabulary list, add it.

        Args:
            dword (`int`): The dword

        Returns:
            :`int`: The token ID
        """
        if dword not in self.word_to_index:
            # Add new word to vocabulary
            new_token_id = len(self.index_to_word)
            #print(f"new_token_id: {new_token_id}")
            self.word_to_index[dword] = new_token_id
            self.index_to_word.append(dword)
            return new_token_id
        else:
            return self.word_to_index[dword]

    def save(self, path, pretty=True):
        """
        Save the :class:`MusicTokenizer` to the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a file in which to save the serialized tokenizer.

            pretty (:obj:`bool`, defaults to :obj:`True`):
                Whether the JSON file should be pretty formatted.
        """
        if pretty:
            indent = 4
        else:
            indent = 0
        try:
            with open(path, 'w') as file:
                json.dump({"word_to_index": self.word_to_index,
                        "index_to_word": self.index_to_word}, file, indent=indent)
        except IOError as e:
            print("Error saving tokenizer state:", e)

    def from_file(self, path):
        """
        Load the saved :class:`MusicTokenizer` fields from the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a file in which the serialized tokenizer was saved.
        """
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                if "word_to_index" not in data or "index_to_word" not in data:
                    raise ValueError("JSON data does not contain required keys")
                self.word_to_index = data["word_to_index"]
                self.index_to_word = data["index_to_word"]
        except IOError as e:
            print("Error loading tokenizer from file:", e)
        except ValueError as e:
            print("Error loading tokenizer from file:", e)

    def _pad_sequence(self, tokens, max_length):
        if len(tokens) < max_length:
            tokens += [self.PADDING_TOKEN] * (max_length - len(tokens))
        elif len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    def tokenize(self, music_bin_path: Path, padding=True, truncate_output=False):
        all_tokens = []
        if not music_bin_path.exists():
            raise FileNotFoundError(f"MusicBin File path {str(music_bin_path)} does not exist")

        expected_hdr = bytes([ord('M'), ord('u'), ord('B'), ord('i')])
        entry_count = 0
        # Add words to the dictionary
        with open(music_bin_path, 'rb') as f:
            """ Read the entry header """
            hdr = f.read(4)
            if hdr != expected_hdr:
                raise ValueError('Did not find a compatible MusicBin header.')

            """ Read the entry length """
            length_bytes = f.read(4)
            if len(length_bytes) != 4:
                raise ValueError('Did not read expected number of length bytes')

            length = struct.unpack('<I', length_bytes)[0]
            #print(f"MusicBin data payload length is {length}")
            if length % 4 != 0:
                raise ValueError('Token length is not an evenly divisible number of 32-bit dwords')
            #print(repr(chunk))
            """ Read the entry """
            token_count = int(length / 4)
            tokens = []
            for _ in range(token_count):
                dword_int = f.read(4)
                if len(length_bytes) != 4:
                    raise ValueError('Did not read expected number of dword bytes')
                dword_int = struct.unpack('<I', dword_int)[0]
                _token_idx = self._encode(dword_int)
                tokens.append(dword_int)

            print(f"{music_bin_path}: Total tokens encoded are {token_count}")

        if token_count > self.max_context_length:
            if truncate_output:
                tokens = tokens[:self.max_context_length]
            else:
                raise RuntimeError(f"Input {str(music_bin_path)} contains payload that is too long for the context length {self.max_context_length}")
        if padding:
            tokens = self._pad_sequence(tokens, self.max_context_length)

        return tokens