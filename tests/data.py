import os
from io import open
import torch
import struct
import sys

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'concatenated.bin'))

        print(f"size {self.train.size()}, elems {self.train.numel()}")
        #self.valid = self.tokenize(os.path.join(path, 'valid.bin'))
        #self.test = self.tokenize(os.path.join(path, 'test.bin'))

    def tokenize(self, path):
        """Tokenizes a music bin file."""
        assert os.path.exists(path)
        expected_values = bytes([ord('M'), ord('u'), ord('B'), ord('i')])
        entry_count = 0
        # Add words to the dictionary
        with open(path, 'rb') as f:
            """ Read the entry header """
            chunk = f.read(4)
            while chunk:
                #print(repr(chunk))
                if chunk == bytes(expected_values):
                    """ Read the entry length """
                    length_bytes = f.read(4)
                    if len(length_bytes) == 4:
                        length = struct.unpack('<I', length_bytes)[0]
                        print(f"Entry {entry_count} is length {length}")
                        """ Read the entry """
                        count = length / 4
                        while count > 0:
                            chunk = f.read(4)
                            self.dictionary.add_word(chunk)
                            count -= 1
                        print(f"Finished reading entry {entry_count}")
                        entry_count += 1
                    else:
                        print("Did not read expected number of length bytes")
                        sys.exit(1)
                    """ Read next entry header """
                    chunk = f.read(4)
                    continue
                else:
                    print("Did not find matching entry header.")
                    sys.exit(1)

            print(f"Dictionary size is {len(self.dictionary)}")

        # Tokenize file content
        # with open(path, 'rb') as f:
        #     idss = []
        #     chunk = f.read(4)
        #     while chunk:
        #         ids = []
        #         ids.append(self.dictionary.word2idx[chunk])
        #         idss.append(torch.tensor(ids).type(torch.int32))
        #         chunk = f.read(4)
        #     ids = torch.cat(idss)
        entry_count = 0
        # Tokenizes words based on dictionary
        with open(path, 'rb') as f:
            idss = []
            ids = []
            """ Read the entry header """
            chunk = f.read(4)
            while chunk:
                #print(repr(chunk))
                if chunk == bytes(expected_values):
                    """ Read the entry length """
                    length_bytes = f.read(4)
                    if len(length_bytes) == 4:
                        length = struct.unpack('<I', length_bytes)[0]
                        print(f"Entry {entry_count} is length {length}")
                        """ Read the entry """
                        count = length / 4
                        while count > 0:
                            chunk = f.read(4)
                            idx = self.dictionary.word2idx[chunk]
                            #print(f"index is {idx}")
                            ids.append(idx)
                            count -= 1
                        print(f"Finished reading entry {entry_count}")
                        entry_count += 1
                    else:
                        print("Did not read expected number of length bytes")
                        sys.exit(1)
                    """ Read next entry header """
                    chunk = f.read(4)
                    continue
                else:
                    print("Did not find matching entry header.")
                    sys.exit(1)
            print(f"ids is len {len(ids)}")
            idss.append(torch.tensor(ids).type(torch.int32))
            ids = torch.cat(idss)
        return ids