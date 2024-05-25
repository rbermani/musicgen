# test_music_tokenizer.py

from tokenizers_rbermani.mubin_tokenizer import MuBinTokenizer
import os
#from pathlib import Path

def test_tokenize_mubin():
    PROJECT_PATH = os.getcwd()
    music_bin_path = os.path.join(PROJECT_PATH,"tests/frelise.bin")
    tokenizer = MuBinTokenizer()
    try:
        result = tokenizer.tokenize(music_bin_path)
    except Exception as e:
        print(f"Exception {e} occurred")
        assert False
    assert True

def test_tokenize_file_exception():
    music_bin_path = "BADFILENAME"
    tokenizer = MuBinTokenizer()
    try:
        result = tokenizer.tokenize(music_bin_path)
    except FileNotFoundError:
        assert True
        return
    assert False
