
config = {
    "nhead": 16,
    "d_hid": 4 * 48, # 4 * hidden_size
    "nlayers": 16,
    "dropout": 0.5,
    "pos_dropout": 0.1,
    "max_len": 5000,
    "text_prompt_ntoken": 128,
    "mubin_ntoken": 4096,
    "mubin_vocabsize": 51000,
    "distilbert_model": "distilbert-base-uncased",
    "mubin_embedding_dim": 768,
}

