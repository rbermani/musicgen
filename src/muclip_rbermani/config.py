import torch
config = {
    "debug": True,
    "image_path": "datasets/mubins",
    "captions_path": "datasets/captions",
    "batch_size": 8,
    "num_workers": 0,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "patience": 2,
    "factor": 0.5,
    "epochs": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_name": 'resnet50',
    "mubin_embedding": 2048,
    "text_encoder_model": "distilbert-base-uncased",
    "text_embedding": 768,
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,
    "max_token_length": 64,
    "pretrained": False,  # for both musicbin encoder and text encoder
    "trainable": False,  # for both musicbin encoder and text encoder
    "temperature": 1.0,
    "size": 224,  # image size
    "num_projection_layers": 1,  # for projection head; used for both musicbin and text encoders
    "projection_dim": 256,
    "dropout": 0.1
}