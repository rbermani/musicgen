import torch

config = {
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "num_classes": 10, # num_classes
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
    "token_limit": 512,
    "mubin_token_limit" : 4096,
    "vocab_size": 512,
    "embedding_dim": 48,
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
