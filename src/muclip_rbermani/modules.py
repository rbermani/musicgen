import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig

class MusicBinEncoder(nn.Module):
    """
    Encode MusicBins to a fixed size vector
    """

    def __init__(
        self, config
    ):
        super().__init__()
        self.model = timm.create_model(
            config["model_name"], config["pretrained"], num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = config["trainable"]

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["pretrained"]:
            self.model = DistilBertModel.from_pretrained(config["text_encoder_model"])
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = config["trainable"]

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.projection = nn.Linear(config["embedding_dim"], config["projection_dim"])
        self.gelu = nn.GELU()
        self.fc = nn.Linear(config["projection_dim"], config["projection_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        self.layer_norm = nn.LayerNorm(config["projection_dim"])

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

