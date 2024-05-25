import torch
from torch import nn
import torch.nn.functional as F

from modules import TextEncoder, ProjectionHead, MusicBinEncoder


class MuClipModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.mubin_encoder = MusicBinEncoder()
        self.text_encoder = TextEncoder()
        self.mubin_projection = ProjectionHead(embedding_dim=config["mubin_embedding"])
        self.text_projection = ProjectionHead(embedding_dim=config["text_embedding"])
        self.temperature = config["temperature"]

    def forward(self, batch):
        # Getting MusicBin and Text Features
        mubin_features = self.mubin_encoder(batch["mubin"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting MusicBin and Text Embeddings (with same dimension)
        mubin_embeddings = self.mubin_projection(mubin_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ mubin_embeddings.T) / self.temperature
        mubins_similarity = mubin_embeddings @ mubin_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (mubins_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        mubins_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (mubins_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    mubins = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'mubin': mubins,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    MuCLIP = MuClipModel()
    loss = MuCLIP(batch)
    print("")