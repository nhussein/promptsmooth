import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import CFG


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.projection(x)

        return x


loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()


class CLIPModel(nn.Module):
    def __init__(
            self,
            temperature=CFG.temperature,
            image_embedding=CFG.image_embedding,
            text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = batch["image"].type(torch.float32).to(CFG.device)
        text_features = batch["caption"].type(torch.float32).to(CFG.device)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        #print(image_embeddings.shape)
        #print(text_embeddings.shape)

        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        #print(torch.norm(image_embeddings))
        #print(torch.norm(text_embeddings))

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        # print(logits[0])
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        #print(logits)

        matrix = []
        for l in batch["label"]:
            # derm7pt
            if CFG.dataset == "derm7pt":
                if l == 0:
                    matrix.append(batch["label"].cpu().numpy())
                else:
                    matrix.append(np.where(batch["label"].cpu().numpy() >= 1, 0, 1))
            elif CFG.dataset == "ISIC_2018":
                matrix.append(np.where(batch["label"].cpu().numpy() != l.cpu().numpy(), 0., 1.))

        #print(np.array(matrix))

        targets = torch.from_numpy(np.array(matrix)).to(CFG.device)
        texts_loss = loss_txt(logits.type(torch.float32), targets.type(torch.float32))
        images_loss = loss_img(logits.T.type(torch.float32), targets.T.type(torch.float32))
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
