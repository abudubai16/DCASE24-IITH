import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dcase24t6.tokenization.aac_tokenizer import AACTokenizer


class Sec_tfmer(nn.Module):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        nhead: int = 4,
        num_encoder_layer: int = 4,
        activation: str = "gelu",
        loss_scale: float = 0.2,
    ) -> None:
        super().__init__()
        # Save hyperparameters:
        self.d_model = d_model
        self.loss_scale = loss_scale

        # Instantiate the required classes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layer,
            enable_nested_tensor=True,
        )

        self.tokenizer = tokenizer  # vocab size : 4371
        self.vocab_size = 4371

        self.init_embedding_layer()

        self.classifier = nn.Linear(d_model, self.vocab_size)
        self.projection = nn.Linear(self.d_embs, d_model)

        self.fin_projec = nn.Linear(512, 256)

    def init_embedding_layer(self):
        save_location = "/home/akhil/models/DCASE24/dcase2024-task6-baseline/src/dcase24t6/nn/sec_tfm_embeddings.pt"

        if not os.path.exists(save_location):
            model = SentenceTransformer(
                "paraphrase-MiniLM-L6-v2",
                device="cuda",
            )  # d_embs : 384
            embeddings = []

            with tqdm(total=self.vocab_size, desc="Embedding Generation") as pbar:
                for i in range(self.vocab_size):
                    word = self.tokenizer.decode([i])
                    word_embs = model.encode(word, show_progress_bar=False)
                    embeddings.append(torch.from_numpy(word_embs))
                    pbar.update(1)
            del model
            embeddings = torch.stack(embeddings)
            torch.save(embeddings, save_location)
        else:
            embeddings = torch.load(save_location)

        self.vocab_size, self.d_embs = embeddings.shape
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings=embeddings)

    def generate_logits_from_tokens(
        self, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # generate the predicted words
        tokens = torch.argmax(values, dim=1)
        embeddings = self.embedding_layer(tokens)
        logits = self.projection(embeddings)

        # mask the used locations

        return logits, values

    def forward(
        self, embs: torch.Tensor, keywords: List[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """
        embs: (64, 55, 256)
        logits: (64, 256)
        """
        frame_embs = embs["frame_embs"].transpose(-1, -2)
        encoded = self.encoder(frame_embs)
        encoded = torch.mean(encoded, dim=1)
        values = self.classifier(encoded)

        # handle the keyword generation
        logits, values = self.generate_logits_from_tokens(values=values)
        if keywords is not None:
            loss = F.cross_entropy(values, keywords.squeeze(1))
        else:
            loss = 0

        logits = torch.stack([logits] * embs["frame_embs"].shape[2], dim=2)
        embs["frame_embs"] = torch.cat([embs["frame_embs"], logits], dim=1)
        embs["frame_embs"] = self.fin_projec(
            embs["frame_embs"].transpose(-1, -2)
        ).transpose(-1, -2)

        return embs, loss * self.loss_scale


# Local testing
if __name__ == "__main__":
    a = torch.rand(64, 256, 93)
    b = torch.rand(64, 256)
    b = torch.stack([b] * a.shape[2], dim=2)

    print(torch.cat([a, b], dim=1).shape)

    pass
