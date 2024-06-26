import os

import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from aac_metrics.functional.fense import fense
from pytorch_metric_learning import losses, miners
from transformers import AutoModel, AutoTokenizer

from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad
def get_cosine_similarity(word1: str, word2: str):
    if word1 == word2:
        return torch.tensor(1)

    words = tokenizer(
        [word1, word2], padding=True, truncation=True, return_tensors="pt"
    ).to("cuda")

    embs = model(**words)

    sentences_embed = mean_pooling(embs, words["attention_mask"])
    sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
    return F.cosine_similarity(sentences_embed[0, :], sentences_embed[1, :], dim=0)


def generate_similarity_matrix(
    tokenizer: AACTokenizer, vocab_size: int | None = None, cutoff: float = 0.9
):
    if vocab_size is None:
        vocab_size = tokenizer.get_vocab_size
    # place holder matrix
    similarity_matrix = torch.zeros(vocab_size, vocab_size)

    for i in range(vocab_size):
        for j in range(i + 1):
            similarity_matrix[i, j] = get_cosine_similarity(
                tokenizer.decode([i]), tokenizer.decode([j])
            )

    for i in range(vocab_size):
        for j in range(vocab_size):
            if j > i:
                similarity_matrix[i, j] = similarity_matrix[j, i]

    # Create a cutoff so that words that have no similarity don't dilute the meaning of those words with close meanings
    similarity_matrix = torch.where(similarity_matrix > cutoff, similarity_matrix, 0)
    similarity_matrix = F.softmax(similarity_matrix, dim=1)

    return similarity_matrix


class Similarity_Check(nn.Module):
    def __init__(self, tokenizer: AACTokenizer, vocab_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.cutoff = 0.9
        self.file_dir = (
            "/home/akhil/models/DCASE24/dcase2024-task6-baseline/src/dcase24t6/nn"
        )
        self.file_name = f"sim_mat_{str(100*self.cutoff)}.pt"
        self.file_path = f"{self.file_dir}/{self.file_name}"

        if self.file_name not in os.listdir(self.file_dir):
            print("Generating Cosine Similarity Matrix: \n")
            self.sim_matrix: torch.Tensor = generate_similarity_matrix(
                tokenizer=tokenizer, vocab_size=vocab_size, cutoff=0.5
            )
            torch.save(
                self.sim_matrix,
                self.file_path,
            )
            print("Cosine Similarity Matrix Computed")
        else:
            print("Loading similarity matrix:\n")
            self.sim_matrix = torch.load(self.file_path)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        assert logits.ndim == 3
        assert targets.ndim == 2

        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        targets = torch.cat([self.sim_matrix[int(i)].unsqueeze(0) for i in targets]).to(
            "cuda"
        )

        logits = F.normalize(logits, dim=1)
        return F.cosine_embedding_loss(
            logits, targets, target=torch.ones(targets.shape[0], device="cuda")
        )


def get_fense_scores(
    tokenizer: AACTokenizer, logits: torch.Tensor, target: torch.Tensor
):
    pred_captions = torch.argmax(logits, dim=1)
    pred_captions = [
        tokenizer.decode(list(pred_caption)) for pred_caption in pred_captions
    ]
    captions = [[tokenizer.decode(list(caption))] for caption in target]
    return fense(pred_captions, captions, batch_size=64)[0]["fense"]


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X: torch.Tensor, T: torch.Tensor):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(
            dim=0
        )
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(
            dim=0
        )

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(
            num_classes=self.nb_classes,
            embedding_size=self.sz_embed,
            softmax_scale=self.scale,
        ).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(
            self.scale_pos, self.scale_neg, self.thresh
        )

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets="semihard")
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(
            l2_reg_weight=self.l2_reg, normalize_embeddings=False
        )

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


if __name__ == "__main__":
    """
    logits : (64, 4371, 22)
    target : (64, 22)
    """

    batch_size = 64
    seq_length = 21

    num_proxies = 4371
    embed_dim = 4371
    logits = torch.randn(batch_size, seq_length, num_proxies).view(-1, embed_dim)
    targets = (torch.rand(batch_size * 21) * num_proxies).int()
    print(logits.shape, targets.shape)
    proxy_loss_fn = Proxy_Anchor(nb_classes=num_proxies, sz_embed=embed_dim)
    print(proxy_loss_fn(logits, targets))

    pass
