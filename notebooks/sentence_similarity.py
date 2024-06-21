import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Load the required models from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda")


# Mean Pooling - Take attention mask into account for correct averaging
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


# Sentences we want sentence embeddings for
def find_embeddings(filename: str, num_lines: int | None = None):
    df = pd.read_csv(filename)
    df["caption_5"] = df["caption_5"][:-2]

    if num_lines is None:
        num_lines = len(df)

    # Tokenize sentences
    filename_similarity_dict = {}
    embeddings = []
    for pos in tqdm(range(num_lines)):

        filename = df["file_name"][pos]
        sentences = [df[f"caption_{i}"][pos] for i in range(1, 6)]

        try:
            sentence_embeddings = get_embeddings(sentences=sentences).to("cpu").numpy()
        except Exception as e:
            print(e)
            for _ in range(4):
                sentence_embeddings = np.zeros(shape=(5, 384))

        embeddings.append(sentence_embeddings)
        # filename_embeddings_dict[filename] = sentence_embeddings

        if False:
            similarity_score = get_similarity_scores(sentence_embeddings)
            filename_similarity_dict[filename] = similarity_score

    embeddings = np.vstack(embeddings)

    db = DBSCAN(min_samples=3, n_jobs=-1)
    db.fit(embeddings)
    print(db.labels_)
    labels = db.fit_predict(embeddings)

    print(sum(labels == -1))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)

    exit()
    basename = filename.split("/")[-1].split(".")[0]
    np.savez(
        "similarity_scores/" + basename + "_scores.npz", **filename_similarity_dict
    )


def print_similarity_scores(filename):
    if not os.path.exists(filename):
        print("File doesnot exist.......")
        exit()

    similarity_scores = np.load(filename)
    for i in list(similarity_scores.keys())[:5]:
        print("\n{} \n {}".format(i, similarity_scores[i]))


# Gets the embeddings of the sentences in the
@torch.no_grad
def get_embeddings(sentences: List[str]) -> torch.Tensor:
    encoded = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    ).to("cuda")
    model_output = model(**encoded)
    sentences_embed = mean_pooling(model_output, encoded["attention_mask"])
    sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
    return sentences_embed


# Generates the similarity scores from imput embeddings
def get_similarity_scores(embeds: torch.tensor):
    """
    embeds: (num_sentences, embed_dim)
    """
    assert embeds.ndim == 2

    cosine_similarity_matrix = np.empty(
        (embeds.shape[0], embeds.shape[0]), dtype=np.float64
    )

    for idx1 in range(embeds.shape[0]):
        for idx2 in range(embeds.shape[0]):
            cosine_similarity_matrix[idx1][idx2] = F.cosine_similarity(
                embeds[idx1], embeds[idx2], dim=0
            )

    return cosine_similarity_matrix


if __name__ == "__main__":
    # File paths
    DEV_FILE = "/home/akhil/models/DCASE24/dcase2024-task6-baseline/data/CLOTHO_v2.1/clotho_csv_files/clotho_captions_development.csv"
    DEV_SCORES_FILE = "/home/akhil/models/DCASE24/dcase2024-task6-baseline/notebooks/similarity_scores/clotho_captions_development.npz"

    from mixture_of_experts import MoE

    a = torch.rand()
    moe = MoE(512)
    print(moe(a))
    pass
