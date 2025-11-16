"""
KNN-based consistency stuff for the structural/hybrid modes.
Nothing too fancy — just nearest neighbors + simple smoothing loss.
"""

import torch
import torch.nn.functional as F


# ------------------------------------------------
# KNN: find K neighbors based on cosine sim
# ------------------------------------------------
def knn_neighbors(features, K=5):
    # normalize so cosine sim becomes dot product
    x = F.normalize(features, dim=1)

    # sim matrix (B x B)
    sim = x @ x.t()

    # remove self-sim (otherwise each sample picks itself)
    sim.fill_diagonal_(-1e9)

    # top-K indices
    _, idx = sim.topk(K, dim=1)
    return idx


# ------------------------------------------------
# Consistency loss: force logits to be close to
# the mean logits of neighbors.
# ------------------------------------------------
def neighbor_consistency_loss(logits, neighbor_idx):
    B, C = logits.shape
    K = neighbor_idx.size(1)

    # gather neighbor logits: (B, K, C)
    n_logits = logits[neighbor_idx]

    # average across neighbors → (B, C)
    n_mean = n_logits.mean(dim=1)

    # simple L2 distance between logits and neighbor-mean
    loss = F.mse_loss(logits, n_mean)
    return loss


# ------------------------------------------------
# Hybrid trick:
# down-weight samples whose prediction doesn't
# agree with neighbors AND low confidence.
# ------------------------------------------------
def compute_agreement_weights(logits, labels, neighbor_idx, confidence_threshold=0.6):
    with torch.no_grad():
        # softmax for confidence
        p = F.softmax(logits, dim=1)

        # predicted probs & labels
        conf, pred = p.max(dim=1)

        # neighbors' predicted labels
        neigh_pred = pred[neighbor_idx]      # (B, K)
        neigh_vote = neigh_pred.mode(dim=1).values  # (B,)

        # agreement mask
        agree = (neigh_vote == labels)

        # low-confidence flag
        low_conf = conf < confidence_threshold

        # if disagree + low conf → reduce weight
        # otherwise keep 1.0
        w = torch.ones_like(conf)
        bad = (~agree) & low_conf
        w[bad] = 0.5     # I just used 0.5, nothing magical

        return w