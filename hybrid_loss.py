import torch
import torch.nn.functional as F


def train_step_hybrid(
    model,
    optimizer,
    batch,
    feature_bank,
    pred_bank,
    criterion_cls,
    device,
    K=5,
    confidence_thresh=0.8,
    reduce_weight=0.2,
    lambda_nc=1.0,
    neighbor_temp=1.0,
):
    """
    One training step implementing Neighbor-Consistency and Agreement-Conditional Weighting.

    Returns a dict with losses and the features/logits for updating banks.
    """

    model.train()
    images, noisy_labels, indices = batch
    images = images.to(device)
    noisy_labels = noisy_labels.to(device)
    indices = indices.to(device)

    # Forward: model expected to return (logits, features) or (features, proj)
    out = model(images)
    if isinstance(out, tuple) and len(out) == 2:
        # Heuristic: if first tensor looks like features (shape B x D) and second like proj,
        # we'll try detect if any is logits by checking second dim size later.
        a, b = out
        # If b has same dim as classes is unknown; we expect the user test model to return (logits, features)
        # Here assume model returns (logits, features)
        logits, features = a, b
    else:
        # single output: assume logits
        logits = out
        # for features fallback, use logits as features
        features = logits

    logits = logits.to(device)
    features = features.to(device)
    B, C = logits.shape

    # Classification loss per-sample
    cls_losses = criterion_cls(logits, noisy_labels)

    probs = F.softmax(logits, dim=1)
    max_confidence, model_pred = torch.max(probs, dim=1)

    # Normalize features and bank for cosine similarity
    feat = F.normalize(features, dim=1)
    bank = F.normalize(feature_bank.to(device), dim=1)

    sims = torch.matmul(feat, bank.t())

    K_effective = min(K + 1, bank.size(0))
    topk_vals, topk_idx = sims.topk(K_effective, dim=1, largest=True, sorted=True)

    neighbors_idx_list = []
    for i in range(B):
        idxs = topk_idx[i].tolist()
        sample_index = int(indices[i].item())
        filtered = [x for x in idxs if x != sample_index]
        filtered = filtered[:K]
        neighbors_idx_list.append(filtered)

    pred_bank = pred_bank.to(device)
    bank_probs = F.softmax(pred_bank, dim=1)

    neighbors_avg_probs = torch.zeros_like(probs)
    actual_neighbor_counts = torch.zeros(B, device=device)

    eps = 1e-9
    for i in range(B):
        nidxs = neighbors_idx_list[i]
        if len(nidxs) == 0:
            neighbors_avg_probs[i] = probs[i].detach()
            actual_neighbor_counts[i] = 1
        else:
            neigh_probs = bank_probs[torch.tensor(nidxs, device=device)]
            neighbors_avg_probs[i] = neigh_probs.mean(dim=0)
            actual_neighbor_counts[i] = neigh_probs.shape[0]

    neighbors_avg_probs = neighbors_avg_probs.clamp(min=eps)
    neighbors_consensus = torch.argmax(neighbors_avg_probs, dim=1)

    low_conf_mask = max_confidence < confidence_thresh
    disagree_mask = neighbors_consensus != noisy_labels
    reduce_mask = low_conf_mask & disagree_mask

    weights = torch.ones_like(cls_losses, device=device)
    weights[reduce_mask] = reduce_weight

    weighted_cls_loss = (weights * cls_losses).mean()

    log_probs = F.log_softmax(logits, dim=1)
    per_sample_kl = F.kl_div(log_probs, neighbors_avg_probs, reduction='none').sum(dim=1)
    neighbor_consistency_loss = per_sample_kl.mean()

    total_loss = weighted_cls_loss + lambda_nc * neighbor_consistency_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "cls_loss": weighted_cls_loss.item(),
        "neighbor_loss": neighbor_consistency_loss.item(),
        "avg_confidence": max_confidence.mean().item(),
        "reduced_count": int(reduce_mask.sum().item()),
        "features": features.detach().cpu(),
        "logits": logits.detach().cpu(),
        "indices": indices.detach().cpu(),
    }
