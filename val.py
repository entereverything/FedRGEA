import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_bacc(model, dataloader, get_confusion_matrix, args):
    all_preds = []
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for (x, label) in dataloader:
            if isinstance(x, list):
                x = x[0]
            x = x.cuda()
            _, logits = model(x)
            pred = torch.argmax(logits, dim=1)
            prob = torch.softmax(logits, dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(label)
            all_probs.append(prob.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    bacc = balanced_accuracy_score(all_labels, all_preds)
    acc_plain = accuracy_score(all_labels, all_preds)
    try:
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    except TypeError:
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average="macro",
                labels=list(range(all_probs.shape[1])),
            )
    except ValueError:
        auc = float("nan")

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(all_labels, all_preds)
        return bacc, acc_plain, macro_f1, auc, conf_matrix
    return bacc


def compute_loss(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0.
    with torch.no_grad():
        for (x, label) in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.cuda(), label.cuda()
            _, logits = model(x)
            loss += criterion(logits, label)
    return loss


def compute_loss_of_classes(model, dataloader, n_classes):
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    loss_class = torch.zeros(n_classes).float()
    loss_list = []
    label_list = []

    with torch.no_grad():
        for (x, label) in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.cuda(), label.cuda()
            _, logits = model(x)
            loss = criterion(logits, label)
            loss_list.append(loss)
            label_list.append(label)

    loss_list = torch.cat(loss_list).cpu()
    label_list = torch.cat(label_list).cpu()

    for i in range(n_classes):
        idx = torch.where(label_list==i)[0]
        loss_class[i] = loss_list[idx].sum()

    return loss_class


def compute_global_prototypes_from_loader(model, dataloader, n_classes):
    """
    Server-side: class-mean projector features on a labeled loader (e.g. val set).
    Returns [n_classes, feat_dim] tensor on CUDA (no grad).
    """
    if not hasattr(model, "projector"):
        raise ValueError("model needs .projector for global prototypes")
    feat_dim = int(model.projector[-1].out_features)
    device = next(model.parameters()).device
    sums = torch.zeros(n_classes, feat_dim, device=device, dtype=torch.float32)
    counts = torch.zeros(n_classes, device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            if isinstance(x, list):
                x = x[0]
            x = x.to(device)
            y = y.to(device)
            feats, _ = model(x, project=True)
            for c in range(n_classes):
                mask = y == c
                if mask.any():
                    sums[c] = sums[c] + feats[mask].sum(dim=0)
                    counts[c] = counts[c] + float(mask.sum().item())
    protos = torch.zeros(n_classes, feat_dim, device=device, dtype=torch.float32)
    for c in range(n_classes):
        if counts[c] > 0:
            protos[c] = sums[c] / counts[c]
        else:
            protos[c] = torch.randn(feat_dim, device=device)
    return protos.detach()
