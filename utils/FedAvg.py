import copy
from typing import Dict, List, Optional, Sequence

import torch

StateDict = Dict[str, torch.Tensor]


def FedAvg(
    w: Sequence[StateDict],
    dict_len: Sequence[int],
    normal_clients: Optional[Sequence[int]] = None,
) -> StateDict:

    n_clients = len(w)
    if normal_clients is None:
        indices: List[int] = list(range(n_clients))
    else:
        indices = list(normal_clients)
        if len(indices) == 0:
            return copy.deepcopy(w[0])

    ref_idx = indices[0]
    w_avg = copy.deepcopy(w[ref_idx])

    total = float(sum(dict_len[i] for i in indices))

    for k in w_avg.keys():
        base = w_avg[k]

        if not torch.is_floating_point(base) and not torch.is_complex(base):
            w_avg[k] = w[ref_idx][k]
            continue

        acc = torch.zeros_like(base, dtype=torch.float32)
        for i in indices:
            wi = w[i]
            ni = dict_len[i]
            t = wi[k]
            if not torch.is_floating_point(t) and not torch.is_complex(t):
                raise TypeError(f"FedAvg dtype mismatch at key={k}: got {t.dtype}")
            acc += t.detach().to(torch.float32) * float(ni)
        w_avg[k] = (acc / total).to(base.dtype)

    return w_avg