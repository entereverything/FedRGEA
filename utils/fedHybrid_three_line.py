import copy
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

StateDict = Dict[str, torch.Tensor]


def _normalize_position(x: torch.Tensor) -> torch.Tensor:
    """Project to simplex: x >= 0 and sum(x)=1."""
    x = torch.clamp(x, min=0.0)
    s = x.sum()
    if s.item() <= 0:
        return torch.ones_like(x) / float(x.numel())
    return x / s


def _float_keys(w_global: StateDict) -> List[str]:
    return [k for k, v in w_global.items() if torch.is_floating_point(v)]


def _stack_client_deltas(
    w_locals: Sequence[StateDict],
    w_global: StateDict,
    clients: Sequence[int],
    key: str,
) -> torch.Tensor:
    values = torch.stack(
        [w_locals[cid][key].detach() for cid in clients],
        dim=0,
    )
    deltas = values - w_global[key].detach().unsqueeze(0)
    return deltas.to(torch.float32)


def _median_delta_reference(
    w_locals: Sequence[StateDict],
    w_global: StateDict,
    clients: Sequence[int],
    float_keys: Sequence[str],
) -> StateDict:
    reference: StateDict = {}
    for k in float_keys:
        deltas = _stack_client_deltas(w_locals, w_global, clients, k)
        reference[k] = deltas.median(dim=0).values
    return reference


def _robust_normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores

    center = scores.median()
    mad = (scores - center).abs().median()
    scale = torch.clamp(1.4826 * mad, min=1e-12)
    normalized = torch.clamp((scores - center) / scale, min=0.0)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _reference_conflict_scores(
    w_locals: Sequence[StateDict],
    w_global: StateDict,
    clients: Sequence[int],
    reference_delta: StateDict,
    float_keys: Sequence[str],
) -> torch.Tensor:
    """
    c_i = ||Delta_i - Delta_ref||^2 / P.

    This is the fast CSEA surrogate: it avoids building the O(m^2) pairwise
    client matrix while still measuring update-direction conflict.
    """
    dev = next(iter(w_global.values())).device
    scores = torch.zeros(len(clients), dtype=torch.float32, device=dev)
    n_params = 0

    for k in float_keys:
        deltas = _stack_client_deltas(w_locals, w_global, clients, k)
        diff = deltas - reference_delta[k].unsqueeze(0)
        scores += torch.sum(diff.reshape(len(clients), -1) ** 2, dim=1)
        n_params += reference_delta[k].numel()

    return _robust_normalize_scores(scores / float(max(1, n_params)))


def _build_prior(
    clients: Sequence[int],
    dev: torch.device,
    dict_len: Optional[Sequence[int]] = None,
    client_reliability: Optional[Dict[int, float]] = None,
    prior_mode: str = "reliability",
) -> torch.Tensor:
    prior_mode = prior_mode.lower()
    values: List[float] = []

    if prior_mode == "sample_size" and dict_len is not None:
        values = [float(max(0, dict_len[cid])) for cid in clients]
    elif prior_mode == "reliability" and client_reliability is not None:
        values = [float(max(0.0, client_reliability.get(cid, 0.0))) for cid in clients]

    if len(values) != len(clients) or sum(values) <= 0.0:
        values = [1.0 for _ in clients]

    return _normalize_position(torch.tensor(values, dtype=torch.float32, device=dev))


def _fitness(
    x: torch.Tensor,
    conflict_scores: torch.Tensor,
    prior: torch.Tensor,
    prior_strength: float,
    diversity_strength: float,
) -> float:
    conflict_cost = torch.dot(x, conflict_scores)
    prior_cost = torch.sum((x - prior) * (x - prior))
    concentration_cost = torch.sum(x * x)
    objective = conflict_cost + prior_strength * prior_cost + diversity_strength * concentration_cost
    return -objective.item()


def _split_three_lines(
    pop: List[torch.Tensor],
    fit: List[float],
) -> Tuple[List[int], List[int], List[int]]:
    n = len(pop)
    order = sorted(range(n), key=lambda i: fit[i], reverse=True)

    m_num = max(1, n // 3)
    s_num = max(1, n // 3)
    r_num = max(1, n - m_num - s_num)

    while m_num + r_num + s_num > n:
        r_num = max(1, r_num - 1)
    while m_num + r_num + s_num < n:
        r_num += 1

    M = order[:m_num]
    R = order[m_num:m_num + r_num]
    S = order[m_num + r_num:]
    return M, R, S


def _initial_population(
    d: int,
    dev: torch.device,
    pop_size: int,
    prior: torch.Tensor,
    conflict_scores: torch.Tensor,
) -> List[torch.Tensor]:
    pop: List[torch.Tensor] = []
    uniform = torch.ones(d, dtype=torch.float32, device=dev) / float(d)
    conflict_seed = _normalize_position(prior * torch.exp(-torch.clamp(conflict_scores, max=20.0)))

    for seed in (prior, conflict_seed, uniform):
        if len(pop) >= pop_size:
            break
        pop.append(seed.clone())

    for _ in range(max(0, pop_size - len(pop))):
        x = 0.8 * torch.rand(d, dtype=torch.float32, device=dev) + 0.2 * prior
        pop.append(_normalize_position(x))

    return pop


def fedHybridThreeLine(
    w_locals: Sequence[StateDict],
    dict_len: Optional[Sequence[int]],
    w_global: StateDict,
    normal_clients: Sequence[int],
    pop_size: int = 8,
    generations: int = 2,
    reset_patience: int = 2,
    client_reliability: Optional[Dict[int, float]] = None,
    prior_mode: str = "reliability",
    prior_strength: float = 0.15,
    diversity_strength: float = 0.02,
) -> StateDict:
    if len(normal_clients) == 0:
        return copy.deepcopy(w_global)

    dev = next(iter(w_global.values())).device
    parents = list(normal_clients)
    d = len(parents)
    pop_size = max(3, int(pop_size))
    generations = max(0, int(generations))
    prior_strength = float(max(0.0, prior_strength))
    diversity_strength = float(max(0.0, diversity_strength))

    float_keys = _float_keys(w_global)
    if len(float_keys) == 0:
        return copy.deepcopy(w_global)

    reference_delta = _median_delta_reference(w_locals, w_global, parents, float_keys)
    conflict_scores = _reference_conflict_scores(
        w_locals,
        w_global,
        parents,
        reference_delta,
        float_keys,
    )
    # The evolutionary search only operates on short client-weight vectors.
    # Keep it on CPU to avoid repeated GPU synchronization from Python control
    # flow and scalar fitness comparisons.
    search_device = torch.device("cpu")
    search_scores = conflict_scores.detach().to(search_device)
    prior = _build_prior(
        parents,
        search_device,
        dict_len=dict_len,
        client_reliability=client_reliability,
        prior_mode=prior_mode,
    )

    pop = _initial_population(d, search_device, pop_size, prior, search_scores)
    fit = [
        _fitness(x, search_scores, prior, prior_strength, diversity_strength)
        for x in pop
    ]
    stagnation = [0 for _ in pop]

    for _ in range(generations):
        M, R, S = _split_three_lines(pop, fit)
        best_idx = max(range(len(pop)), key=lambda i: fit[i])
        best_x = pop[best_idx]

        # Hybridization for S line.
        for s_idx in S:
            mr_idx = random.choice(M)
            x_sr = pop[s_idx]
            x_mr = pop[mr_idx]
            r1 = torch.rand(1, device=search_device).item()
            r2 = torch.rand(1, device=search_device).item()
            x_new = (r1 * x_sr + r2 * x_mr) / max(1e-12, (r1 + r2))
            x_new = _normalize_position(x_new)

            f_new = _fitness(
                x_new,
                search_scores,
                prior,
                prior_strength,
                diversity_strength,
            )
            if f_new > fit[s_idx]:
                pop[s_idx] = x_new
                fit[s_idx] = f_new

        # Selfing for R line.
        for r_idx in R:
            if len(R) > 1:
                other = random.choice([idx for idx in R if idx != r_idx])
            else:
                other = r_idx

            x_i = pop[r_idx]
            x_j = pop[other]
            a = torch.rand(1, device=search_device).item()
            b = torch.rand(1, device=search_device).item()

            x_new = x_i + a * (x_j - x_i) + b * (best_x - x_i)
            x_new = _normalize_position(x_new)

            f_new = _fitness(
                x_new,
                search_scores,
                prior,
                prior_strength,
                diversity_strength,
            )
            if f_new > fit[r_idx]:
                pop[r_idx] = x_new
                fit[r_idx] = f_new
                stagnation[r_idx] = 0
            else:
                stagnation[r_idx] += 1

        # Reset stale R individuals.
        for r_idx in R:
            if stagnation[r_idx] >= reset_patience:
                x_reset = (
                    0.8 * torch.rand(d, dtype=torch.float32, device=search_device)
                    + 0.2 * prior
                )
                x_reset = _normalize_position(x_reset)
                f_reset = _fitness(
                    x_reset,
                    search_scores,
                    prior,
                    prior_strength,
                    diversity_strength,
                )
                pop[r_idx] = x_reset
                fit[r_idx] = f_reset
                stagnation[r_idx] = 0

    best_idx = max(range(len(pop)), key=lambda i: fit[i])
    best_w = pop[best_idx].to(dev)

    w_new = copy.deepcopy(w_global)
    for k in w_new.keys():
        base = w_global[k]
        if not torch.is_floating_point(base):
            w_new[k] = w_locals[parents[0]][k]
            continue

        deltas = _stack_client_deltas(w_locals, w_global, parents, k)
        acc = torch.tensordot(best_w, deltas, dims=([0], [0]))
        w_new[k] = (w_global[k].detach().to(torch.float32) + acc).to(base.dtype)

    return w_new
