import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


StateDict = Dict[str, torch.Tensor]

DEFAULT_MAD_SCALE = 3.5
DEFAULT_MIN_CLIENTS_FOR_FILTERING = 3
DEFAULT_MAX_REJECT_RATIO = 0.5
DEFAULT_REFERENCE_TOP_RATIO = 0.5
DEFAULT_REFERENCE_WARMUP_ROUNDS = 1
DEFAULT_REFERENCE_BLEND = 0.7
DEFAULT_REFERENCE_TRIM_RATIO = 0.0
DEFAULT_REFERENCE_FILTER_SCALE = 0.0
DEFAULT_RELIABILITY_MOMENTUM = 0.6
DEFAULT_RELIABILITY_FLOOR = 1e-3


class ImmuneDetector:
    def __init__(
        self,
        threshold: float = 0.8,
        mad_scale: float = DEFAULT_MAD_SCALE,
        min_clients_for_filtering: int = DEFAULT_MIN_CLIENTS_FOR_FILTERING,
        max_reject_ratio: float = DEFAULT_MAX_REJECT_RATIO,
        reference_top_k: Optional[int] = None,
        reference_top_ratio: float = DEFAULT_REFERENCE_TOP_RATIO,
        reference_warmup_rounds: int = DEFAULT_REFERENCE_WARMUP_ROUNDS,
        reference_blend: float = DEFAULT_REFERENCE_BLEND,
        reliability_momentum: float = DEFAULT_RELIABILITY_MOMENTUM,
        reliability_floor: float = DEFAULT_RELIABILITY_FLOOR,
    ):
        # threshold controls tolerated abnormal ratio: allowed <= (1 - threshold).
        # It also works as rho when reliability-threshold reference selection is used.
        self.threshold = float(max(0.0, min(1.0, threshold)))
        self.mad_scale = float(mad_scale)
        self.min_clients_for_filtering = int(max(1, min_clients_for_filtering))
        self.max_reject_ratio = float(max(0.0, min(1.0, max_reject_ratio)))
        self.reference_top_k = reference_top_k
        self.reference_top_ratio = float(max(0.0, min(1.0, reference_top_ratio)))
        self.reference_warmup_rounds = int(max(0, reference_warmup_rounds))
        self.reference_blend = float(max(0.0, min(1.0, reference_blend)))
        self.reference_trim_ratio = DEFAULT_REFERENCE_TRIM_RATIO
        self.reference_filter_scale = DEFAULT_REFERENCE_FILTER_SCALE
        self.reliability_momentum = float(max(0.0, min(1.0, reliability_momentum)))
        self.reliability_floor = float(max(0.0, reliability_floor))

        # Persistent cross-round memory. r_i^{t-1} is read before scoring round t.
        # No client is assumed to be clean a priori.
        self.reliability_scores: Dict[int, float] = {}
        self.round_index = 0
        self.last_reference_clients: List[int] = []
        self.last_client_scores: Dict[int, float] = {}

    def calculate_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Euclidean distance between two flattened vectors."""
        a = a.reshape(-1).to(torch.float32)
        b = b.reshape(-1).to(torch.float32)
        return torch.norm(a - b, p=2).item()

    def calculate_kl(
        self,
        mu_q: torch.Tensor,
        var_q: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        mu_q = torch.as_tensor(mu_q, dtype=torch.float32)
        mu_p = torch.as_tensor(mu_p, dtype=torch.float32)
        var_q = torch.as_tensor(var_q, dtype=torch.float32).clamp(min=eps)
        var_p = torch.as_tensor(var_p, dtype=torch.float32).clamp(min=eps)

        diff = mu_q - mu_p
        term = torch.log(var_p / var_q) + (var_q + diff * diff) / var_p - 1.0
        return 0.5 * term.item()

    def calculate_cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity in [-1, 1]."""
        a_flat = a.reshape(-1).to(torch.float32)
        b_flat = b.reshape(-1).to(torch.float32)
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        if norm_a.item() == 0.0 and norm_b.item() == 0.0:
            return 1.0
        if norm_a.item() == 0.0 or norm_b.item() == 0.0:
            return 0.0
        return torch.dot(a_flat, b_flat).div(norm_a * norm_b).item()

    def _get_client_weight(self, client_id: int, default_weight: float = 1.0) -> float:
        return float(default_weight)

    def _float_keys(self, sd: StateDict) -> List[str]:
        return [k for k, v in sd.items() if torch.is_floating_point(v)]

    def _delta(
        self,
        client_params: Dict[int, StateDict],
        cid: int,
        key: str,
        previous_model: Optional[StateDict],
    ) -> torch.Tensor:
        base = client_params[cid][key]
        if previous_model is None or key not in previous_model:
            return base.detach().to(torch.float32)
        return (base - previous_model[key]).detach().to(torch.float32)

    def _robust_center_scale(self, values: Sequence[float]) -> Tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        x = torch.tensor(list(values), dtype=torch.float32)
        center = x.median().item()
        mad = (x - center).abs().median().item()
        scale = max(1.4826 * mad, 1e-12)
        return center, scale

    def _has_reliability_history(self, clients: Sequence[int]) -> bool:
        return any(cid in self.reliability_scores for cid in clients)

    def _rank_by_reliability(self, clients: Sequence[int]) -> List[int]:
        order = {cid: pos for pos, cid in enumerate(clients)}
        return sorted(
            clients,
            key=lambda cid: (
                self.reliability_scores.get(cid, 0.0),
                -order[cid],
            ),
            reverse=True,
        )

    def _select_reference_clients(
        self,
        present_clients: Sequence[int],
    ) -> List[int]:
        """
        From round 2 onward, choose C_ref from previous reliability only.

        Early rounds use the coordinate-wise median reference, which avoids
        requiring a manually clean client set.
        """
        if (
            self.round_index <= self.reference_warmup_rounds
            or not self._has_reliability_history(present_clients)
        ):
            return []

        ranked = self._rank_by_reliability(present_clients)
        min_ref = min(len(present_clients), self.min_clients_for_filtering)

        if self.reference_top_k is not None and self.reference_top_k > 0:
            k = min(len(present_clients), max(1, int(self.reference_top_k)))
            dynamic = ranked[:k]
        else:
            k = max(min_ref, int(math.ceil(len(present_clients) * self.reference_top_ratio)))
            k = min(len(present_clients), max(1, k))
            candidates = [
                cid
                for cid in ranked
                if self.reliability_scores.get(cid, 0.0) >= self.threshold
            ]
            dynamic = candidates[:k]
            if len(dynamic) < min_ref:
                dynamic = ranked[:min_ref]

        ref_clients: List[int] = []
        for cid in dynamic:
            if cid not in ref_clients:
                ref_clients.append(cid)
        return ref_clients

    def _median_reference(
        self,
        present_clients: Sequence[int],
        client_params: Dict[int, StateDict],
        float_keys: Sequence[str],
        previous_model: Optional[StateDict],
    ) -> StateDict:
        reference: StateDict = {}
        for k in float_keys:
            deltas = torch.stack(
                [self._delta(client_params, cid, k, previous_model) for cid in present_clients],
                dim=0,
            )
            reference[k] = deltas.median(dim=0).values
        return reference

    def _weighted_reference(
        self,
        ref_clients: Sequence[int],
        present_clients: Sequence[int],
        client_params: Dict[int, StateDict],
        float_keys: Sequence[str],
        previous_model: Optional[StateDict],
    ) -> StateDict:
        if len(ref_clients) == 0:
            return self._median_reference(
                present_clients,
                client_params,
                float_keys,
                previous_model,
            )

        first_cid = ref_clients[0]
        reference: StateDict = {
            k: torch.zeros_like(client_params[first_cid][k], dtype=torch.float32)
            for k in float_keys
        }
        total_weight = 0.0
        for cid in ref_clients:
            reliability = self.reliability_scores.get(cid, 1.0)
            w = max(self.reliability_floor, reliability) * self._get_client_weight(cid)
            for k in float_keys:
                reference[k] += self._delta(client_params, cid, k, previous_model) * w
            total_weight += w

        if total_weight <= 1e-12:
            return self._median_reference(
                present_clients,
                client_params,
                float_keys,
                previous_model,
            )

        for k in float_keys:
            reference[k] /= total_weight

        if self.reference_blend < 1.0:
            median_reference = self._median_reference(
                present_clients,
                client_params,
                float_keys,
                previous_model,
            )
            for k in float_keys:
                reference[k] = (
                    self.reference_blend * reference[k]
                    + (1.0 - self.reference_blend) * median_reference[k]
                )
        return reference

    def _update_reliability(
        self,
        present_clients: Sequence[int],
        anomaly_scores: Dict[int, float],
        normal_clients: Sequence[int],
    ) -> None:
        normal_set = set(normal_clients)

        for cid in present_clients:
            score = max(0.0, float(anomaly_scores.get(cid, 0.0)))
            quality = 1.0 / (1.0 + score)
            if cid not in normal_set:
                quality *= 0.25

            old = self.reliability_scores.get(cid)
            if old is None:
                new_value = quality
            else:
                m = self.reliability_momentum
                new_value = m * old + (1.0 - m) * quality
            self.reliability_scores[cid] = float(max(0.0, min(1.0, new_value)))

    def _cap_rejections(
        self,
        present_clients: Sequence[int],
        normal_clients: Sequence[int],
        anomaly_scores: Dict[int, float],
    ) -> List[int]:
        if len(present_clients) == 0:
            return []

        max_reject = int(math.floor(len(present_clients) * self.max_reject_ratio))
        if len(present_clients) > 0:
            max_reject = min(max_reject, len(present_clients) - 1)
        keep_count = len(present_clients) - max_reject
        normal_set = set(normal_clients)

        if len(normal_set) < keep_count:
            ranked = sorted(present_clients, key=lambda cid: anomaly_scores.get(cid, 0.0))
            normal_set.update(ranked[:keep_count])

        return [cid for cid in present_clients if cid in normal_set]

    def detect(
        self,
        client_params: Dict[int, StateDict],
        all_clients: List[int],
        previous_model: Optional[StateDict] = None,
    ) -> List[int]:
        self.round_index += 1

        present_clients = [cid for cid in all_clients if cid in client_params]
        if len(present_clients) == 0:
            self.last_reference_clients = []
            self.last_client_scores = {}
            return []

        first_cid = present_clients[0]
        float_keys = self._float_keys(client_params[first_cid])
        if len(float_keys) == 0:
            self.last_reference_clients = list(present_clients)
            self.last_client_scores = {cid: 0.0 for cid in present_clients}
            self._update_reliability(
                present_clients,
                self.last_client_scores,
                present_clients,
            )
            return list(present_clients)

        # Too few clients make robust filtering unstable; still refresh reliability gently.
        if len(present_clients) < self.min_clients_for_filtering:
            self.last_reference_clients = list(present_clients)
            self.last_client_scores = {cid: 0.0 for cid in present_clients}
            self._update_reliability(
                present_clients,
                self.last_client_scores,
                present_clients,
            )
            return list(present_clients)

        ref_clients = self._select_reference_clients(present_clients)
        reference = self._weighted_reference(
            ref_clients,
            present_clients,
            client_params,
            float_keys,
            previous_model,
        )
        self.last_reference_clients = list(ref_clients) if len(ref_clients) > 0 else list(present_clients)

        dist_values: Dict[str, Dict[int, float]] = {k: {} for k in float_keys}
        cos_loss_values: Dict[str, Dict[int, float]] = {k: {} for k in float_keys}
        kl_values: Dict[str, Dict[int, float]] = {k: {} for k in float_keys}

        for k in float_keys:
            ref_delta = reference[k]
            ref_flat = ref_delta.reshape(-1).to(torch.float32)
            if ref_flat.numel() == 0:
                ref_mu = torch.tensor(0.0, dtype=torch.float32)
                ref_var = torch.tensor(1.0, dtype=torch.float32)
            else:
                ref_mu = ref_flat.mean()
                ref_var = ref_flat.var(unbiased=False)

            for cid in present_clients:
                delta = self._delta(client_params, cid, k, previous_model)
                flat = delta.reshape(-1).to(torch.float32)

                dist_values[k][cid] = self.calculate_distance(delta, ref_delta)
                cos = self.calculate_cosine_similarity(delta, ref_delta)
                cos_loss_values[k][cid] = 1.0 - cos

                if flat.numel() == 0:
                    kl_values[k][cid] = 0.0
                else:
                    mu_q = flat.mean()
                    var_q = flat.var(unbiased=False)
                    kl_values[k][cid] = self.calculate_kl(mu_q, var_q, ref_mu, ref_var)

        metric_groups = (dist_values, cos_loss_values, kl_values)
        thresholds: List[Dict[str, float]] = []
        centers: List[Dict[str, float]] = []
        scales: List[Dict[str, float]] = []
        for group in metric_groups:
            group_thresholds: Dict[str, float] = {}
            group_centers: Dict[str, float] = {}
            group_scales: Dict[str, float] = {}
            for k in float_keys:
                center, scale = self._robust_center_scale(list(group[k].values()))
                group_centers[k] = center
                group_scales[k] = scale
                group_thresholds[k] = center + self.mad_scale * scale
            thresholds.append(group_thresholds)
            centers.append(group_centers)
            scales.append(group_scales)

        total_tests = max(1, len(float_keys) * len(metric_groups))
        allowed_abnormal = total_tests * max(0.0, 1.0 - self.threshold)
        normal_clients: List[int] = []
        anomaly_scores: Dict[int, float] = {}

        for cid in present_clients:
            abnormal_count = 0
            score = 0.0

            for group_idx, group in enumerate(metric_groups):
                for k in float_keys:
                    value = group[k][cid]
                    if value > thresholds[group_idx][k]:
                        abnormal_count += 1

                    center = centers[group_idx][k]
                    scale = scales[group_idx][k]
                    score += max(0.0, (value - center) / scale)

            score /= float(total_tests)
            anomaly_scores[cid] = score
            if abnormal_count <= allowed_abnormal:
                normal_clients.append(cid)

        normal_clients = self._cap_rejections(present_clients, normal_clients, anomaly_scores)
        self.last_client_scores = dict(anomaly_scores)
        self._update_reliability(
            present_clients,
            anomaly_scores,
            normal_clients,
        )
        return normal_clients
