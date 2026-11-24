"""Released UnderGS event ranking and an explicitly separate causal adaptation."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .temporal import ranking_mrr, ranking_tie_diagnostics


TRACKS = ("official_compatible", "causal_adaptation")
FROZEN_DISPLAY_NAME = "Incremental Predictive Analytics over Evolving Graphs"


@torch.no_grad()
def frozen_query_features(scorer, history, pairs, timestamp):
    """Query shared visible topology; neither labels nor query roles are inputs."""
    tensor = torch.from_numpy(pairs)
    features = torch.cat(
        [scorer.features(block) for block in tensor.split(1024, dim=1)]
    )
    return features, history.statistics(pairs, timestamp)


def score_metrics(auc_positive, auc_negative, rank_positive, rank_negative):
    """Snapshot metrics, retaining the released NumPy tie rule as its own field."""
    pos = np.asarray(auc_positive).reshape(-1)
    neg = np.asarray(auc_negative).reshape(-1)
    rank_pos = np.asarray(rank_positive).reshape(-1)
    rank_neg = np.asarray(rank_negative).reshape(-1)
    if rank_neg.size != 100 * rank_pos.size:
        raise ValueError("UnderGS ranking requires 100 negatives per positive event")
    truth = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    values = dict(
        auc=float(roc_auc_score(truth, np.r_[pos, neg])),
        ap=float(average_precision_score(truth, np.r_[pos, neg])),
        mrr=ranking_mrr("undergs", rank_pos, rank_neg, None, None),
    )
    values.update(ranking_tie_diagnostics("undergs", rank_pos, rank_neg, None, None))
    return values


def runtime_state(model):
    return dict(
        node_features=model.node_features.detach().clone(),
        embedding_node_features=model.embedding_module.node_features.detach().clone(),
        tcns=model.embedding_module.tcns.backup(),
    )


def restore_runtime(model, state):
    model.node_features.copy_(state["node_features"])
    model.embedding_module.node_features.copy_(state["embedding_node_features"])
    model.embedding_module.tcns.restore(state["tcns"])


class CausalNeighborhood:
    """Persist released Container updates, but expose them only after observation.

    This is an adaptation, not a repair silently applied to the released path.
    Querying a node reads the same stored past neighbors regardless of whether
    that node occurs in a positive or negative candidate.
    """

    def __init__(self, model, util):
        self.model, self.util = model, util
        self.store = model.embedding_module.tcns
        self.original_update = self.store.update_tcns
        self.observing = False
        self.store.update_tcns = self.update

    def update(
        self,
        source_nodes,
        update_node_list,
        current_timestamp,
        edge_idxs,
        n_samples,
        unique_edge_indices,
    ):
        if self.observing:
            container = self.util.Container(
                self.store.num_nodes,
                self.store.k,
                self.store.lambda_1,
                self.store.lambda_2,
            )
            container.load_tcns(
                self.store.tcns_keys.cpu().numpy(),
                self.store.tcns_weights.cpu().numpy(),
            )
            result = container.maintain(
                source_nodes,
                update_node_list,
                current_timestamp,
                edge_idxs,
                n_samples,
                unique_edge_indices,
            )
            # The release returns temporary neighborhoods without this writeback.
            self.store.tcns_keys[update_node_list] = -1
            self.store.tcns_weights[update_node_list] = 0
            for node in update_node_list:
                for slot, (key, weight) in enumerate(
                    container.container_list[node].items()
                ):
                    self.store.tcns_keys[node, slot] = torch.tensor(key)
                    self.store.tcns_weights[node, slot] = weight
            return tuple(
                torch.from_numpy(value)
                .to(self.store.device)
                .to(torch.long if i < 2 else torch.float32)
                for i, value in enumerate(result)
            )
        keys = self.store.tcns_keys[source_nodes]
        valid = keys[:, :, 0] >= 0
        nodes = keys[:, :, 1].clamp_min(0)
        edges = keys[:, :, 0].clamp_min(0)
        delta = torch.where(valid, current_timestamp - keys[:, :, 2], 0).float()
        weights = self.store.tcns_weights[source_nodes].clone()
        return nodes, edges, delta, weights

    def score(self, sources, destinations, negatives, times, edge_idxs, train=False):
        state = runtime_state(self.model)
        result = self.model.compute_edge_probabilities(
            sources, destinations, negatives, times, edge_idxs, train=train
        )
        restore_runtime(self.model, state)
        return result

    @torch.no_grad()
    def observe(self, sources, destinations, times, edge_idxs):
        self.observing = True
        try:
            self.model.compute_edge_probabilities(
                sources, destinations, destinations, times, edge_idxs, train=False
            )
        finally:
            self.observing = False
