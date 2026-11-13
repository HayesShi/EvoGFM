"""Original matched LP support/query sampling and disjointness rules."""

import numpy as np
import torch


def keys(p, n):
    p = p.numpy()
    return set((np.minimum(p[0], p[1]) * n + np.maximum(p[0], p[1])).tolist())


def build_protocol(pack, seed, k=200, context_mode="uniform"):
    native = pack["native"]
    n = len(pack["y"])
    vis = pack["edges"]
    full = native["full_edges"]
    cal = native["calibration"]
    pos = cal["pairs"][:, cal["y"] == 1]
    neg = cal["pairs"][:, cal["y"] == 0]
    if min(pos.shape[1], neg.shape[1]) < k:
        raise ValueError("Not enough calibration labels for requested k")
    gen = torch.Generator().manual_seed(seed)
    cp = pos[:, torch.randperm(pos.shape[1], generator=gen)[:k]]
    cn = neg[:, torch.randperm(neg.shape[1], generator=gen)[:k]]
    test = native["lp_test"]
    tp = test["pairs"][:, test["y"] == 1]
    un = test["pairs"][:, test["y"] == 0]
    # Original targets have hard_test. New stream graphs have disjoint hard_train/hard_val pools.
    if "hard_test" in pack:
        hard = pack["hard_test"]
    else:
        hard = pack["hard_val"]
        forbidden = keys(full, n)
        for split in ["calibration", "lp_val", "lp_test"]:
            forbidden |= keys(native[split]["pairs"][:, native[split]["y"] == 0], n)
        keep = [
            i
            for i, x in enumerate((hard[0] * n + hard[1]).tolist())
            if x not in forbidden
        ]
        hard = hard[:, keep]
        hg = torch.Generator().manual_seed(42)
        hard = hard[:, torch.randperm(hard.shape[1], generator=hg)[: tp.shape[1]]]
    support_origin = ["native.calibration positive"] * k + [
        "native.calibration uniform negative"
    ] * k
    if context_mode == "mixed":
        # Match the original sample_links calibration pool and RNG order, at 400 labels.
        assert k % 2 == 0
        hard_support = pack["hard_train"]
        forbidden_support = keys(full, n)
        for split in ["calibration", "lp_val", "lp_test"]:
            forbidden_support |= keys(
                native[split]["pairs"][:, native[split]["y"] == 0], n
            )
        keep = [
            i
            for i, x in enumerate((hard_support[0] * n + hard_support[1]).tolist())
            if x not in forbidden_support
        ]
        hard_support = hard_support[:, keep]
        assert hard_support.shape[1] >= k // 2
        hn = hard_support[
            :, torch.randperm(hard_support.shape[1], generator=gen)[: k // 2]
        ]
        assert not (keys(hn, n) & keys(pack["hard_val"], n))
        cn = torch.cat([cn[:, : k // 2], hn], 1)
        support_origin = (
            ["native.calibration positive"] * k
            + ["native.calibration uniform negative"] * (k // 2)
            + ["hard_train calibration-assigned pool"] * (k // 2)
        )
    assert hard.shape[1] == tp.shape[1]
    support = torch.cat([cp, cn], 1)
    query = torch.cat([tp, un, hard], 1)
    assert not (keys(support, n) & keys(query, n))
    assert not (keys(vis, n) & keys(tp, n))
    assert not (keys(cn, n) & keys(full, n))
    assert not (keys(hard, n) & keys(full, n))
    return dict(
        support_pairs=support,
        query_pairs=query,
        positive_count=tp.shape[1],
        context_labels=torch.cat([torch.ones(k), torch.zeros(k)]),
        support_origin=support_origin,
    )
