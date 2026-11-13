"""Load untouched public SNAP recipients; prepare full native NC/LP protocols."""

import json
import zipfile
import gc
from . import cross_task as cs
import pandas as pd
from evogfm.data import canonical_edges, make_pack
from evogfm.features import preprocess, pair_context, pair_channels, contexts
from evogfm.benchmarks import graphany_masks

torch, np = cs.torch, cs.np


def load_raw(name):
    if name.startswith("twitch-"):
        region = {"twitch-en": "ENGB", "twitch-de": "DE", "twitch-fr": "FR"}[name]
        z = zipfile.ZipFile(cs.ROOT / "raw/twitch.zip")
        directory = f"twitch/{region}/"
        prefix = f"musae_{region}"
        feature_file = directory + (
            prefix + ".json" if region == "DE" else prefix + "_features.json"
        )
        features = json.loads(z.read(feature_file))
        frame = pd.read_csv(z.open(directory + prefix + "_target.csv"))
        # The FR archive repeats two node IDs; their mature labels agree.
        assert int(frame.groupby("new_id").mature.nunique().max()) == 1
        original_rows = len(frame)
        frame = frame.drop_duplicates("new_id", keep="first")
        if len(frame) != original_rows:
            cs.event(
                "targets_duplicate_labels",
                graph=name,
                original_rows=original_rows,
                unique_nodes=len(frame),
                conflicting_labels=0,
            )
        # Target rows are not in node-ID order in the public archive.
        ids = frame.new_id.to_numpy()
        labels = frame.mature.to_numpy().astype(np.int64)
        edges = pd.read_csv(z.open(directory + prefix + "_edges.csv")).to_numpy().T
        source = "https://snap.stanford.edu/data/twitch.zip"
    else:
        z = zipfile.ZipFile(cs.ROOT / "raw/git_web_ml.zip")
        features = json.loads(z.read("git_web_ml/musae_git_features.json"))
        frame = pd.read_csv(z.open("git_web_ml/musae_git_target.csv"))
        ids = frame.id.to_numpy()
        labels = frame.ml_target.to_numpy()
        edges = pd.read_csv(z.open("git_web_ml/musae_git_edges.csv")).to_numpy().T
        source = "https://snap.stanford.edu/data/git_web_ml.zip"
    n = len(frame)
    dim = max(max(v, default=-1) for v in features.values()) + 1
    assert set(ids.tolist()) == set(range(n))
    assert set(map(int, features)) == set(range(n))
    x = torch.zeros(n, dim)
    y = torch.empty(n, dtype=torch.long)
    y[torch.from_numpy(ids)] = torch.from_numpy(labels)
    for key, indices in features.items():
        x[int(key), indices] = 1.0
    graph = graphany_masks(dict(x=x, y=y, edges=canonical_edges(edges, n)))
    return graph, source


def prepare(name, model):
    path = cs.ROOT / "cache" / f"{name}.pt"
    if path.exists():
        return
    cs.event("targets_prepare", graph=name)
    graph, source = load_raw(name)
    raw, x = preprocess(graph)
    native = make_pack(name, x, graph["edges"], seed=42, query_cap=10**9)
    visible = native["state_edges"]
    with torch.no_grad():
        state = model.state(x, visible, 0.6)
    hard = cs.safe_hard_pool(visible, graph["edges"], len(raw))
    uniforms = torch.cat(
        [
            native[k]["pairs"][:, native[k]["y"] == 0]
            for k in ["calibration", "lp_val", "lp_test"]
        ],
        dim=1,
    )
    keys = torch.sort(uniforms[0] * len(raw) + uniforms[1]).values
    hardkeys = hard[0] * len(raw) + hard[1]
    loc = torch.searchsorted(keys, hardkeys)
    hard = hard[
        :, (loc == len(keys)) | (keys[loc.clamp_max(len(keys) - 1)] != hardkeys)
    ]
    positive = native["lp_test"]["pairs"][:, native["lp_test"]["y"] == 1]
    count = positive.shape[1]
    if hard.shape[1] < count:
        raise RuntimeError(f"Insufficient hard negatives: {name}")
    hard = hard[
        :,
        torch.randperm(hard.shape[1], generator=torch.Generator().manual_seed(42))[
            :count
        ],
    ]
    p = dict(
        name=name,
        raw=raw,
        x=x,
        y=graph["y"],
        edges=visible,
        state=state,
        native=native,
        protocol={k: graph[k] for k in ["train", "val", "test"]},
        hard_test=hard,
        source=source,
        full_edge_count=graph["edges"].shape[1],
        feature_dimension=raw.shape[1],
    )
    cs.rs.save_pt(path, p)
    cs.event(
        "targets_saved",
        graph=name,
        nodes=len(raw),
        edges=p["full_edge_count"],
        features=raw.shape[1],
        test_edges=count,
    )
    del graph, p, raw, x, native
    gc.collect()


def tables(name):
    path = cs.ROOT / "tables_v2" / f"eval_{name}.pt"
    if path.exists():
        return
    p = torch.load(cs.ROOT / "cache" / f"{name}.pt", weights_only=False, mmap=True)
    native = p["native"]
    npos = int((native["lp_test"]["y"] == 1).sum())
    pos = native["lp_test"]["pairs"][:, native["lp_test"]["y"] == 1]
    uni = native["lp_test"]["pairs"][:, native["lp_test"]["y"] == 0]
    pairs = torch.cat(
        [
            native["calibration"]["pairs"],
            native["lp_val"]["pairs"],
            pos,
            uni,
            p["hard_test"],
        ],
        dim=1,
    )
    prepared = pair_context(p["raw"], p["edges"])
    records = []
    visible = None
    degree = None
    for start in range(0, pairs.shape[1], 512):
        batch = pairs[:, start : start + 512]
        ctx = contexts(p["edges"], len(p["raw"]), batch)
        if visible is None:
            visible, degree = ctx["visible"], ctx["degree"]
        else:
            ctx["visible"], ctx["degree"] = visible, degree
        records.append(
            dict(
                pairs=batch,
                ctx=ctx,
                base=pair_channels(p["raw"], p["edges"], batch, prepared=prepared),
            )
        )
        if start % 32768 == 0:
            cs.event("targets_tables", graph=name, done=start, total=pairs.shape[1])
    cs.rs.save_pt(
        path,
        dict(
            batches=records,
            nfit=len(native["calibration"]["y"]),
            nval=len(native["lp_val"]["y"]),
            npos=npos,
        ),
    )
    del p, prepared, records
    gc.collect()


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    cs.setup()
    cs.manifest()
    from urllib.request import urlretrieve

    for name in ["twitch.zip", "git_web_ml.zip"]:
        target = cs.ROOT / "raw" / name
        if not target.exists():
            temporary = target.with_suffix(".download")
            urlretrieve("https://snap.stanford.edu/data/" + name, temporary)
            temporary.replace(target)
    model, _, _ = cs.rs.load_checkpoint(cs.settings.CHECKPOINT)
    for name in cs.TARGETS:
        prepare(name, model)
        tables(name)
    cs.event("targets_complete", graphs=cs.TARGETS)


if __name__ == "__main__":
    main()
