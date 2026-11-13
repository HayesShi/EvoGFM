"""Does donor supervision carry information beyond generic coefficient movement?

Shuffle only the donor's public training labels, preserving their histogram.
Use the exact four-step Study.adapt implementation and saved source curvature.
"""

from . import settings

from argparse import Namespace
import gc
from . import study as rs
from . import interventions as exp
from .io import atomic_json

torch, np = rs.torch, rs.np
HERE = settings.REPO
RUN = settings.NC_RUN
OUT = RUN / "internal_exploration"
CONTROL = OUT / "label_shuffle"
DONORS = ["dblp", "wisconsin", "questions"]


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(4)
    args = Namespace(
        output=str(CONTROL),
        threads=4,
        eval_device=settings.DEVICE,
        data="unused_cached_graphs",
    )
    study = rs.Study(args)
    atomic_json(
        OUT / "label_shuffle_manifest.json",
        dict(
            donors=DONORS,
            seeds=[42, 43, 44],
            recipients=rs.RECIPIENTS,
            shots=[1, 5, 20],
            update_steps=4,
            operation="Permute labels within each donor public training split; preserve label histogram, graph features, edges, original source checkpoint and seed-specific source curvature.",
            control_budget="Same class counts, training/held-out construction, episodes and GGN update count as the real-label donor experiment.",
            selection="No recipient scores select updates. Same donor training-only held-out selection as Study.adapt.",
            interpretation="Post hoc supervision-information control on three donor examples; not a replacement benchmark model",
        ),
    )
    for seed in [42, 43, 44]:
        prior = torch.load(RUN / "states" / f"s{seed}_initial.pt", weights_only=False)
        for name in DONORS:
            path = CONTROL / "states" / f"s{seed}_shuffle_{name}.pt"
            if path.exists():
                continue
            pack = torch.load(
                RUN / "cache" / f"{name}.pt", weights_only=False, mmap=True
            )
            modified = dict(pack)
            labels = pack["y"].clone()
            ids = pack["protocol"]["train"]
            generator = torch.Generator().manual_seed(
                seed * 10000 + rs.DONORS.index(name) * 100 + 73
            )
            permutation = torch.randperm(len(ids), generator=generator)
            labels[ids] = labels[ids[permutation]]
            modified["y"] = labels
            if not torch.equal(
                torch.bincount(labels[ids]), torch.bincount(pack["y"][ids])
            ):
                raise RuntimeError("Shuffled label counts changed")
            ons = study.new_online(prior)
            exp.event("shuffle_train", graph=name, seed=seed)
            info = study.adapt(modified, ons, seed, f"s{seed}_shuffle_{name}")
            state = ons.state_dict()
            state.update(
                seed=seed,
                graph=name,
                kind="label_shuffle",
                **info,
                unchanged_training_label_fraction=float(
                    labels[ids].eq(pack["y"][ids]).float().mean()
                ),
            )
            rs.save_pt(path, state)
            del pack, modified, labels
            gc.collect()
    for name in rs.RECIPIENTS:
        pack = torch.load(RUN / "cache" / f"{name}.pt", weights_only=False, mmap=True)
        study.model.inverse.set_theta(study.initial)
        h, _ = exp.make_views(study.model, pack)
        heads = exp.baseline_heads(h, pack, [42, 43, 44], [1, 5, 20])
        del h
        for seed in [42, 43, 44]:
            for donor in DONORS:
                path = (
                    OUT / "measurements" / f"{name}__label_shuffle-{donor}-{seed}.json"
                )
                if path.exists():
                    continue
                state = torch.load(
                    CONTROL / "states" / f"s{seed}_shuffle_{donor}.pt",
                    weights_only=False,
                )
                study.model.inverse.set_theta(torch.as_tensor(state["theta"]).float())
                spec = dict(
                    kind="label_shuffle", donor=donor, update_seed=seed, value=1.0
                )
                exp.event("shuffle_evaluate", graph=name, donor=donor, seed=seed)
                result = exp.evaluate(study.model, pack, heads, spec)
                atomic_json(path, result)
        exp.collect()
        del pack, heads
        gc.collect()
        torch.cuda.empty_cache()
    exp.event("complete", phase="label_shuffle")


if __name__ == "__main__":
    main()
