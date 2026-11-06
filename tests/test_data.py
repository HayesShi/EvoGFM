import numpy as np
import torch

from evogfm.data import load_graph


def test_wisconsin_cache_gains_public_masks(tmp_path):
    raw = tmp_path / "raw" / "wisconsin"
    raw.mkdir(parents=True)
    (raw / "out1_node_feature_label.txt").write_text(
        "node_id\tfeature\tlabel\n0\t1,0\t0\n1\t0,1\t1\n2\t1,1\t0\n"
    )
    (raw / "out1_graph_edges.txt").write_text("node_id\tnode_id\n0\t1\n1\t2\n")
    masks = dict(
        train_mask=np.array([True, False, False]),
        val_mask=np.array([False, True, False]),
        test_mask=np.array([False, False, True]),
    )
    np.savez(raw / "split_0.npz", **masks)
    cache = tmp_path / "graphs" / "wisconsin.pt"
    cache.parent.mkdir()
    torch.save(dict(x=torch.zeros(3, 2)), cache)

    graph = load_graph("wisconsin", tmp_path)
    for index, part in enumerate(("train", "val", "test")):
        assert graph[part].tolist() == [index]
    assert graph["edges"].tolist() == [[0, 1], [1, 2]]
    assert graph["y"].tolist() == [0, 1, 0]
    cached = torch.load(cache, weights_only=False)
    assert torch.equal(cached["train"], graph["train"])
