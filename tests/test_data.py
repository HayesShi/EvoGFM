import numpy as np
import torch

from evogfm.data import load_graph, load_tu


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


def test_featureless_tu_uses_undirected_degree_without_graph_labels(tmp_path):
    raw = tmp_path / "raw" / "imdb-binary" / "IMDB-BINARY"
    raw.mkdir(parents=True)
    (raw / "IMDB-BINARY_A.txt").write_text("1,2\n2,1\n2,3\n3,2\n4,5\n5,4\n")
    (raw / "IMDB-BINARY_graph_indicator.txt").write_text("1\n1\n1\n2\n2\n")
    (raw / "IMDB-BINARY_graph_labels.txt").write_text("0\n1\n")
    graphs, _ = load_tu("IMDB-BINARY", tmp_path)
    assert graphs[0]["x"].argmax(1).tolist() == [1, 2, 1]
    assert graphs[1]["x"].argmax(1).tolist() == [1, 1]
    assert graphs[0]["x"].shape[1] == 64
    (raw / "IMDB-BINARY_graph_labels.txt").write_text("1\n0\n")
    changed, _ = load_tu("IMDB-BINARY", tmp_path)
    assert all(torch.equal(a["x"], b["x"]) for a, b in zip(graphs, changed))
