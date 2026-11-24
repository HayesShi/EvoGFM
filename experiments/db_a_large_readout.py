"""Bound the published histogram readout's working input on full large graphs."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state

from experiments.db_a_quality import features


class FeatureArchive:
    def __init__(self, directory):
        directory.mkdir(exist_ok=True)
        self.directory = directory
        self.paths = defaultdict(list)
        self.rows = defaultdict(int)

    def append(self, phase, values):
        position = self.rows[phase]
        path = self.directory / f"{phase}_{position:09d}.npz"
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, values=values)
        temp.replace(path)
        self.paths[phase].append(path)
        self.rows[phase] += len(values)

    def blocks(self, phase, half_life):
        offset = 0
        for path in self.paths[phase]:
            with np.load(path) as part:
                values = part["values"]
            for start in range(0, len(values), 65_536):
                block = values[start : start + 65_536]
                yield offset, features(
                    block[:, :17], block[:, 17:], "union24", half_life
                )
                offset += len(block)

    def validation_auc(self, classifier, half_life):
        size = self.rows["validation"]
        scores = np.empty(size, dtype=np.float64)
        for start, block in self.blocks("validation", half_life):
            scores[start : start + len(block)] = classifier.predict_proba(block)[:, 1]
        return roc_auc_score(alternating_labels(size), scores)


def alternating_labels(size):
    positive = np.zeros(size, dtype=bool)
    positive[::2] = True
    return positive


def fit_batched(classifier, rows, blocks):
    """Stream the estimator's ordinary quantile fit and preserve its histogram learner."""
    if classifier.early_stopping or classifier.warm_start:
        raise ValueError(
            "Batched histogram fitting requires a fresh fit without internal validation"
        )

    def bin_data(input_shape, is_training_data):
        if not is_training_data:
            raise ValueError("The locked readout uses a separate validation split")
        mapper = classifier._bin_mapper
        shape = input_shape.shape
        if mapper.subsample is not None and rows > mapper.subsample:
            rng = check_random_state(mapper.random_state)
            subset = rng.choice(rows, mapper.subsample, replace=False)
        else:
            subset = np.arange(rows)
        order = np.argsort(subset)
        sorted_subset = subset[order]
        sample = np.empty((len(subset), shape[1]), dtype=np.float64)
        for start, block in blocks():
            first, last = np.searchsorted(sorted_subset, [start, start + len(block)])
            sample[order[first:last]] = block[sorted_subset[first:last] - start]
        mapper.fit(sample)
        binned = np.empty(shape, dtype=np.uint8, order="F")
        for start, block in blocks():
            binned[start : start + len(block)] = mapper.transform(block)
        return binned

    # The estimator uses this view only to validate shape and input type. Its
    # numeric trainer consumes the binned observations returned above.
    shape_only = np.broadcast_to(np.float64(0), (rows, 24))
    classifier._bin_data = bin_data
    try:
        classifier.fit(shape_only, alternating_labels(rows))
    finally:
        del classifier._bin_data
    return classifier
