import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from classify_graphs import GraphScale, fit_balanced


def test_reused_fold_kernels_match_standard_pipeline_search():
    generator = np.random.default_rng(42)
    labels = np.repeat([0, 1], 16)
    features = generator.normal(size=(32, 6))
    features[:, 0] += labels
    features[:, 2] = 0  # Constant channels must also match SVC's gamma='scale'.
    actual, parameters = fit_balanced(features, labels, split=2)
    reference = GridSearchCV(
        make_pipeline(GraphScale(2), SVC()),
        {
            "graphscale__weight": [0.1, 1.0, 10.0],
            "svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ).fit(features, labels)
    assert parameters == reference.best_params_
    assert np.array_equal(actual.predict(features), reference.predict(features))
