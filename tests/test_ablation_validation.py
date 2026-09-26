"""Verify that an explicitly recorded logistic repair preserves ablation comparability."""

import json

import pytest

from plot_results import Experiment, validate_comparison


def _experiment(path, feature_set, metrics_hash):
    config = {
        'feature_set': feature_set, 'cache_sha256': 'cache',
        'sample_names': ['image'], 'seeds': [42], 'inner_cv': {'folds': 3},
        'parameter_grid': {'C': [1]}, 'standardization': 'train only',
        'svr_defaults': {'kernel': 'rbf'}, 'logistic': 'test post-hoc',
        'source_sha256': {'train_svr.py': 'same', 'metrics.py': metrics_hash},
    }
    return Experiment(path, config, [], [], {})


def test_repaired_main_run_can_be_compared_to_current_ablation(tmp_path):
    main = _experiment(tmp_path / 'main', 'combined', 'old')
    low = _experiment(tmp_path / 'low', 'low', 'current')
    main.path.mkdir()
    (main.path / 'recalibration.json').write_text(
        json.dumps({'metrics_source_sha256': 'current'}), encoding='utf-8',
    )

    validate_comparison([low, main])


def test_metric_source_mismatch_without_matching_repair_is_rejected(tmp_path):
    main = _experiment(tmp_path / 'main', 'combined', 'old')
    low = _experiment(tmp_path / 'low', 'low', 'current')
    main.path.mkdir()
    (main.path / 'recalibration.json').write_text(
        json.dumps({'metrics_source_sha256': 'unrelated'}), encoding='utf-8',
    )

    with pytest.raises(ValueError, match='repair source mismatch'):
        validate_comparison([low, main])
