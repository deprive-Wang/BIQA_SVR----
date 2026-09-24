import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import metrics
from metrics import fit_logistic5, logistic5, quality_metrics
from plot_results import load_experiment
from recalibrate_run import recalibrate_run
from train_svr import summarize


def test_projected_fallback_recovers_when_direct_fit_stalls(monkeypatch) -> None:
    monkeypatch.setattr(metrics, 'least_squares',
                        lambda *args, **kwargs: SimpleNamespace(success=False))
    prediction = np.linspace(-2, 2, 233)
    target = logistic5(prediction, np.array([20.0, 3.0, 0.1, 4.0, 50.0]))

    mapped, calibration = fit_logistic5(target, prediction)

    assert calibration['status'] == 'ok'
    assert calibration['optimizer'] == 'variable_projection_differential_evolution'
    assert mapped is not None
    assert np.sqrt(np.mean((mapped - target)**2)) < 0.01


def test_recalibration_keeps_source_and_raw_predictions(tmp_path: Path) -> None:
    source, output = tmp_path / 'source', tmp_path / 'repaired'
    round_path = source / 'round_0000'
    round_path.mkdir(parents=True)
    names = np.array([f'image_{index:04d}.bmp' for index in range(1162)])
    train_indices = np.arange(929)
    test_indices = np.arange(929, 1162)
    prediction = np.linspace(-2, 2, 233)
    target = logistic5(prediction, np.array([20.0, 3.0, 0.1, 4.0, 50.0]))
    report = {'seed': 42, 'train_count': 929, 'test_count': 233,
              'raw': quality_metrics(target, prediction),
              'logistic_mapped': None,
              'calibration': {'status': 'failed', 'reason': 'Direct fit stalled'}}
    (source / 'config.json').write_text(json.dumps({
        'repeats_requested': 1, 'seeds': [42],
        'sample_names': names.tolist(), 'feature_set': 'combined',
    }), encoding='utf-8')
    (source / 'summary.json').write_text(json.dumps({
        'status': 'complete', 'completed_rounds': 1,
        'metrics': summarize([report]),
    }), encoding='utf-8')
    (round_path / 'metrics.json').write_text(json.dumps(report), encoding='utf-8')
    np.savez_compressed(round_path / 'predictions.npz',
                        train_indices=train_indices, test_indices=test_indices,
                        train_names=names[train_indices], test_names=names[test_indices],
                        target=target, prediction_raw=prediction)
    with (round_path / 'predictions.csv').open('w', encoding='utf-8',
                                               newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['name', 'mos', 'prediction_raw', 'prediction_logistic'])
        writer.writerows(zip(names[test_indices], target, prediction, strict=True))

    result = recalibrate_run(source, output, allow_debug=True)
    original = load_experiment(source, allow_debug=True)
    repaired = load_experiment(output, allow_debug=True)

    assert result['repaired_rounds'] == [0]
    assert original.reports[0]['logistic_mapped'] is None
    assert 'prediction_logistic' not in original.predictions[0]
    assert repaired.reports[0]['calibration']['status'] == 'ok'
    assert np.array_equal(original.predictions[0]['prediction_raw'],
                          repaired.predictions[0]['prediction_raw'])
    assert repaired.reports[0]['raw'] == original.reports[0]['raw']
