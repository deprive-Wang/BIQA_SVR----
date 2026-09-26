"""Compute the PWRC area under the sensory-threshold curve from saved scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from metrics import logistic5, quality_metrics
from plot_results import load_experiment


DEFAULT_RUN = Path(__file__).resolve().parent / 'outputs/svr_1000_logistic_fixed'
DEFAULT_BASELINES = (
    Path(__file__).resolve().parent / 'outputs/compare_brisque.json',
    Path(__file__).resolve().parent / 'outputs/compare_niqe.json',
)


def pwrc_auc(
    mos: np.ndarray, prediction: np.ndarray, *, mos_min: float, mos_max: float,
    threshold_min: float, threshold_max: float, steepness: float = 0.175,
) -> float:
    """Integrate Wu et al.'s PWRC, Eqs. (9), (13)-(15), (19), analytically.

    MOS and prediction ties receive average ranks; a tied pair contributes zero
    rank agreement. The original paper assumes distinct ranks in its derivation.
    """
    mos = np.asarray(mos, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if (mos.ndim != 1 or prediction.shape != mos.shape or len(mos) < 2
            or not np.isfinite(mos).all() or not np.isfinite(prediction).all()):
        raise ValueError('Expected finite, aligned one-dimensional scores')
    if (not np.isfinite([mos_min, mos_max, threshold_min, threshold_max,
                         steepness]).all() or mos_min >= mos_max
            or threshold_min >= threshold_max or steepness <= 0):
        raise ValueError('Invalid PWRC normalization or threshold range')
    if mos.min() < mos_min or mos.max() > mos_max:
        raise ValueError('MOS outside declared full-dataset range')

    normalized = (mos - mos_min) * (100.0 / (mos_max - mos_min))
    truth_rank = rankdata(normalized, method='average')
    prediction_rank = rankdata(prediction, method='average')
    left, right = np.triu_indices(len(mos), k=1)
    count = len(mos)

    rank_deviation = (np.abs(truth_rank[left] - prediction_rank[left])
                      + np.abs(truth_rank[right] - prediction_rank[right])) / (2 * count - 2)
    rank_level = (np.maximum(truth_rank[left], truth_rank[right]) - 1) / (count - 1)
    importance = np.expm1(rank_deviation) + np.expm1(rank_level)
    total_importance = importance.sum()
    if total_importance <= 0:
        raise ValueError('PWRC importance weights are degenerate')
    agreement = (np.sign(truth_rank[left] - truth_rank[right])
                 * np.sign(prediction_rank[left] - prediction_rank[right]))

    distance = np.abs(normalized[left] - normalized[right])
    lower = steepness * (distance - threshold_min)
    upper = steepness * (distance - threshold_max)
    # Integral of expit(c * (distance - T)) over [Tmin, Tmax].
    activation_integral = (np.logaddexp(0, lower) - np.logaddexp(0, upper)) / steepness
    return float(np.dot(importance * agreement, activation_integral) / total_importance)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_baseline_report(path: Path, run: Path, run_config_sha256: str,
                          expected_names: list[str], count: int) -> tuple[str, dict]:
    report = json.loads(path.read_text(encoding='utf-8'))
    if (report.get('status') != 'complete' or report.get('debug') is not False
            or Path(report.get('run', '')).resolve() != run.resolve()
            or report.get('run_config_sha256') != run_config_sha256
            or len(report.get('rounds', [])) != count
            or len(report.get('baselines', {})) != 1):
        raise ValueError(f'Baseline report is not this complete run: {path}')
    method, metadata = next(iter(report['baselines'].items()))
    archive = Path(metadata['archive'])
    if _sha256(archive) != metadata['archive_sha256']:
        raise ValueError(f'Baseline score archive changed: {archive}')
    with np.load(archive, allow_pickle=False) as data:
        if data['names'].tolist() != expected_names:
            raise ValueError(f'Baseline score order differs: {archive}')
    return method, report


def _mapped_from_saved(raw: np.ndarray, calibration: dict) -> np.ndarray:
    if calibration.get('status') != 'ok':
        raise ValueError('Saved logistic calibration is unavailable')
    center = float(calibration['prediction_center'])
    scale = float(calibration['prediction_scale'])
    parameters = np.asarray(calibration['parameters'], dtype=float)
    if (not np.isfinite([center, scale]).all() or scale <= 0
            or parameters.shape != (5,) or not np.isfinite(parameters).all()):
        raise ValueError('Invalid saved logistic calibration')
    return logistic5((raw - center) / scale, parameters)


def backfill(run: Path, baseline_paths: tuple[Path, ...], output: Path) -> dict:
    """Backfill 1000-split PWRC values without training or re-fitting mappings."""
    if output.exists() or output.suffix != '.json':
        raise ValueError('PWRC output must be a new .json file')
    experiment = load_experiment(run)
    config = experiment.config
    cache = Path(config['cache_path'])
    if _sha256(cache) != config['cache_sha256']:
        raise ValueError('Run feature cache changed since training')
    with np.load(cache, allow_pickle=False) as data:
        all_mos = data['mos'].astype(float)
        all_stddev = data['stddev'].astype(float)
        names = data['names'].tolist()
    if (names != config['sample_names'] or all_mos.shape != (1162,)
            or all_stddev.shape != (1162,) or not np.isfinite(all_mos).all()
            or not np.isfinite(all_stddev).all() or (all_stddev < 0).any()):
        raise ValueError('Invalid full-dataset MOS, standard deviation, or image order')
    mos_min, mos_max = float(all_mos.min()), float(all_mos.max())
    normalized_stddev = all_stddev * (100.0 / (mos_max - mos_min))
    threshold_min = float(2 * normalized_stddev.min())
    threshold_max = float(2 * normalized_stddev.max())
    run_config_sha256 = _sha256(run / 'config.json')
    baselines = {}
    source_reports = {}
    for path in baseline_paths:
        method, report = _load_baseline_report(
            path, run, run_config_sha256, names, len(experiment.reports),
        )
        if method in baselines:
            raise ValueError(f'Duplicate baseline method: {method}')
        metadata = report['baselines'][method]
        with np.load(metadata['archive'], allow_pickle=False) as data:
            if not np.array_equal(data['mos'], all_mos):
                raise ValueError(f'Baseline MOS differs: {method}')
            baselines[method] = data['scores'].astype(float)
        source_reports[method] = {'path': str(path.resolve()), 'sha256': _sha256(path),
                                  'archive_sha256': metadata['archive_sha256'],
                                  'rounds': report['rounds']}

    methods = ('BCQI', *baselines)
    values = {method: {'raw': [], 'logistic_mapped': []} for method in methods}
    for index, arrays in enumerate(experiment.predictions):
        target = arrays['target']
        test_indices = arrays['test_indices']
        if not np.array_equal(target, all_mos[test_indices]):
            raise ValueError(f'BCQI target differs from cache: round {index}')
        for mode, field in (('raw', 'prediction_raw'),
                            ('logistic_mapped', 'prediction_logistic')):
            values['BCQI'][mode].append(pwrc_auc(
                target, arrays[field], mos_min=mos_min, mos_max=mos_max,
                threshold_min=threshold_min, threshold_max=threshold_max,
            ) if field in arrays else None)
        for method, scores in baselines.items():
            record = source_reports[method]['rounds'][index]
            if (record['round'] != index or record['seed'] != experiment.reports[index]['seed']
                    or record['test_names'] != arrays['test_names'].tolist()):
                raise ValueError(f'Baseline split differs: {method}, round {index}')
            raw = scores[test_indices]
            saved = record['baselines'][method]
            computed_raw = quality_metrics(target, raw)
            if any(not np.isclose(computed_raw[key], saved['raw'][key],
                                  rtol=1e-7, atol=1e-9)
                   for key in computed_raw):
                raise ValueError(f'Baseline raw metrics differ: {method}, round {index}')
            mapped = _mapped_from_saved(raw, saved['calibration'])
            computed_mapped = quality_metrics(target, mapped)
            if any(not np.isclose(computed_mapped[key], saved['logistic_mapped'][key],
                                  rtol=1e-7, atol=1e-9)
                   for key in computed_mapped):
                raise ValueError(f'Baseline mapped metrics differ: {method}, round {index}')
            for mode, prediction in (('raw', raw), ('logistic_mapped', mapped)):
                values[method][mode].append(pwrc_auc(
                    target, prediction, mos_min=mos_min, mos_max=mos_max,
                    threshold_min=threshold_min, threshold_max=threshold_max,
                ))
        if (index + 1) % 100 == 0:
            print(f'PWRC {index + 1}/{len(experiment.reports)}', flush=True)

    summary = {}
    for method in methods:
        summary[method] = {}
        for mode in ('raw', 'logistic_mapped'):
            finite = np.asarray([value for value in values[method][mode]
                                 if value is not None], dtype=float)
            summary[method][mode] = {
                'mean': float(finite.mean()) if len(finite) else None,
                'valid_rounds': len(finite), 'total_rounds': len(experiment.reports),
            }
    result = {
        'status': 'complete', 'rounds': len(experiment.reports),
        'definition': 'PWRC sensory-threshold AUC, Wu et al. 2018 Eqs. (9), (13)-(15), (19)',
        'reference': 'https://arxiv.org/pdf/1705.05126',
        'score_direction': 'higher means better for all three saved score vectors',
        'mos_normalization': {'min': mos_min, 'max': mos_max, 'range': [0, 100]},
        'threshold_range': [threshold_min, threshold_max], 'steepness': 0.175,
        'integration': 'analytic logistic integral',
        'ties': 'average ranks; tied pair agreement=0',
        'source': {'run': str(run.resolve()),
                   'run_config_sha256': run_config_sha256,
                   'feature_cache_sha256': config['cache_sha256'],
                   'baseline_reports': {method: {key: value for key, value in source.items()
                                                 if key != 'rounds'}
                                        for method, source in source_reports.items()}},
        'summary': summary, 'per_round': values,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--baseline', type=Path, nargs='+', default=list(DEFAULT_BASELINES))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        backfill(args.run, tuple(args.baseline), args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'PWRC backfill failed: {error}\n')
    print(f'Saved PWRC result: {args.output}')


if __name__ == '__main__':
    main()
