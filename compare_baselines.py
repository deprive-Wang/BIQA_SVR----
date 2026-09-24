"""Extract a BRISQUE baseline and compare it with BCQI on identical LIVEC splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import ttest_rel

from livec import DEFAULT_ROOT
from metrics import fit_logistic5, quality_metrics
from plot_results import load_experiment
from train_svr import DEFAULT_CACHE, load_training_cache


PROJECT_ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_brisque(cache: Path, root: Path, output: Path, *, device: str = 'cpu',
                    limit: int | None = None) -> None:
    """Save original and quality-oriented BRISQUE scores for aligned LIVEC images."""
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1162):
        raise ValueError('limit must be in [1, 1162]')
    if output.suffix != '.npz' or output.exists():
        raise ValueError('Output must be a new .npz file')
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Output must be outside the raw dataset directory')

    import piq
    import torch

    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    data = load_training_cache(cache, root)
    count = 1162 if limit is None else limit
    names = data['names'][:count]
    scores = []
    image_hashes = []
    hub_dir = PROJECT_ROOT / 'checkpoints' / 'piq'
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))
    for index, name in enumerate(names):
        path = root.resolve() / 'Images' / str(name)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != data['image_sha256'][index]:
                raise ValueError(f'Image changed since BCQI extraction: {name}')
            stream.seek(0)
            with Image.open(stream) as image:
                image.load()
                if image.mode != 'RGB':
                    raise ValueError(f'Expected RGB image: {name}')
                rgb = np.asarray(image).copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            value = float(piq.brisque(tensor, data_range=255.0))
        if not np.isfinite(value):
            raise ValueError(f'Non-finite BRISQUE score: {name}')
        scores.append(value)
        image_hashes.append(digest)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f'BRISQUE {index + 1}/{count}', flush=True)
    weight = hub_dir / 'checkpoints' / 'brisque_svm_weights.pt'
    if not weight.is_file():
        raise ValueError(f'BRISQUE model weights were not cached: {weight}')
    metadata = {
        'method': 'BRISQUE', 'implementation': 'piq.brisque',
        'piq_version': piq.__version__, 'device': device,
        'input': 'original RGB uint8 image; full resolution; no resize',
        'weights_url': 'https://github.com/photosynthesis-team/piq/releases/download/'
                       'v0.4.0/brisque_svm_weights.pt',
        'weight_sha256': _sha256(weight), 'weight_path': str(weight),
        'source_cache_sha256': _sha256(cache),
        'direction': 'lower original score means higher quality; stored score is negated',
        'debug_subset': count != 1162, 'sample_count': count,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        try:
            np.savez_compressed(
                stream, names=names, mos=data['mos'][:count],
                source_scores=np.asarray(scores), scores=-np.asarray(scores),
                image_sha256=np.asarray(image_hashes),
                metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
            )
        except BaseException:
            stream.close()
            output.unlink()
            raise


def _load_baseline(path: Path, names: list[str], mos: np.ndarray,
                   image_hashes: np.ndarray,
                   cache_sha256: str) -> tuple[str, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        required = {'names', 'mos', 'scores', 'image_sha256', 'metadata'}
        if not required.issubset(archive.files):
            raise ValueError(f'Baseline lacks {required - set(archive.files)}: {path}')
        actual_names = archive['names'].tolist()
        actual_mos = archive['mos']
        scores = archive['scores'].astype(float)
        hashes = archive['image_sha256']
        metadata = json.loads(str(archive['metadata']))
    if actual_names != names or not np.array_equal(actual_mos, mos):
        raise ValueError(f'Baseline image IDs or MOS differ: {path}')
    if scores.shape != (len(names),) or not np.isfinite(scores).all():
        raise ValueError(f'Baseline scores must be finite and aligned: {path}')
    if not np.array_equal(hashes, image_hashes):
        raise ValueError(f'Baseline image contents differ from BCQI cache: {path}')
    if (not isinstance(metadata, dict) or metadata.get('debug_subset') is not False
            or metadata.get('source_cache_sha256') != cache_sha256):
        raise ValueError(f'Baseline is not tied to this full feature cache: {path}')
    method = metadata.get('method')
    if not isinstance(method, str) or not method or method == 'BCQI':
        raise ValueError(f'Invalid baseline method name: {path}')
    return method, scores, metadata


def paired_residual_test(target: np.ndarray, bcqi: np.ndarray,
                         baseline: np.ndarray) -> dict:
    """Two-sided paired t-test on per-image absolute residuals."""
    if (target.ndim != 1 or bcqi.shape != target.shape
            or baseline.shape != target.shape or len(target) < 2
            or not np.isfinite(target).all() or not np.isfinite(bcqi).all()
            or not np.isfinite(baseline).all()):
        raise ValueError('Expected finite, aligned score vectors')
    bcqi_error = np.abs(target - bcqi)
    baseline_error = np.abs(target - baseline)
    difference = baseline_error - bcqi_error
    if np.all(difference == 0):
        statistic, p_value = 0.0, 1.0
    elif np.ptp(difference) == 0:
        statistic = float(np.sign(difference[0]) * np.inf)
        p_value = 0.0
    else:
        result = ttest_rel(baseline_error, bcqi_error)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    if not np.isfinite(p_value):
        raise ValueError('Paired t-test produced an undefined p-value')
    return {
        'mean_absolute_error_bcqi': float(bcqi_error.mean()),
        'mean_absolute_error_baseline': float(baseline_error.mean()),
        'mean_baseline_minus_bcqi': float(difference.mean()),
        't_statistic': statistic if np.isfinite(statistic) else None,
        'p_two_sided': p_value, 'n_paired_images': len(target),
    }


def _holm_adjust(tests: dict[str, dict]) -> None:
    ordered = sorted(tests, key=lambda name: tests[name]['p_two_sided'])
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * tests[name]['p_two_sided']))
        test = tests[name]
        test['p_holm'] = running
        test['decision'] = ('equal' if running >= 0.05 else
                            'bcqi_superior' if test['mean_baseline_minus_bcqi'] > 0
                            else 'bcqi_inferior')


def compare(run_path: Path, baseline_paths: list[Path], output: Path,
            *, allow_debug: bool = False) -> dict:
    """Compare independent baseline scores on each saved BCQI test split."""
    if not baseline_paths or output.suffix != '.json' or output.exists():
        raise ValueError('Provide baselines and a new .json output file')
    experiment = load_experiment(run_path, allow_debug=allow_debug)
    config = experiment.config
    names = config['sample_names']
    cache = Path(config['cache_path'])
    if _sha256(cache) != config['cache_sha256']:
        raise ValueError('Run feature cache has changed since training')
    with np.load(cache, allow_pickle=False) as archive:
        if archive['names'].tolist() != names:
            raise ValueError('Run cache image order differs from its config')
        known_mos = archive['mos'].copy()
        image_hashes = archive['image_sha256'].copy()
    if (known_mos.shape != (len(names),) or not np.isfinite(known_mos).all()
            or image_hashes.shape != (len(names),)):
        raise ValueError('Run cache MOS or image hashes are invalid')
    baselines = {}
    metadata = {}
    for path in baseline_paths:
        method, scores, details = _load_baseline(
            path, names, known_mos, image_hashes, config['cache_sha256'],
        )
        if method in baselines:
            raise ValueError(f'Duplicate baseline method: {method}')
        baselines[method] = scores
        metadata[method] = {'archive': str(path.resolve()),
                            'archive_sha256': _sha256(path), **details}
    rounds = []
    metric_names = ('srcc', 'krcc', 'plcc', 'rmse')
    for index, (arrays, bcqi_report) in enumerate(
            zip(experiment.predictions, experiment.reports)):
        test_indices = arrays['test_indices']
        target = arrays['target']
        if not np.array_equal(target, known_mos[test_indices]):
            raise ValueError(f'Run target differs from cache: round {index}')
        baseline_reports = {}
        tests = {}
        for method, all_scores in baselines.items():
            score = all_scores[test_indices]
            mapped, calibration = fit_logistic5(target, score)
            baseline_reports[method] = {
                'raw': quality_metrics(target, score),
                'logistic_mapped': (quality_metrics(target, mapped)
                                    if mapped is not None else None),
                'calibration': calibration,
            }
            if mapped is not None and 'prediction_logistic' in arrays:
                tests[method] = paired_residual_test(
                    target, arrays['prediction_logistic'], mapped,
                )
        _holm_adjust(tests)
        rounds.append({
            'round': index, 'seed': bcqi_report['seed'],
            'test_names': arrays['test_names'].tolist(),
            'bcqi_raw': bcqi_report['raw'],
            'bcqi_logistic_mapped': bcqi_report['logistic_mapped'],
            'baselines': baseline_reports, 'paired_tests': tests,
        })
    summary = {}
    for method in baselines:
        mode_summary = {}
        for mode in ('raw', 'logistic_mapped'):
            mode_summary[mode] = {}
            for metric in metric_names:
                values = [record['baselines'][method][mode][metric] for record in rounds
                          if record['baselines'][method][mode] is not None
                          and record['baselines'][method][mode][metric] is not None]
                mode_summary[mode][metric] = {
                    'mean': float(np.mean(values)) if values else None,
                    'valid_rounds': len(values), 'total_rounds': len(rounds),
                }
        decisions = [record['paired_tests'][method]['decision'] for record in rounds
                     if method in record['paired_tests']]
        summary[method] = {
            'metrics': mode_summary,
            'paired_test': {decision: decisions.count(decision) for decision in
                            ('bcqi_superior', 'equal', 'bcqi_inferior')},
            'valid_rounds': len(decisions), 'total_rounds': len(rounds),
        }
    result = {
        'status': 'complete', 'debug': allow_debug,
        'run': str(run_path.resolve()),
        'run_config_sha256': _sha256(run_path / 'config.json'),
        'protocol': 'same saved test images; test-set five-parameter logistic for reporting',
        'significance': 'two-sided paired t-test of per-image absolute residuals; '
                        'Holm correction across baselines within each split; alpha=0.05; '
                        'splits with repeated images are not pooled',
        'paper_equivalence': 'Exact Table III test procedure is unspecified; this is a project choice',
        'baselines': metadata, 'summary': summary, 'rounds': rounds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extraction = commands.add_parser('extract-brisque')
    extraction.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    extraction.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    extraction.add_argument('--output', type=Path, required=True)
    extraction.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    extraction.add_argument('--limit', type=int)
    comparison = commands.add_parser('compare')
    comparison.add_argument('--run', type=Path, required=True)
    comparison.add_argument('--baseline', type=Path, nargs='+', required=True)
    comparison.add_argument('--output', type=Path, required=True)
    comparison.add_argument('--allow-debug', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'extract-brisque':
            extract_brisque(args.cache, args.root, args.output,
                            device=args.device, limit=args.limit)
        else:
            result = compare(args.run, args.baseline, args.output,
                             allow_debug=args.allow_debug)
            print(json.dumps(result['summary'], ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'Baseline comparison failed: {error}\n')


if __name__ == '__main__':
    main()
