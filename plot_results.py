"""读取 train_svr.py 保存的结果，核对逐轮预测后导出图和统计表。

阅读顺序：load_experiment 核验一次实验 → validate_comparison 核验消融
→ generate_figures 输出图表；这里不重新训练或拟合映射曲线。
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 无窗口渲染，适用于后台实验与自动测试。
import matplotlib.pyplot as plt
import numpy as np

from metrics import logistic5, quality_metrics


METRICS = ('srcc', 'krcc', 'plcc', 'rmse')
MODES = ('raw', 'logistic_mapped')
COLORS = {'combined': '#235789', 'low': '#B47716', 'semantic': '#658149'}
OUTPUT_ROOT = Path(__file__).resolve().parent / 'outputs'


@dataclass
class Experiment:
    """绘图只消费保存的预测，不重训模型、不重新拟合测试集曲线。"""

    path: Path
    config: dict
    reports: list[dict]
    predictions: list[dict[str, np.ndarray]]
    source_hashes: dict[str, str]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _same_number(actual: object, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    return (type(actual) in (int, float) and np.isfinite(actual)
            and bool(np.isclose(actual, expected, rtol=1e-7, atol=1e-9)))


def load_experiment(path: Path, *, allow_debug: bool = False) -> Experiment:
    """拒绝未完成、缺轮、标签错配或汇总与逐轮预测不一致的实验。"""
    path = path.resolve()
    sources = [path / 'config.json', path / 'summary.json']
    config, summary = (_read_json(item) for item in sources)
    count = config.get('repeats_requested')
    seeds = config.get('seeds')
    names = config.get('sample_names')
    if (type(count) is not int or count < 1 or not isinstance(seeds, list)
            or len(seeds) != count or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != count):
        raise ValueError('Invalid repeat count or seeds')
    if (summary.get('status') != 'complete'
            or summary.get('completed_rounds') != count):
        raise ValueError('Experiment is incomplete')
    if not allow_debug and count != 1000:
        raise ValueError('Expected 1000 completed rounds; use --allow-debug for diagnostics')
    if (not isinstance(names, list) or len(names) != 1162
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != 1162):
        raise ValueError('Expected 1162 unique sample names')
    if config.get('feature_set') not in COLORS:
        raise ValueError('Unknown feature set')
    expected_dirs = {f'round_{i:04d}' for i in range(count)}
    if {p.name for p in path.glob('round_*') if p.is_dir()} != expected_dirs:
        raise ValueError('Missing or unexpected round directories')
    reports, predictions = [], []
    known_mos: dict[str, float] = {}
    for index, seed in enumerate(seeds):
        directory = path / f'round_{index:04d}'
        sources.extend([directory / 'metrics.json', directory / 'predictions.npz'])
        report = _read_json(sources[-2])
        with np.load(sources[-1], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        required = {'train_indices', 'test_indices', 'train_names', 'test_names',
                    'target', 'prediction_raw'}
        if not required.issubset(arrays):
            raise ValueError(f'Missing prediction fields: {directory}')
        if (report.get('seed') != seed or report.get('train_count') != 929
                or report.get('test_count') != 233):
            raise ValueError(f'Invalid split counts/seed: {directory}')
        for split, size in [('train', 929), ('test', 233)]:
            ids = arrays[f'{split}_indices']
            if (ids.shape != (size,) or ids.dtype.kind not in 'iu'
                    or (ids < 0).any() or (ids >= 1162).any()
                    or len(np.unique(ids)) != size
                    or not np.array_equal(arrays[f'{split}_names'], np.array(names)[ids])):
                raise ValueError(f'Invalid {split} IDs/names: {directory}')
        if np.intersect1d(arrays['train_indices'], arrays['test_indices']).size:
            raise ValueError(f'Train/test overlap: {directory}')
        for key in ('target', 'prediction_raw', 'prediction_logistic'):
            if key not in arrays and key == 'prediction_logistic':
                continue
            values = arrays[key]
            if (values.shape != (233,) or values.dtype.kind not in 'fiu'
                    or not np.isfinite(values).all()):
                raise ValueError(f'Invalid {key}: {directory}')
        if ((arrays['target'] < 0) | (arrays['target'] > 100)).any():
            raise ValueError('MOS outside [0,100]')
        for name, mos in zip(arrays['test_names'], arrays['target']):
            if name in known_mos and known_mos[name] != mos:
                raise ValueError('MOS changes across rounds for the same image')
            known_mos[name] = float(mos)
        calibration = report.get('calibration', {})
        mapped = arrays.get('prediction_logistic')
        if calibration.get('status') == 'ok':
            parameters = np.asarray(calibration.get('parameters'), dtype=float)
            center, scale = calibration.get('prediction_center'), calibration.get('prediction_scale')
            if (parameters.shape != (5,) or not np.isfinite(parameters).all()
                    or type(center) not in (int, float) or not np.isfinite(center)
                    or type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0):
                raise ValueError('Invalid saved logistic parameters')
            restored = logistic5((arrays['prediction_raw'] - center) / scale, parameters)
            if mapped is None or not np.allclose(restored, mapped, rtol=1e-7, atol=1e-9):
                raise ValueError('Saved logistic predictions disagree with parameters')
        elif (calibration.get('status') not in {'failed', 'unavailable'}
              or mapped is not None or report.get('logistic_mapped') is not None):
            raise ValueError('Inconsistent logistic fit status')
        for mode, key in [('raw', 'prediction_raw'), ('logistic_mapped', 'prediction_logistic')]:
            if key not in arrays:
                continue
            computed = quality_metrics(arrays['target'], arrays[key])
            saved = report.get(mode)
            if not isinstance(saved, dict) or any(
                not _same_number(saved.get(metric), value) for metric, value in computed.items()
            ):
                raise ValueError(f'Metrics disagree with predictions: {directory}/{mode}')
        reports.append(report)
        predictions.append(arrays)
    experiment = Experiment(path, config, reports, predictions, {})
    # 重新从逐轮指标聚合，不能仅相信 summary.json 的 complete 标记。
    for mode in MODES:
        for metric in METRICS:
            values = metric_values(experiment, mode, metric)
            saved = summary.get('metrics', {}).get(mode, {}).get(metric, {})
            if (saved.get('valid_rounds') != len(values) or saved.get('total_rounds') != count
                    or not _same_number(saved.get('mean'), float(values.mean()) if len(values) else None)
                    or not _same_number(saved.get('std_population'), float(values.std()) if len(values) else None)):
                raise ValueError(f'Summary mismatch: {mode}/{metric}')
    experiment.source_hashes = {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    return experiment


def metric_values(experiment: Experiment, mode: str, metric: str) -> np.ndarray:
    """保留有效值，不把失败拟合或未定义相关系数填成零。"""
    return np.array([r[mode][metric] for r in experiment.reports
                     if r[mode] is not None and r[mode][metric] is not None], dtype=float)


def validate_comparison(experiments: list[Experiment]) -> None:
    """消融只允许特征列不同；数据来源、调参协议和每轮划分必须一致。"""
    if len({e.config['feature_set'] for e in experiments}) != len(experiments):
        raise ValueError('Duplicate feature sets')
    reference = experiments[0]
    for experiment in experiments[1:]:
        for key in ('cache_sha256', 'sample_names', 'seeds', 'inner_cv', 'parameter_grid',
                    'standardization', 'svr_defaults', 'source_sha256', 'logistic'):
            if key not in reference.config or experiment.config.get(key) != reference.config[key]:
                raise ValueError(f'Ablation protocol mismatch: {key}')
        for left, right, lr, rr in zip(reference.predictions, experiment.predictions,
                                       reference.reports, experiment.reports):
            for key in ('train_indices', 'test_indices', 'train_names', 'test_names', 'target'):
                if not np.array_equal(left[key], right[key]):
                    raise ValueError(f'Ablation split/label mismatch: {key}')
            if lr.get('inner_folds') is None or lr['inner_folds'] != rr.get('inner_folds'):
                raise ValueError('Ablation inner folds mismatch')


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def _distribution(experiments: list[Experiment], output: Path, mode: str, scope: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, metric in zip(axes.flat, METRICS):
        labels = []
        for position, experiment in enumerate(experiments, 1):
            values = metric_values(experiment, mode, metric)
            feature = experiment.config['feature_set']
            labels.append(f'{feature}\nvalid {len(values)}/{len(experiment.reports)}')
            if len(values):
                ax.boxplot([values], positions=[position], widths=0.45,
                           patch_artist=True, manage_ticks=False,
                           boxprops={'facecolor': COLORS[feature], 'alpha': 0.35},
                           medianprops={'color': '#333333'},
                           flierprops={'marker': '.', 'markersize': 3})
                ax.errorbar(position + 0.22, values.mean(), yerr=values.std(),
                            fmt='D', color=COLORS[feature], capsize=4, markersize=5)
            else:
                ax.text(position, 0.5, 'Unavailable', ha='center',
                        transform=ax.get_xaxis_transform())
        ax.set_xticks(range(1, len(experiments) + 1), labels)
        ax.set_xlim(0.5, len(experiments) + 0.6)
        ax.set_ylabel(metric.upper() + (' (MOS points)' if metric == 'rmse' else ''))
        ax.set_title('Lower is better' if metric == 'rmse' else 'Higher is better')
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle(f'{scope} | {mode}\nBoxes: median/IQR, whiskers: 1.5 IQR; diamonds: mean +/- SD (not CI)')
    fig.supxlabel('One observation per random split; logistic mapping is test-set post-hoc. PWRC not implemented.')
    _save(fig, output, f'metrics_{mode}')


def _diagnostics(experiment: Experiment, output: Path, index: int, scope: str) -> None:
    arrays, report = experiment.predictions[index], experiment.reports[index]
    target, raw = arrays['target'], arrays['prediction_raw']
    mapped = arrays.get('prediction_logistic')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    # 原始/映射后使用同一坐标范围和直方图分箱，避免自动缩放制造改善的错觉。
    available = [raw] if mapped is None else [raw, mapped]
    all_values = np.concatenate([target, *available])
    lo, hi = min(0, all_values.min()), max(100, all_values.max())
    all_residuals = np.concatenate([values - target for values in available])
    residual_limit = max(float(np.abs(all_residuals).max()) * 1.05, 1.0)
    bins = np.histogram_bin_edges(all_residuals, bins='auto')
    axes[1, 2].sharey(axes[0, 2])
    for row, (label, values) in enumerate([('Raw', raw), ('Post-hoc mapped', mapped)]):
        if values is None:
            for ax in axes[row]:
                ax.text(0.5, 0.5, 'Logistic fit unavailable', ha='center', transform=ax.transAxes)
                ax.set_axis_off()
            continue
        axes[row, 0].scatter(target, values, s=12, alpha=0.55, color='#235789')
        axes[row, 0].plot([lo, hi], [lo, hi], '--', color='#333333')
        axes[row, 0].set(xlabel='MOS', ylabel=f'{label} prediction', xlim=(lo, hi), ylim=(lo, hi))
        axes[row, 0].set_aspect('equal', adjustable='box')
        residual = values - target
        axes[row, 1].scatter(target, residual, s=12, alpha=0.55, color='#235789')
        axes[row, 1].axhline(0, linestyle='--', color='#333333')
        axes[row, 1].set(xlabel='MOS', ylabel=f'{label} prediction - MOS',
                         ylim=(-residual_limit, residual_limit))
        axes[row, 2].hist(residual, bins=bins, color='#235789', alpha=0.75)
        axes[row, 2].set(xlabel='Prediction - MOS', ylabel='Image count')
    feature = experiment.config['feature_set']
    fig.suptitle(f'{scope} | {feature} | diagnostic round {index}, seed {report["seed"]}, n=233\n'
                 'Single-round illustration, not the aggregate experiment result')
    _save(fig, output, f'{feature}_round_{index:04d}_diagnostics')
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    ax.scatter(raw, target, s=15, alpha=0.5, color='#235789', label='Test images')
    if mapped is not None:
        calibration = report['calibration']
        grid = np.linspace(raw.min(), raw.max(), 300)
        curve = logistic5((grid - calibration['prediction_center']) / calibration['prediction_scale'],
                          np.array(calibration['parameters']))
        ax.plot(grid, curve, color='#B47716', label='Saved five-parameter mapping')
    else:
        ax.text(0.05, 0.95, 'Logistic fit unavailable', transform=ax.transAxes, va='top')
    ax.set(xlabel='Raw prediction', ylabel='MOS / mapped prediction',
           title=f'{scope}\n{feature}, round {index}: test-set post-hoc fit only')
    ax.legend()
    _save(fig, output, f'{feature}_round_{index:04d}_logistic')


def generate_figures(paths: list[Path], output: Path, *, allow_debug: bool = False,
                     round_index: int = 0) -> None:
    """默认读取 1000 轮；诊断固定取第 0 轮，不按测试指标挑选最好轮次。"""
    if not paths:
        raise ValueError('At least one experiment is required')
    output = output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output == OUTPUT_ROOT:
        raise ValueError('Figure output must be a new subdirectory of project outputs/')
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    experiments = [load_experiment(path, allow_debug=allow_debug) for path in paths]
    validate_comparison(experiments)
    if type(round_index) is not int or not 0 <= round_index < len(experiments[0].reports):
        raise ValueError('Diagnostic round index out of range')
    count = len(experiments[0].reports)
    scope = f'{"DEBUG" if allow_debug else "LIVEC"} | {count} completed splits'
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42}):
        for mode in MODES:
            _distribution(experiments, output, mode, scope)
        for experiment in experiments:
            _diagnostics(experiment, output, round_index, scope)
    # CSV 与图使用相同有效轮次，避免将跨轮相关样本视为独立样本计算置信区间。
    with (output / 'metrics_summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['feature_set', 'mode', 'metric', 'mean', 'std_population',
                         'valid_rounds', 'total_rounds'])
        for experiment in experiments:
            for mode in MODES:
                for metric in METRICS:
                    values = metric_values(experiment, mode, metric)
                    writer.writerow([experiment.config['feature_set'], mode, metric,
                                     values.mean() if len(values) else '',
                                     values.std() if len(values) else '', len(values), count])
    manifest = {
        'status': 'complete', 'debug': allow_debug, 'completed_rounds': count,
        'diagnostic_round': round_index, 'matplotlib': matplotlib.__version__,
        'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['PWRC not implemented; not complete Table II replication',
                        'Single-round diagnostics are not aggregate results',
                        'Error bars are population SD over splits, not confidence intervals',
                        'Logistic mapping uses test MOS for post-hoc reporting only'],
        'experiments': [{'path': str(e.path), 'feature_set': e.config['feature_set'],
                         'source_sha256': e.source_hashes,
                         'protocol_limits': e.config.get('protocol_limits', [])} for e in experiments],
    }
    with (output / 'plot_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, default=0, dest='round_index')
    parser.add_argument('--allow-debug', action='store_true', help='Label all figures DEBUG')
    args = parser.parse_args()
    try:
        generate_figures(args.runs, args.output, allow_debug=args.allow_debug,
                         round_index=args.round_index)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Plotting failed: {error}\n')
    print(f'Saved figures: {args.output}')


if __name__ == '__main__':
    main()
