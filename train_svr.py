"""从 (1162,1007) 特征缓存训练 RBF SVR，并保存每轮预测与评估。

阅读顺序：load_training_cache 核对图像和特征 → train_one_split 划分与训练
→ summarize 汇总多轮指标 → main 保存可追溯的实验产物。
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from livec import DEFAULT_ROOT, load_livec
from low_level import FEATURE_NAMES
from metrics import fit_logistic5, quality_metrics
from semantic import SEMANTIC_NAMES


DEFAULT_CACHE = Path(__file__).resolve().parent / 'features/livec_combined_v2.npz'
# II-D / 式 (19)：C 控制超出 epsilon 容忍带的误差惩罚，gamma 控制 RBF 局部性。
# 下面的搜索范围与三折 CV 属于本项目选择，未核验为作者的超参数配置。
PARAMETER_GRID = {
    'svr__C': [1.0, 10.0, 100.0],
    'svr__gamma': ['scale', 0.001, 0.01],
    'svr__epsilon': [0.1, 1.0],
}


def make_pipeline() -> Pipeline:
    """Single source of truth for the estimator and the defaults recorded in config."""
    # 各列的量纲不同，需要训练集均值/标准差。放在 Pipeline 内保证每个 CV 训练折
    # 独立拟合 scaler，验证折和外层测试集只调用 transform，避免数据泄漏。
    return Pipeline([('scale', StandardScaler()),
                     ('svr', SVR(kernel='rbf', cache_size=256))])


def load_training_cache(path: Path, root: Path) -> dict[str, np.ndarray]:
    """Validate the full combined cache against current annotations and code."""
    with np.load(path, allow_pickle=False) as archive:
        required = {'features', 'names', 'mos', 'stddev', 'annotation_indices',
                    'image_sha256', 'feature_names', 'metadata'}
        if not required.issubset(archive.files):
            raise ValueError('Combined cache is missing required fields')
        data = {key: archive[key].copy() for key in required}
    samples = load_livec(root, verify_images=False)
    features = data['features']
    if (features.shape != (1162, 1007) or features.dtype.kind not in 'fiu'
            or not np.isfinite(features).all()):
        raise ValueError('Expected finite full LIVEC features of shape (1162,1007)')
    expected = {
        'names': [sample.name for sample in samples],
        'mos': [sample.mos for sample in samples],
        'stddev': [sample.stddev for sample in samples],
        'annotation_indices': [sample.annotation_index for sample in samples],
        'feature_names': FEATURE_NAMES + SEMANTIC_NAMES,
    }
    for key, values in expected.items():
        if not np.array_equal(data[key], np.asarray(values)):
            raise ValueError(f'Cache {key} does not match current dataset/order')
    if data['image_sha256'].shape != (1162,):
        raise ValueError('Invalid image hash vector')
    for sample, expected_hash in zip(samples, data['image_sha256']):
        with sample.path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_hash:
                raise ValueError(f'Image has changed since extraction: {sample.name}')
    metadata = json.loads(str(data['metadata']))
    if not isinstance(metadata, dict) or metadata.get('sample_count') != 1162:
        raise ValueError('Invalid cache metadata')
    if metadata.get('debug_subset') is not False:
        raise ValueError('A debug subset cannot be used as the full dataset')
    if metadata.get('feature_layout') != {'low_level': [0, 7], 'semantic': [7, 1007]}:
        raise ValueError('Unexpected feature layout')
    low_metadata = metadata.get('low_level')
    if not isinstance(low_metadata, dict):
        raise ValueError('Combined cache is missing nested low-level metadata')
    # Both halves of the 1007-D cache are checked: the semantic extractor itself
    # and the low-level extractor recorded in the nested low-level metadata.
    recorded = {
        ('semantic.py', 'extract_semantic.py'): metadata.get('source_sha256'),
        ('low_level.py', 'livec.py', 'extract_features.py'):
            low_metadata.get('source_sha256'),
    }
    for names, sources in recorded.items():
        if sources != {name: hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()
        ).hexdigest() for name in names}:
            raise ValueError(f'Source changed since extraction {names}; '
                             'verify or regenerate the feature cache')
    return data


def train_one_split(
    features: np.ndarray, mos: np.ndarray, *, seed: int,
    parameter_grid: dict | None = None, cv_folds: int = 3, jobs: int = 1,
) -> tuple[Pipeline, dict, dict[str, np.ndarray]]:
    """输入 X=(N,D)、MOS=(N,)；返回模型、指标报告和测试集预测。"""
    if (features.ndim != 2 or mos.shape != (len(features),)
            or not np.isfinite(features).all() or not np.isfinite(mos).all()):
        raise ValueError('Expected finite (N,D) features and (N,) MOS')
    if features.shape[1] == 0 or len(mos) < 10 or np.ptp(mos) == 0:
        raise ValueError('Need at least 10 samples, nonempty features and varying MOS')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0,2**32)')
    if type(cv_folds) is not int or not 2 <= cv_folds <= int(0.8 * len(mos)):
        raise ValueError('Invalid inner CV fold count')
    if type(jobs) is not int or jobs == 0 or jobs < -1:
        raise ValueError('jobs must be positive or -1')
    # III-B：按图像随机分 80%/20%；1162 张时测试数向上取整为 233，训练为 929。
    # 返回的是全数据行索引，同一索引同时切分特征 X 和主观评分 y。
    train_indices, test_indices = train_test_split(
        np.arange(len(mos)), test_size=0.2, random_state=seed, shuffle=True,
    )
    if np.ptp(mos[train_indices]) == 0:
        raise ValueError('Training MOS is constant for this split')
    # 外层测试集只用于最终评估；内层交叉验证只看到训练集。
    # inner_splits 的索引相对训练子集，而非全数据。
    inner_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    inner_splits = list(inner_cv.split(features[train_indices]))
    search = GridSearchCV(
        make_pipeline(), PARAMETER_GRID if parameter_grid is None else parameter_grid,
        scoring='neg_mean_squared_error', cv=inner_splits, n_jobs=jobs,
        refit=True, error_score='raise', return_train_score=False,
    )
    start = time.perf_counter()
    # 式 (1)、(18)-(19)：以 (929,D) 特征和 (929,) MOS 训练；D 默认 1007。
    # 负 MSE 越大越好；refit=True 用最优参数在整个外层训练集重新拟合 Pipeline。
    search.fit(features[train_indices], mos[train_indices])
    # 式 (2)：测试特征 (233,D) -> 原始质量预测 (233,)，此后才接触测试 MOS。
    raw = search.predict(features[test_indices])
    # III-A / 式 (20)：利用测试 MOS 的事后曲线拟合仅用于报告，不反馈给调参。
    # 保存的模型仍输出 raw；它不含这条依赖测试标签的曲线。
    mapped, calibration = fit_logistic5(mos[test_indices], raw)
    report = {
        'seed': seed, 'train_count': len(train_indices), 'test_count': len(test_indices),
        'best_params': search.best_params_, 'inner_cv_mse': float(-search.best_score_),
        'raw': quality_metrics(mos[test_indices], raw),
        'logistic_mapped': (quality_metrics(mos[test_indices], mapped)
                            if mapped is not None else None),
        'calibration': calibration, 'seconds': time.perf_counter() - start,
        'inner_folds': [{'train_indices': train_indices[train].tolist(),
                         'validation_indices': train_indices[validation].tolist()}
                        for train, validation in inner_splits],
        'cv_candidates': [
            {'params': params, 'mean_validation_mse': float(-mean),
             'std_validation_mse': float(std)}
            for params, mean, std in zip(search.cv_results_['params'],
                                        search.cv_results_['mean_test_score'],
                                        search.cv_results_['std_test_score'])
        ],
    }
    arrays = {'train_indices': train_indices, 'test_indices': test_indices,
              'target': mos[test_indices], 'prediction_raw': raw}
    if mapped is not None:
        arrays['prediction_logistic'] = mapped
    return search.best_estimator_, report, arrays


def summarize(reports: list[dict]) -> dict:
    """Keep unavailable metrics explicit with their valid-round counts."""
    summary = {}
    for mode in ('raw', 'logistic_mapped'):
        summary[mode] = {}
        for metric in ('srcc', 'krcc', 'plcc', 'rmse'):
            # 未定义的相关系数或失败的映射不记作 0；同时记录有效轮数以揭示缺失。
            values = [report[mode][metric] for report in reports
                      if report[mode] is not None and report[mode][metric] is not None]
            summary[mode][metric] = {
                'mean': float(np.mean(values)) if values else None,
                'std_population': float(np.std(values)) if values else None,
                'valid_rounds': len(values), 'total_rounds': len(reports),
            }
    return summary


def _write_json(path: Path, content: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main() -> None:
    """读取已校验缓存，按种子重复训练，逐轮保存模型、预测和指标。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv-folds', type=int, default=3)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--save-models', choices=('first', 'all', 'none'), default='first',
                        help='Save only the first round by default to bound disk usage')
    parser.add_argument('--feature-set', choices=('combined', 'low', 'semantic'), default='combined')
    args = parser.parse_args()
    if args.repeats < 1 or not 0 <= args.seed <= 2**32 - args.repeats:
        parser.error('Invalid repeat count or seed range')
    if not 2 <= args.cv_folds <= 929 or args.jobs == 0 or args.jobs < -1:
        parser.error('Invalid CV fold count or jobs')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new run name')
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error('Output must be outside the raw dataset directory')
    try:
        data = load_training_cache(args.features, args.root)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot load training data: {error}\n')
    # 消融只改变 X 的列：前 7 列为低层，后 1000 列为语义。
    # 相同 seed/repeats 下，三种设置仍使用相同图像划分。
    selection = {'combined': slice(None), 'low': slice(0, 7),
                 'semantic': slice(7, 1007)}[args.feature_set]
    features = data['features'][:, selection]
    config = {
        'feature_set': args.feature_set, 'save_models': args.save_models,
        'feature_names': data['feature_names'][selection].tolist(),
        'sample_names': data['names'].tolist(),
        'cache_path': str(args.features.resolve()),
        'cache_sha256': hashlib.sha256(args.features.read_bytes()).hexdigest(),
        'feature_metadata': json.loads(str(data['metadata'])),
        'repeats_requested': args.repeats,
        'split': 'train_test_split test_size=0.2; ceil test size = 233; train = 929',
        'seeds': list(range(args.seed, args.seed + args.repeats)),
        'inner_cv': {'folds': args.cv_folds, 'shuffle': True, 'scoring': 'neg_mean_squared_error'},
        'parameter_grid': PARAMETER_GRID, 'svr_device': 'CPU (scikit-learn/libsvm)',
        # Read back from the estimator so the record cannot drift from the code.
        'svr_defaults': {key: value for key, value in make_pipeline()['svr'].get_params().items()
                         if f'svr__{key}' not in PARAMETER_GRID},
        'standardization': 'StandardScaler in each inner fold; refit on outer training only',
        'logistic': 'Post-hoc on outer test MOS only for reporting; not model selection',
        'protocol_limits': ['Hyperparameter grid is a project choice, not author-confirmed',
                            'PWRC is not implemented; this is not complete Table II replication'],
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__},
        'source_sha256': {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('train_svr.py', 'metrics.py')
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output / 'config.json', config)
    reports = []
    # III-B 的正式协议为 1000 次并取均值；默认 1 次只是开发验证，不是正式复现。
    # 每轮重新划分、调参和训练，并保存索引、预测及指标以便追溯。
    for round_index, seed in enumerate(config['seeds']):
        model, report, arrays = train_one_split(
            features, data['mos'], seed=seed, cv_folds=args.cv_folds, jobs=args.jobs,
        )
        run = args.output / f'round_{round_index:04d}'
        run.mkdir()
        if args.save_models == 'all' or (args.save_models == 'first' and round_index == 0):
            with (run / 'model.pkl').open('xb') as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(run / 'predictions.npz', **arrays,
                            train_names=data['names'][arrays['train_indices']],
                            test_names=data['names'][arrays['test_indices']])
        with (run / 'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['name', 'mos', 'prediction_raw', 'prediction_logistic'])
            mapped = arrays.get('prediction_logistic', [None] * len(arrays['target']))
            writer.writerows(zip(data['names'][arrays['test_indices']], arrays['target'],
                                 arrays['prediction_raw'], mapped))
        _write_json(run / 'metrics.json', report)
        reports.append(report)
        print(f"Round {round_index + 1}/{args.repeats}, seed={seed}: "
              f"raw={report['raw']}, logistic={report['logistic_mapped']}", flush=True)
    _write_json(args.output / 'summary.json', {
        'status': 'complete', 'completed_rounds': len(reports),
        'metrics': summarize(reports),
    })
    print(f'Saved experiment: {args.output}')


if __name__ == '__main__':
    main()
