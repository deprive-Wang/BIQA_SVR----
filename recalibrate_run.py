"""从已保存的原始预测修复失败的测试集 logistic 映射，不重训 SVR。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from metrics import fit_logistic5, quality_metrics
from plot_results import load_experiment
from train_svr import DEFAULT_ROOT, summarize


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')


def recalibrate_run(source: Path, output: Path, *, allow_debug: bool = False) -> dict:
    """复制已验证的实验，只重算映射失败的轮次并核验完整副本。"""
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('Output must be a new directory outside the source run')
    if output.is_relative_to(DEFAULT_ROOT.resolve()):
        raise ValueError('Output must be outside the raw dataset directory')

    experiment = load_experiment(source, allow_debug=allow_debug)
    failed_indices = [index for index, report in enumerate(experiment.reports)
                      if report['calibration']['status'] != 'ok']
    if not failed_indices:
        raise ValueError('Source run has no failed logistic mappings')

    original_summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        shutil.copytree(source, stage)
        reports = list(experiment.reports)
        for index in failed_indices:
            arrays = dict(experiment.predictions[index])
            mapped, calibration = fit_logistic5(
                arrays['target'], arrays['prediction_raw'], projected_only=True,
            )
            if mapped is None:
                raise ValueError(f'Logistic repair failed for round {index}: {calibration}')
            arrays['prediction_logistic'] = mapped
            report = dict(reports[index])
            report['calibration'] = calibration
            report['logistic_mapped'] = quality_metrics(arrays['target'], mapped)
            reports[index] = report

            round_path = stage / f'round_{index:04d}'
            _write_json(round_path / 'metrics.json', report)
            np.savez_compressed(round_path / 'predictions.npz', **arrays)
            with (round_path / 'predictions.csv').open('w', encoding='utf-8',
                                                        newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['name', 'mos', 'prediction_raw', 'prediction_logistic'])
                writer.writerows(zip(arrays['test_names'], arrays['target'],
                                     arrays['prediction_raw'], mapped, strict=True))

        summary = dict(original_summary)
        summary['metrics'] = summarize(reports)
        _write_json(stage / 'summary.json', summary)
        provenance = {
            'source_run': str(source),
            'source_summary_sha256': hashlib.sha256(
                (source / 'summary.json').read_bytes(),
            ).hexdigest(),
            'repair_method': 'projected two-dimensional search for failed '
                             'five-parameter logistic fits; test MOS used '
                             'only for post-hoc reporting',
            'repaired_rounds': failed_indices,
            'metrics_source_sha256': hashlib.sha256(
                Path(__file__).with_name('metrics.py').read_bytes(),
            ).hexdigest(),
            'repair_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        _write_json(stage / 'recalibration.json', provenance)

        verified = load_experiment(stage, allow_debug=allow_debug)
        if (verified.config != experiment.config
                or summary['metrics']['raw'] != original_summary['metrics']['raw']
                or any(report['calibration']['status'] != 'ok'
                       for report in verified.reports)):
            raise ValueError('Recalibrated run failed consistency checks')
        if output.exists():
            raise ValueError(f'Output appeared during recalibration: {output}')
        stage.rename(output)
    return {'source': str(source), 'output': str(output),
            'repaired_rounds': failed_indices,
            'valid_mapped_rounds': len(reports)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-debug', action='store_true')
    args = parser.parse_args()
    result = recalibrate_run(args.run, args.output, allow_debug=args.allow_debug)
    print(json.dumps({'output': result['output'],
                      'repaired_count': len(result['repaired_rounds']),
                      'valid_mapped_rounds': result['valid_mapped_rounds']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
