import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import compare_baselines
from compare_baselines import _holm_adjust, _load_baseline, paired_residual_test


def test_paired_test_uses_same_images_and_error_direction() -> None:
    target = np.arange(20, dtype=float)
    bcqi = target + np.linspace(0.1, 0.5, len(target))
    baseline = target + np.linspace(1.0, 3.0, len(target))

    result = paired_residual_test(target, bcqi, baseline)

    assert result['n_paired_images'] == 20
    assert result['mean_baseline_minus_bcqi'] > 0
    assert result['p_two_sided'] < 0.05


def test_holm_adjusts_within_one_split() -> None:
    tests = {
        'one': {'p_two_sided': 0.01, 'mean_baseline_minus_bcqi': 1.0},
        'two': {'p_two_sided': 0.04, 'mean_baseline_minus_bcqi': -1.0},
    }

    _holm_adjust(tests)

    assert tests['one']['p_holm'] == pytest.approx(0.02)
    assert tests['one']['decision'] == 'bcqi_superior'
    assert tests['two']['p_holm'] == pytest.approx(0.04)
    assert tests['two']['decision'] == 'bcqi_inferior'


def test_baseline_rejects_misaligned_image_hashes(tmp_path) -> None:
    path = tmp_path / 'baseline.npz'
    names = ['a.jpg', 'b.jpg']
    mos = np.array([10.0, 20.0])
    hashes = np.array(['0' * 64, '1' * 64])
    metadata = {'method': 'BRISQUE', 'debug_subset': False,
                'source_cache_sha256': '2' * 64}
    np.savez(path, names=names, mos=mos, scores=np.array([1.0, 2.0]),
             image_sha256=hashes,
             metadata=np.array(json.dumps(metadata)))

    with pytest.raises(ValueError, match='image contents differ'):
        _load_baseline(path, names, mos, hashes[::-1], '2' * 64)


def test_niqe_archive_preserves_alignment_and_score_direction(
        tmp_path, monkeypatch) -> None:
    root = tmp_path / 'data'
    images = root / 'Images'
    images.mkdir(parents=True)
    pixels = np.full((192, 192, 3), [60, 100, 140], dtype=np.uint8)
    image_path = images / 'a.png'
    Image.fromarray(pixels, 'RGB').save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    cache = tmp_path / 'source.npz'
    cache.write_bytes(b'fixture')
    monkeypatch.setattr(compare_baselines, 'load_training_cache',
                        lambda *_: {'names': np.array(['a.png']),
                                    'mos': np.array([75.0]),
                                    'image_sha256': np.array([digest])})
    observed = []
    module = ModuleType('skvideo.measure')

    def fake_niqe(gray):
        observed.append(gray)
        return np.array([5.5])

    module.niqe = fake_niqe
    package = ModuleType('skvideo')
    package.__path__ = []
    monkeypatch.setitem(sys.modules, 'skvideo', package)
    monkeypatch.setitem(sys.modules, 'skvideo.measure', module)
    monkeypatch.setattr(compare_baselines, 'version', lambda _: '1.3.0')

    output = tmp_path / 'niqe.npz'
    compare_baselines.extract_niqe(cache, root, output, limit=1)

    with np.load(output, allow_pickle=False) as archive:
        assert archive['names'].tolist() == ['a.png']
        assert archive['mos'].tolist() == [75.0]
        assert archive['source_scores'].tolist() == [5.5]
        assert archive['scores'].tolist() == [-5.5]
        assert archive['image_sha256'].tolist() == [digest]
        metadata = json.loads(str(archive['metadata']))
    assert observed[0].shape == (192, 192)
    assert observed[0].dtype == np.uint8
    assert metadata['debug_subset'] is True
    assert metadata['source_cache_sha256'] == hashlib.sha256(b'fixture').hexdigest()


def test_partial_comparison_requires_debug(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(compare_baselines, 'load_experiment',
                        lambda *_args, **_kwargs: SimpleNamespace(
                            reports=[{}, {}]))

    with pytest.raises(ValueError, match='requires --allow-debug'):
        compare_baselines.compare(tmp_path, [tmp_path / 'baseline.npz'],
                                  tmp_path / 'report.json', max_rounds=1)
