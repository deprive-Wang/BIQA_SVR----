import json

import numpy as np
import pytest

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
