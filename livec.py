"""Read and validate the official LIVE Challenge release without changing it."""

import argparse
import json
from collections import Counter   #统计元素出现的次数
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat      #读mat文件



# 主要用于预先核查LIVEC数据集的完整性和正确性，确保图像文件、MOS评分和标准差等信息与官方发布一致。

DEFAULT_ROOT = Path(__file__).resolve().parent / 'data' / 'ChallengeDB_release'
# trainingImages 是主观实验的练习图，不是供 SVR 使用的训练集。
PRACTICE_NAMES = frozenset(f't{index}.bmp' for index in range(1, 8))   #frozenset() 函数创建一个不可变的集合，可哈希化
IMAGE_SUFFIXES = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}

# 定义一个数据类，用于存储每张 LIVEC 图像的样本信息。后续list[LivecSample] 用于存储所有样本。
@dataclass(frozen=True)
class LivecSample:
    """A scored image in the original MAT ordering."""

    name: str
    path: Path
    mos: float  # 人类主观评分均值，作为 SVR 的回归目标 y。
    stddev: float  # 主观评分的分散程度；当前训练不把它作为特征或权重。
    annotation_index: int  # 保留原始 1169 项标注中的位置，便于追溯对齐。

# 定义最小load函数，用于读取 LIVEC 数据集中的向量变量。
def _read_vector(root: Path, variable: str) -> np.ndarray:
    
    path = root / 'Data' / f'{variable}.mat'
    try:
        content = loadmat(path)
    except Exception as error:
        raise ValueError(f'Cannot read MAT file {path}: {error}') from error
    if variable not in content:
        raise ValueError(f'{path} does not contain {variable}')
    values = content[variable]
    if values.ndim != 2 or 1 not in values.shape:
        raise ValueError(f'{variable} must be a row or column vector')
    # MAT 可能保存为行向量或列向量；展平不改变标注的先后次序。
    return values.reshape(-1)


def load_livec(
    root: str | Path = DEFAULT_ROOT,
    *,
    verify_images: bool = True,
) -> tuple[LivecSample, ...]:
    """Load 1162 scored images, excluding the seven observer practice images.

    Strictly checks release size, names, scores and file coverage. By default,
    fully decodes every retained image; turning that off only skips decoding.
    """
    root = Path(root).resolve()
    # 三个向量按同一索引对应，不能分别对文件名或 MOS 排序。
    raw_names = _read_vector(root, 'AllImages_release')
    mos = _read_vector(root, 'AllMOS_release')
    stddev = _read_vector(root, 'AllStdDev_release')

    # 检查标注是否完整，若不完整则抛出异常。

    if not (len(raw_names) == len(mos) == len(stddev) == 1169):
        raise ValueError('Expected 1169 aligned names, MOS values and deviations')
    names = []
    for entry in raw_names:
        if not isinstance(entry, np.ndarray) or entry.size != 1:
            raise ValueError('Each image-name cell must contain one string')
        value = entry.item()
        if not isinstance(value, str) or not value:
            raise ValueError('Image names must be nonempty strings')
        # Release names are basenames; reject paths before joining the root.
        if any(character in value for character in '/\\:') or value in {'.', '..'}:
            raise ValueError(f'Invalid image basename: {value!r}')
        names.append(value)
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError('Duplicate image names in annotations')
    if not PRACTICE_NAMES.issubset(names):
        raise ValueError('Expected all seven named observer practice images')
    for label, scores in [('MOS', mos), ('standard deviation', stddev)]:
        if scores.dtype.kind not in 'fiu' or not np.isfinite(scores).all():
            raise ValueError(f'{label} must contain finite real numbers')
    if ((mos < 0) | (mos > 100)).any():
        raise ValueError('MOS must be within the LIVE Challenge scale [0, 100]')
    if (stddev < 0).any():
        raise ValueError('Standard deviations must be nonnegative')

    # 检查图像文件是否完整，若不完整则抛出异常。

    image_root = root / 'Images'
    if not image_root.is_dir():
        raise ValueError(f'Missing image directory: {image_root}')
    expected = set(names) - PRACTICE_NAMES
    actual = {
        path.name for path in image_root.iterdir()
        if path.is_file() and not path.name.startswith('.')
        and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if actual != expected:
        raise ValueError(
            f'Image coverage mismatch: missing={sorted(expected - actual)}, '
            f'extra={sorted(actual - expected)}'
        )

    # 同一次循环过滤名称和标签：1169 项减去 7 张练习图，得到 1162 项。
    # 这里尚未划分训练/测试集；划分由 train_svr.py 在实验阶段完成。
    samples = []
    for index, name in enumerate(names):
        if name in PRACTICE_NAMES:
            continue
        path = image_root / name
        if verify_images:
            try:
                with Image.open(path) as image:
                    image.load()
            except Exception as error:
                raise ValueError(f'Cannot decode image {path}: {error}') from error
        samples.append(LivecSample(name, path, float(mos[index]),
                                  float(stddev[index]), index))
    return tuple(samples)


def main() -> None:
    """Validate a release and print a compact JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    try:
        samples = load_livec(args.root)
    except ValueError as error:
        parser.exit(1, f'Dataset validation failed: {error}\n')
    report = {
        'root': str(args.root.resolve()),
        'samples': len(samples),
        'excluded_practice_images': len(PRACTICE_NAMES),
        'mos_min': min(sample.mos for sample in samples),
        'mos_max': max(sample.mos for sample in samples),
        'extensions': dict(Counter(sample.path.suffix.lower() for sample in samples)),
        'first_sample': {
            'name': samples[0].name,
            'mos': samples[0].mos,
            'annotation_index': samples[0].annotation_index,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
