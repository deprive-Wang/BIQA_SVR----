"""Extract BCQI low-level features in LIVEC annotation order."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import pywt
import scipy

from livec import DEFAULT_ROOT, load_livec
from low_level import FEATURE_NAMES, LowLevelConfig, extract_low_level


# 逐张读取 LIVEC 图像，算出每张图的 7 维低层特征，再连同图像名、MOS 等信息保存为一个 .npz 文件
def main() -> None:
    """Write an auditable NPZ cache; refuse to overwrite an existing result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Debug subset only; omitted means all images')
    args = parser.parse_args()

    # 数据校对，若异常则抛出

    if args.limit is not None and not 1 <= args.limit <= 1162:
        parser.error('--limit must be between 1 and 1162')
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error('Output must be outside the raw dataset directory')

    # 只解码一次，下面提取特征

    samples = load_livec(args.root, verify_images=False)
    samples = samples[:args.limit] if args.limit else samples
    config = LowLevelConfig()

    # 按 MAT 标注顺序逐图提取，features[i]、MOS[i]、图像哈希始终描述同一张图。
    features, hashes = [], []
    for index, sample in enumerate(samples, 1):
        try:
            with sample.path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                stream.seek(0)
                with Image.open(stream) as image:
                    if image.mode != 'RGB':
                        raise ValueError(f'Expected RGB, received {image.mode}')
                    rgb = np.array(image)
            features.append(extract_low_level(rgb, config))
            hashes.append(digest)
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f'Feature extraction failed for {sample.name}: {error}\n')
        if index == 1 or index % 25 == 0 or index == len(samples):
            print(f'Extracted {index}/{len(samples)}', flush=True)
    code_root = Path(__file__).resolve().parent


    # 以下是复现追溯信息，不属于论文特征。源码按完整字节计算哈希：改注释也会变化。
    metadata = {
        'config': config.metadata(),
        'dataset_root': str(args.root.resolve()),
        'sample_count': len(samples),
        'debug_subset': len(samples) != 1162,
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__,
                     'PyWavelets': pywt.__version__, 'Pillow': PIL.__version__},
        'source_sha256': {
            name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
            for name in ('low_level.py', 'livec.py', 'extract_features.py')
        },
    }

    # 写入npz文件
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing runs, including concurrent writers.
    with args.output.open('xb') as stream:
        try:
            # N 个 (7,) 堆叠为 (N,7)；此时不拟合标准化，避免利用后续测试集统计量。
            np.savez_compressed(
                stream, features=np.stack(features),
                names=np.array([sample.name for sample in samples]),
                mos=np.array([sample.mos for sample in samples]),
                stddev=np.array([sample.stddev for sample in samples]),
                annotation_indices=np.array([sample.annotation_index for sample in samples]),
                image_sha256=np.array(hashes), feature_names=np.array(FEATURE_NAMES),
                metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
            )
        except BaseException:
            stream.close()
            args.output.unlink()
            raise
    print(f'Saved {len(samples)} x {len(FEATURE_NAMES)} features: {args.output}')


if __name__ == '__main__':
    main()
