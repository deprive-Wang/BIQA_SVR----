"""核验低层缓存，提取语义特征并保存 1007 维拼接特征。"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch

from livec import DEFAULT_ROOT, LivecSample, load_livec
from low_level import FEATURE_NAMES, LowLevelConfig
from semantic import DEFAULT_WEIGHTS, SEMANTIC_NAMES, SemanticExtractor

# 缓存检查，确定当前低级特征与高级特征是否一致
def read_low_level_cache(
    path: Path, samples: Sequence[LivecSample],
) -> tuple[dict[str, np.ndarray], dict]:
    """拒绝过期、顺序错乱、不完整或配置不兼容的低层缓存。"""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            'features', 'names', 'mos', 'stddev', 'annotation_indices',
            'image_sha256', 'feature_names', 'metadata',
        }
        if not required.issubset(archive.files):
            raise ValueError(f'Missing cache fields: {required - set(archive.files)}')
        data = {name: archive[name].copy() for name in required}
    count = len(samples)
    expected = {
        'names': np.array([sample.name for sample in samples]),
        'mos': np.array([sample.mos for sample in samples]),
        'stddev': np.array([sample.stddev for sample in samples]),
        'annotation_indices': np.array([sample.annotation_index for sample in samples]),
        'feature_names': np.array(FEATURE_NAMES),
    }
    # 相同维度不代表同一批图：必须逐项验证图像 ID、标注及特征列的顺序。
    for key, values in expected.items():
        if not np.array_equal(data[key], values):
            raise ValueError(f'Low-level cache {key} does not match dataset/order')
    features = data['features']
    if (features.shape != (count, 7) or features.dtype.kind not in 'fiu'
            or not np.isfinite(features).all()):
        raise ValueError('Low-level cache must contain finite (N,7) features')
    hashes = data['image_sha256']
    if hashes.shape != (count,) or hashes.dtype.kind != 'U':
        raise ValueError('Invalid image SHA-256 vector')
    if any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
           for value in hashes):
        raise ValueError('Invalid image SHA-256 values')
    if data['metadata'].shape != () or data['metadata'].dtype.kind != 'U':
        raise ValueError('Cache metadata must be a JSON string scalar')
    metadata = json.loads(str(data['metadata']))
    if not isinstance(metadata, dict):
        raise ValueError('Cache metadata must be a JSON object')
    if metadata.get('config') != LowLevelConfig().metadata():
        raise ValueError('Low-level configuration differs from the current extractor')
    if metadata.get('sample_count') != count or metadata.get('debug_subset') != (count != 1162):
        raise ValueError('Low-level cache sample count/subset metadata is inconsistent')
    code_root = Path(__file__).resolve().parent
    # 全文哈希包含注释；源码变动后拒绝把旧缓存冒充为当前源码生成的结果。
    expected_sources = {
        name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
        for name in ('low_level.py', 'livec.py', 'extract_features.py')
    }
    if metadata.get('source_sha256') != expected_sources:
        raise ValueError('Low-level cache source hashes differ; regenerate the cache')
    return data, metadata

# 用同一批图像提取 1000 维语义特征，与已有的 7 维低层特征按行拼接，保存成 (N,1007) 的缓存
def run_extraction(
    root: Path, low_level_path: Path, output: Path, *, weights: Path = DEFAULT_WEIGHTS,
    batch_size: int = 16, device: str = 'cuda', limit: int | None = None,
) -> None:
    """核对图像 ID 与内容后提取并拼接特征，不覆盖已有结果。"""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1162):
        raise ValueError('limit must be an integer in [1,1162]')
    if output.suffix != '.npz':
        raise ValueError('Output must end in .npz')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Output must be outside the raw dataset directory')

    # 取得图像列表，核对底层特征缓存
    samples = load_livec(root, verify_images=False)
    # 先检查完整低层缓存，再截取调试子集，防止用不完整缓存替代正式数据。
    low, low_metadata = read_low_level_cache(low_level_path, samples)
    if limit is not None:
        samples = samples[:limit]

    # 使用预训练模型提取高维语义特征，确保图像内容与低层缓存相对应
    extractor = SemanticExtractor(weights, device=device)
    extractor_metadata = extractor.metadata()
    print(f"Device: {device}; {extractor_metadata.get('device_name', device)}; "
          f'batch size: {batch_size}', flush=True)
    high_batches = []
    for start in range(0, len(samples), batch_size):
        images = []
        for index in range(start, min(start + batch_size, len(samples))):
            sample = samples[index]
            try:
                with sample.path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if digest != low['image_sha256'][index]:
                        raise ValueError('Image content differs from the low-level cache')
                    stream.seek(0)
                    with Image.open(stream) as image:
                        if image.mode != 'RGB':
                            raise ValueError(f'Expected RGB; received {image.mode}')
                        images.append(np.array(image))
            except (OSError, ValueError) as error:
                raise ValueError(f'{sample.name}: {error}') from error
        high_batches.append(extractor.extract(images))
        print(f'Extracted semantics {min(start + batch_size, len(samples))}/{len(samples)}',
              flush=True)
    # 各批 (B,1000) 沿样本轴合并为 (N,1000)，保留最后一个不足 B 的批次。
    high = np.concatenate(high_batches)
    count = len(samples)
    # 论文 II-A / 表 I：沿特征轴拼接 (N,7)+(N,1000) -> (N,1007)。
    # 每一行对应一张图；前 7 列是低层属性，后 1000 列是高层语义。
    features = np.concatenate((low['features'][:count], high), axis=1)

    # 记
    metadata = {
        'sample_count': count, 'debug_subset': count != 1162,
        'low_level': low_metadata, 'semantic': extractor_metadata,
        'batch_size': batch_size,
        'low_level_cache_sha256': hashlib.sha256(low_level_path.read_bytes()).hexdigest(),
        'source_sha256': {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('semantic.py', 'extract_semantic.py')
        },
        'feature_layout': {'low_level': [0, 7], 'semantic': [7, 1007]},
        'torch_num_threads': torch.get_num_threads(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        try:
            np.savez_compressed(
                stream, features=features,
                **{key: low[key][:count] for key in (
                    'names', 'mos', 'stddev', 'annotation_indices', 'image_sha256',
                )},
                feature_names=np.array(FEATURE_NAMES + SEMANTIC_NAMES),
                metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
            )
        except BaseException:
            stream.close()
            output.unlink()
            raise
    print(f'Saved {count} x 1007 features: {output}')


def main() -> None:
    """使用本地已校验的预训练权重提取池化后的激活特征。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--low-level', type=Path,
                        default=Path(__file__).resolve().parent / 'features/livec_low_level_v2.npz')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda',
                        help='CUDA by default; CPU requires explicit selection')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        run_extraction(args.root, args.low_level, args.output, weights=args.weights,
                       batch_size=args.batch_size, device=args.device, limit=args.limit)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Semantic extraction failed: {error}\n')


if __name__ == '__main__':
    main()
