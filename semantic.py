"""BCQI-style 1000-D SqueezeNet features using torchvision ImageNet weights."""

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1


DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent / 'checkpoints' / 'squeezenet1_1-b8a52dc0.pth'
)
WEIGHTS_URL = SqueezeNet1_1_Weights.IMAGENET1K_V1.url
WEIGHTS_SHA256 = 'b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b'
CROP_SIZE = 227
SEMANTIC_NAMES = tuple(f'semantic_{index:04d}' for index in range(1000))
# Built once at import; the per-image values are fixed by the chosen weights.
WEIGHT_TRANSFORM = SqueezeNet1_1_Weights.IMAGENET1K_V1.transforms()
NORMALIZE_MEAN = torch.tensor(WEIGHT_TRANSFORM.mean)[:, None, None]
NORMALIZE_STD = torch.tensor(WEIGHT_TRANSFORM.std)[:, None, None]


def prepare_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Center-crop RGB uint8 to 227 square and apply weight-specific normalization.

    Odd excess pixels are dropped from the bottom/right (floor crop origin).
    No resize, padding, EXIF orientation or ICC conversion is performed.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError('Expected a uint8 RGB numpy array')
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected image shape (H, W, 3)')
    height, width = rgb.shape[:2]
    if min(height, width) < CROP_SIZE:
        raise ValueError('Image must be at least 227 x 227; resizing is disabled')
    # 论文 II-C、式 (17)、图 5：直接取原图中心 227×227，保留原始失真尺度。
    top, left = (height - CROP_SIZE) // 2, (width - CROP_SIZE) // 2
    crop = rgb[top:top + CROP_SIZE, left:left + CROP_SIZE]
    # (227,227,3) -> (3,227,227)，转换为 PyTorch 的通道优先格式。
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255
    # mean/std 按通道广播；这是 torchvision 权重的预处理，未核验与作者权重等价。
    return (tensor - NORMALIZE_MEAN) / NORMALIZE_STD


def _load_verified_model(weights_path: Path) -> nn.Module:
    # Verify bytes before deserializing; never silently fall back to random weights.
    with weights_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != WEIGHTS_SHA256:
            raise ValueError(f'Weight SHA-256 mismatch: {weights_path}')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
    model = squeezenet1_1(weights=None)
    model.load_state_dict(state, strict=True)
    return model


class SemanticExtractor:
    """Frozen SqueezeNet v1.1; returns pooled activations, without softmax."""

    def __init__(
        self, weights_path: str | Path = DEFAULT_WEIGHTS, *, device: str = 'cuda',
    ) -> None:
        if device not in {'cpu', 'cuda'}:
            raise ValueError('device must be cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError(
                'CUDA is unavailable. Install CUDA-enabled torch/torchvision in '
                'BIQA_SVR and verify torch.cuda.is_available(); no CPU fallback is used.'
            )
        self.device = torch.device(device)
        self.weights_path = Path(weights_path).resolve()
        self.model = _load_verified_model(self.weights_path).to(self.device)
        # 网络只作固定的特征提取器：关闭 Dropout 并冻结参数，不用 MOS 微调网络。
        self.model.eval()
        self.model.requires_grad_(False)

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return float32 (B,1000) features for a nonempty batch of RGB arrays."""
        if len(images) == 0:
            raise ValueError('Image batch must not be empty')
        # 新增批维，B 张图组成 (B,3,227,227)，不要求这些原图的尺寸相同。
        batch = torch.stack([prepare_rgb(image) for image in images])
        if self.device.type == 'cuda':
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)
        # Ampere TF32 can choose different reduced-precision kernels for
        # different batch sizes. Keep feature extraction in IEEE float32.
        with torch.inference_mode(), torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False,
        ):
            # torchvision's classifier ends in ReLU + AdaptiveAvgPool2d,
            # and forward flattens that output; there is no softmax layer.
            # 图 5 的全局平均池化：每个通道的空间响应压成一个数，得到 (B,1000)。
            # 这 1000 维是激活特征，不是类别编号，也不是 softmax 分类概率。
            features = self.model(batch)
        result = features.cpu().numpy().copy()
        if result.shape != (len(images), 1000) or not np.isfinite(result).all():
            raise RuntimeError('Expected finite SqueezeNet features of shape (B,1000)')
        return result

    def metadata(self) -> dict:
        """Document the reproducible Python variant and its MATLAB boundary."""
        return {
            'version': 'bcqi-semantic-torchvision-v1',
            'architecture': 'squeezenet1_1',
            'weights': 'SqueezeNet1_1_Weights.IMAGENET1K_V1',
            'weights_url': WEIGHTS_URL,
            'weights_sha256': WEIGHTS_SHA256,
            'crop_size': CROP_SIZE,
            'crop_origin': 'floor((size-227)/2)',
            'resize': False,
            'input': 'RGB uint8 / 255; no EXIF transpose or ICC conversion',
            'mean': WEIGHT_TRANSFORM.mean, 'std': WEIGHT_TRANSFORM.std,
            'output': 'classifier.3 global average pool; flattened; no softmax',
            'dimensions': 1000, 'eval_mode': True, 'gradients': False,
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'device': str(self.device),
            'cuda_runtime': torch.version.cuda,
            'device_name': (torch.cuda.get_device_name(self.device)
                            if self.device.type == 'cuda' else 'CPU'),
            'dtype': 'float32; no autocast',
            'cudnn': {'benchmark': False, 'deterministic': True, 'allow_tf32': False},
            'matlab_equivalence': 'Unverified weights and preprocessing equivalence',
        }
