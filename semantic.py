"""使用 torchvision ImageNet 权重提取 BCQI 风格的 1000 维 SqueezeNet 特征。"""

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
# 导入模块时只构建一次；每张图的预处理参数由所选权重确定。
WEIGHT_TRANSFORM = SqueezeNet1_1_Weights.IMAGENET1K_V1.transforms()
NORMALIZE_MEAN = torch.tensor(WEIGHT_TRANSFORM.mean)[:, None, None]
NORMALIZE_STD = torch.tensor(WEIGHT_TRANSFORM.std)[:, None, None]

# SqueezeNet 输入前的图像预处理
'''
原图 RGB 数组 (H, W, 3)，uint8，像素值 0～255
从中心裁出 (227, 227, 3)
permute(2, 0, 1) 变成 (3, 227, 227)
转 float 并除以 255，像素值变成 0～1
每个颜色通道分别减 mean、除 std
返回 PyTorch Tensor (3, 227, 227)
'''
def prepare_rgb(rgb: np.ndarray) -> torch.Tensor:
    """将 RGB uint8 图像中心裁剪为 227×227，并按权重要求做归一化。

    多余像素数为奇数时，底部或右侧多舍去一个像素。这里不缩放、不填充，
    也不应用 EXIF 方向调整或 ICC 色彩转换。
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

# 加载并验证 SqueezeNet 模型权重
def _load_verified_model(weights_path: Path) -> nn.Module:
    # 反序列化前先核验权重文件内容，失败时不回退到随机权重。
    with weights_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != WEIGHTS_SHA256:
            raise ValueError(f'Weight SHA-256 mismatch: {weights_path}')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
    model = squeezenet1_1(weights=None)
    model.load_state_dict(state, strict=True)
    return model

# 定义一个类，封装 SqueezeNet 模型，提供特征提取和元数据查询功能
class SemanticExtractor:
    """冻结的 SqueezeNet v1.1；返回池化后的激活值，不应用 softmax。"""

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

    # 提取图像特征
    '''
    B 张 RGB 数组，原图尺寸可以不同
    → prepare_rgb()：每张变成 (3,227,227)
    → torch.stack()：合成 (B,3,227,227)
    → 移到 CPU 或 GPU
    → SqueezeNet：得到 (B,1000)35
    → 移回 CPU，转 NumPy 数组返回
    '''
    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """对非空 RGB 图像批次返回形状 (B,1000) 的 float32 特征。"""
        if len(images) == 0:
            raise ValueError('Image batch must not be empty')
        # 新增批维，B 张图组成 (B,3,227,227)，不要求这些原图的尺寸相同。
        batch = torch.stack([prepare_rgb(image) for image in images])
        if self.device.type == 'cuda':
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)
        # Ampere 的 TF32 可能随批大小选择不同的降精度卷积实现；
        # 关闭相关选项，让特征提取保持 float32 精度。
        with torch.inference_mode(), torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False,
        ):
            # torchvision 分类头末尾是 ReLU 和 AdaptiveAvgPool2d；
            # 前向传播将池化结果展平，不包含 softmax 层。
            # 图 5 的全局平均池化：每个通道的空间响应压成一个数，得到 (B,1000)。
            # 这 1000 维是激活特征，不是类别编号，也不是 softmax 分类概率。
            features = self.model(batch)
        result = features.cpu().numpy().copy()
        if result.shape != (len(images), 1000) or not np.isfinite(result).all():
            raise RuntimeError('Expected finite SqueezeNet features of shape (B,1000)')
        return result


    # 查询模型元数据，记录版本、架构、权重、预处理、输出、维度、评估模式、梯度、设备、dtype、cudnn、MATLAB 等价
    def metadata(self) -> dict:
        """记录 Python 复现配置及其与 MATLAB 实现尚未核验的边界。"""
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
