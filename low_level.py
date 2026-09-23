"""实现 BCQI 式 (3)～(16)：六类低层属性，共七维特征。

这是 Python 复现，并非作者源码。论文未明确的数值选择记录在
LowLevelConfig 和 README.md 中。
"""

from dataclasses import asdict, dataclass

import numpy as np
import pywt # 用于小波变换
from scipy.ndimage import correlate1d, gaussian_filter # 用于计算对比度和锐度
from scipy.optimize import minimize_scalar # 用于优化 alpha、beta
from scipy.special import gammaln, rel_entr # 用于计算自然度的 alpha、beta 分布


# 论文 II-B / 表 I：六类属性、七个数值；自然度由 alpha、beta 两维表示。
# 列顺序是缓存与后续特征拼接的约定，不能单独调整。
FEATURE_NAMES = (
    'brightness', 'saturation', 'contrast_js', 'noise_variance',
    'sharpness', 'naturalness_alpha', 'naturalness_beta',
)
EXTRACTOR_VERSION = 'bcqi-low-level-v1'

# 定义低级特征的参数并进行校对和说明
@dataclass(frozen=True)
class LowLevelConfig:
    """本项目的默认配置，并非已核验的作者实现参数。"""

    # 以下是本项目的数值选择，不能视为已经核验的作者代码参数。
    dct_size: int = 7
    gaussian_sigma: float = 7 / 6
    gaussian_radius: int = 3
    alpha_min: float = 0.2
    alpha_max: float = 10.0
    noise_grid_size: int = 129

    def __post_init__(self) -> None:
        for name in ('dct_size', 'gaussian_radius', 'noise_grid_size'):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.dct_size < 3 or self.dct_size % 2 != 1:
            raise ValueError('dct_size must be odd and at least 3')
        if self.gaussian_radius < 1 or self.noise_grid_size < 5:
            raise ValueError('Invalid Gaussian radius or noise grid size')
        if not np.isfinite([self.gaussian_sigma, self.alpha_min, self.alpha_max]).all():
            raise ValueError('Configuration values must be finite')
        if self.gaussian_sigma <= 0 or not 0 < self.alpha_min < self.alpha_max:
            raise ValueError('Invalid Gaussian sigma or GGD shape bounds')

    def metadata(self) -> dict:
        """返回识别特征缓存所需的数值计算约定。"""
        return {
            **asdict(self), 'version': EXTRACTOR_VERSION,
            'input': 'RGB uint8; no resize, EXIF transpose or ICC conversion',
            'brightness_scale': '[0,1]',
            'gray': 'float64 BT.601: 0.299R+0.587G+0.114B, [0,255]',
            'histogram': '256 bins; floor(gray+0.5); natural logarithm',
            'wavelet': 'bior4.4, symmetric, 3 levels',
            'dct': 'orthonormal DCT-II, all AC filters, valid support',
            'moments': 'population centered variance and Pearson kurtosis',
            'noise_fit': 'Eq8 L1; profile weighted median; grid+bounded minima',
            'noise_bounds': '0 <= variance < min AC variance; kurtosis >= 1',
            'mscn': 'Gaussian reflect padding; denominator sigma+1',
            'ggd': 'zero-mean raw moments; beta is scale, not variance',
            'degenerate': 'constant noise=0; zero MSCN alpha=2,beta=0',
        }

# 用 48 个不同的 DCT 滤波器扫描灰度图，分别统计每种滤波响应的方差和峰度，供后续噪声估计使用
# 灰度图 → 滤波响应 → 方差与峰度
def _dct_moments(gray: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    # 二维基可分离为两次一维相关；size=7 时共有 49 个基，去掉 DC 后剩 48 个。
    
    # 根据DCT基的定义来计算
    positions = np.arange(size) + 0.5
    basis = np.sqrt(2 / size) * np.cos(
        np.pi * np.arange(size)[:, None] * positions / size
    )
    basis[0] /= np.sqrt(2)

    '''
    gray (H, W)
    └─ 沿 axis=0（高度方向）滤波 → vertical (H, W)
       └─ 沿 axis=1（宽度方向）滤波 → response (H, W)
    '''
    margin = size // 2
    variances, kurtoses = [], []
    for row in range(size):
        vertical = correlate1d(gray, basis[row], axis=0)
        for column in range(size):
            if row == column == 0:
                continue
            response = correlate1d(vertical, basis[column], axis=1)
            # 去掉依赖边界延拓的位置，只统计完整滤波窗口内的 valid 响应。
            response = response[margin:-margin, margin:-margin]

            # 计算方差
            centered = response - response.mean()
            squared = centered * centered
            variance = float(squared.mean())
            variances.append(variance)
            # Pearson 峰度为四阶中心矩 / 方差平方，高斯分布对应 3 而不是 0。
            
            #计算峰度
            kurtoses.append(float((squared * squared).mean() / variance**2)
                            if variance > 1e-20 else 3.0)
            
    # 返回48个不同滤波响应的方差和峰度，下一步交给fit来计算
    return np.array(variances), np.array(kurtoses)

# 输入48个滤波响应的方差和峰度，输出噪声方差v，v为局部最优解，数学公式对应论文
'''
代码假设这 48 种响应对应的“干净图像部分”具有共同峰度，然后寻找最能解释观测数据的噪声方差。
'''
def _fit_noise(variances: np.ndarray, kurtoses: np.ndarray, grid_size: int) -> float:
    # 论文式 (7)-(8)：假设干净图像的各滤波响应具有共同峰度，反推噪声方差 v。
    # 令 v < min(variances)，保证估计的干净响应方差 variance_i-v 为正。
    # 式 (8) 为 (kappa_x-3)*(1-v/variance_i)^2 + 3 - kappa_i。
    # 固定 v 后，目标变为单参数加权绝对误差回归；其精确最优解是加权中位数，
    # 再将共同峰度约束为 kappa_x >= 1。
    upper = float(variances.min())
    if upper <= 1e-20:
        return 0.0

    # 给一个候选噪声，算预测峰度与观测峰度有多少误差，即寻找局部最优解
    def objective(fraction: float) -> float:
        # fraction=v/upper；固定 v 后，L1 目标对共同超额峰度的最优解是加权中位数。
        weights = (1 - fraction * upper / variances)**2
        ratios = (kurtoses - 3) / weights
        order = np.argsort(ratios)
        cumulative = np.cumsum(weights[order])
        index = np.searchsorted(cumulative, cumulative[-1] / 2)
        excess = max(float(ratios[order[index]]), -2.0)  # 峰度 >= 1，即超额峰度 >= -2。
        return float(np.abs(excess * weights + 3 - kurtoses).sum())

    # 网格先寻找候选谷底，再局部细化；这是项目求解策略，不保证数学全局最优。
    grid = np.linspace(0, 1 - 1e-6, grid_size)
    costs = np.array([objective(value) for value in grid])
    candidates = [(float(costs[index]), float(grid[index]))
                  for index in (0, int(costs.argmin()), len(grid) - 1)]
    for index in range(1, len(grid) - 1):
        if costs[index] <= costs[index - 1] and costs[index] <= costs[index + 1]:
            result = minimize_scalar(
                objective, bounds=(grid[index - 1], grid[index + 1]),
                method='bounded', options={'xatol': 1e-9},
            )
            if not result.success:
                raise RuntimeError(f'Noise optimization failed: {result.message}')
            candidates.append((float(result.fun), float(result.x)))
    return min(candidates)[1] * upper  # 返回方差 sigma_n^2，不取平方根。


'''
MSCN 数组 (H, W)
→ 求 E[|X|] 和 E[X²]
→ 用两者的比值拟合 alpha
→ 用 E[X²] 和 alpha 算 beta
→ 返回 (alpha, beta)
'''
# 讲MSCN系数拟合为广义高斯分布（GGD），返回两个数值：alpha描述MSCN数值分布的形状，beta描述MSCN数值分布的尺度
def _ggd_parameters(values: np.ndarray, config: LowLevelConfig) -> tuple[float, float]:
    # 比值 (E[|X|])^2 / E[X^2] 消去 beta，只与 alpha 有关。
    second = float(np.mean(values**2))
    if second <= 1e-20:
        return 2.0, 0.0
    ratio = float(np.mean(np.abs(values)))**2 / second

    # 计算 alpha 的局部最优解，beta 由 alpha 和 E[X^2] 反推。
    def objective(alpha: float) -> float:
        predicted = np.exp(2 * gammaln(2 / alpha)
                           - gammaln(1 / alpha) - gammaln(3 / alpha))
        return float((predicted - ratio)**2)

    result = minimize_scalar(objective, bounds=(config.alpha_min, config.alpha_max),
                             method='bounded', options={'xatol': 1e-10})
    if not result.success:
        raise RuntimeError(f'GGD optimization failed: {result.message}')
    alpha = min((config.alpha_min, float(result.x), config.alpha_max), key=objective)
    # E[X^2] = beta^2 * Gamma(3/alpha)/Gamma(1/alpha)；用对数 Gamma 避免溢出。
    beta = np.sqrt(second * np.exp(gammaln(1 / alpha) - gammaln(3 / alpha)))
    return alpha, float(beta)

# 输入RGB图像，输出7维低层特征，综合使用上述函数，提供统一入口
def extract_low_level(
    rgb: np.ndarray, config: LowLevelConfig = LowLevelConfig(),
) -> np.ndarray:
    """将形状 (H,W,3) 的 RGB uint8 图像转为形状 (7,) 的 float64 特征。

    最小尺寸保证三级 bior4.4 分解不会让所有系数都受边界影响。
    不缩放图像，也不应用从数据中学习的标准化参数。
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError('Input must be a uint8 RGB numpy array')
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Input shape must be (H, W, 3)')
    if min(rgb.shape[:2]) < max(72, config.dct_size + 1):
        raise ValueError('Image is too small for three wavelet levels and DCT')
    # 低层分支使用原图 (H,W,3)，先转浮点防止 uint8 运算溢出。
    # 式 (3)-(4)：只计算亮度和饱和度的均值，输出两个标量。
    pixels = rgb.astype(np.float64)
    total = pixels.sum(axis=2)  # 沿 RGB 通道求和，(H,W,3) -> (H,W)。
    intensity = total / (3 * 255)

    # 饱和度部分
    ratio = np.ones_like(total)
    np.divide(3 * pixels.min(axis=2), total, out=ratio, where=total > 0)
    saturation = 1 - ratio

    # 灰度部分计算对比度
    gray = pixels @ np.array([0.299, 0.587, 0.114])

    # 式 (5)-(6)：灰度概率直方图与均匀分布的 J-S 散度，输出一个标量。
    # 它描述分布偏离程度，不是最终质量分数；此处不取散度的平方根。
    histogram = np.bincount(np.floor(gray + 0.5).astype(np.int64).ravel(),
                            minlength=256).astype(np.float64)
    histogram /= histogram.sum()

    # 噪声特征计算
    uniform = np.full(256, 1 / 256)
    mixture = (histogram + uniform) / 2
    contrast = float((rel_entr(histogram, mixture).sum()
                      + rel_entr(uniform, mixture).sum()) / 2)
    variances, kurtoses = _dct_moments(gray, config.dct_size)
    noise = _fit_noise(variances, kurtoses, config.noise_grid_size)

    # 锐度特征计算
    coefficients = pywt.wavedec2(gray, 'bior4.4', mode='symmetric', level=3)
    sharpness = 0.0
    # PyWavelets 按粗到细返回三级细节，故权重是 1/7、2/7、4/7；不使用低频近似。
    for weight, bands in zip((1 / 7, 2 / 7, 4 / 7), coefficients[1:]):
        energies = [np.log10(1 + np.mean(band**2)) for band in bands]
        sharpness += weight * np.dot((0.1, 0.1, 0.8), energies)

    
    # 计算自然度的两个参数，alpha和beta
    def smooth(values: np.ndarray) -> np.ndarray:
        return gaussian_filter(values, sigma=config.gaussian_sigma,
                               radius=config.gaussian_radius, mode='reflect')

    # 式 (12)-(14)：MSCN=(灰度-局部均值)/(局部标准差+1)，仍为 (H,W)。
    # E[X^2]-E[X]^2 计算局部方差；截断浮点误差导致的微小负数，避免 sqrt 出 NaN。
    # 计算MSCN系数,'减去局部均值，再按局部对比度归一化'
    mean = smooth(gray)
    deviation = np.sqrt(np.maximum(smooth(gray**2) - mean**2, 0))
    mscn = (gray - mean) / (deviation + 1)
    alpha, beta = _ggd_parameters(mscn, config)
    # 拼接7个参数
    features = np.array([intensity.mean(), saturation.mean(), contrast, noise,
                         sharpness, alpha, beta], dtype=np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError('Non-finite low-level features')
    return features
# features顺序：[平均亮度, 平均饱和度, 对比度, 噪声, 锐度, alpha, beta]
