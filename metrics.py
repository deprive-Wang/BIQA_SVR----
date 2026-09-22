"""比较测试集 MOS 与预测分数，分别得到原始及映射后的质量指标。"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr, spearmanr


def _validate_pair(target: np.ndarray, prediction: np.ndarray) -> None:
    if target.ndim != 1 or prediction.shape != target.shape or len(target) < 2:
        raise ValueError('Expected matching one-dimensional arrays with at least 2 scores')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Scores must be finite')


def quality_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Compute signed correlations; undefined constant-vector correlations are null."""
    _validate_pair(target, prediction)
    # III-A：SRCC/KRCC 衡量排序一致性，PLCC 衡量线性相关，RMSE 衡量分数误差。
    # 常量向量的相关系数未定义，保留 None；相关系数保留正负号，不用绝对值掩盖反向。
    scores = {'srcc': None, 'krcc': None, 'plcc': None,
              'rmse': float(np.sqrt(np.mean((target - prediction)**2)))}
    if np.ptp(target) > 0 and np.ptp(prediction) > 0:
        scores.update(srcc=float(spearmanr(target, prediction).statistic),
                      krcc=float(kendalltau(target, prediction).statistic),
                      plcc=float(pearsonr(target, prediction).statistic))
    return scores


def logistic5(values: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Paper Eq. (20); values may be standardized using saved center/scale."""
    # 式 (20)：expit(t)-1/2 等价于 1/2-1/(1+exp(t))，且数值计算更稳定。
    # 五个参数依次对应 beta1~beta5；输入标准化时参数表示在标准化坐标中的形式。
    amplitude, slope, midpoint, linear, offset = parameters
    return amplitude * (expit(slope * (values - midpoint)) - 0.5) + linear * values + offset


def fit_logistic5(
    target: np.ndarray, prediction: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """将预测 (N,) 映射到 MOS 尺度，仅供论文式事后报告。

    拟合时使用测试集 MOS，因此不能用于 SVR 调参或部署预测。预测先标准化
    只是五参数曲线的等价重参数化；拟合失败返回明确状态，不冒充原始预测。
    """
    _validate_pair(target, prediction)
    if len(target) < 6 or len(np.unique(prediction)) < 6 or np.ptp(target) == 0:
        return None, {'status': 'unavailable', 'reason': 'Insufficient distinct scores'}
    center, scale = float(prediction.mean()), float(prediction.std())
    if scale <= np.finfo(float).eps:
        return None, {'status': 'unavailable', 'reason': 'Prediction scale is degenerate'}
    # 这里只为曲线优化改善数值尺度，不是 SVR 训练中的 StandardScaler。
    # 用三个起点降低局部解风险；斜率/中点范围是本项目的数值约束。
    standardized = (prediction - center) / scale
    linear, offset = np.polyfit(standardized, target, 1)
    fits = []
    for slope in (0.5, 2.0, 5.0):
        result = least_squares(
            lambda parameters: logistic5(standardized, parameters) - target,
            x0=[np.sign(linear) * np.std(target), slope, 0, linear, offset],
            bounds=([-np.inf, 0.01, -20, -np.inf, -np.inf],
                    [np.inf, 100, 20, np.inf, np.inf]),
            max_nfev=3000, x_scale='jac',
        )
        if result.success and np.isfinite(result.fun).all():
            fits.append(result)
    if not fits:
        return None, {'status': 'failed', 'reason': 'All logistic optimization starts failed'}
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    mapped = logistic5(standardized, best.x)
    return mapped, {
        'status': 'ok', 'fit_scope': 'test-set post-hoc reporting only',
        'prediction_center': center, 'prediction_scale': scale,
        'parameters': best.x.tolist(), 'nfev': best.nfev,
        'parameter_order': ['amplitude', 'slope', 'midpoint', 'linear', 'offset'],
        'slope_bounds': [0.01, 100], 'midpoint_bounds': [-20, 20],
        'initial_slopes': [0.5, 2.0, 5.0],
    }
