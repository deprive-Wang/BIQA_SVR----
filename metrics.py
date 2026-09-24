"""比较测试集 MOS 与预测分数，分别得到原始及映射后的质量指标。"""

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr, spearmanr


def _validate_pair(target: np.ndarray, prediction: np.ndarray) -> None:
    if target.ndim != 1 or prediction.shape != target.shape or len(target) < 2:
        raise ValueError('Expected matching one-dimensional arrays with at least 2 scores')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Scores must be finite')


def quality_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """计算带符号的相关系数；常量向量使相关系数未定义，返回 None。"""
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
    """论文式 (20)；输入可用保存的中心值和尺度进行标准化。"""
    # 式 (20)：expit(t)-1/2 等价于 1/2-1/(1+exp(t))，且数值计算更稳定。
    # 五个参数依次对应 beta1~beta5；输入标准化时参数表示在标准化坐标中的形式。
    amplitude, slope, midpoint, linear, offset = parameters
    return amplitude * (expit(slope * (values - midpoint)) - 0.5) + linear * values + offset


def _fit_projected_logistic5(
    standardized: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    """在五参数直接优化停滞时，消去三个线性参数后搜索另外两个。"""
    def design(parameters: np.ndarray) -> np.ndarray:
        slope, midpoint = parameters
        sigmoid = expit(slope * (standardized - midpoint)) - 0.5
        return np.column_stack((sigmoid, standardized,
                                np.ones_like(standardized)))

    def squared_error(parameters: np.ndarray) -> float:
        matrix = design(parameters)
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = matrix @ coefficients - target
        return float(np.dot(residual, residual))

    result = differential_evolution(
        squared_error, bounds=[(0.01, 100), (-20, 20)],
        seed=0, maxiter=300, tol=1e-7,
    )
    if not result.success or not np.isfinite(result.fun):
        return None
    amplitude, linear, offset = np.linalg.lstsq(
        design(result.x), target, rcond=None,
    )[0]
    slope, midpoint = result.x
    parameters = np.array([amplitude, slope, midpoint, linear, offset])
    if not np.isfinite(parameters).all():
        return None
    return parameters, result.nfev


def fit_logistic5(
    target: np.ndarray, prediction: np.ndarray,
    *, projected_only: bool = False,
) -> tuple[np.ndarray | None, dict]:
    """将预测 (N,) 映射到 MOS 尺度，仅供论文式事后报告。

    拟合时使用测试集 MOS，因此不能用于 SVR 调参或部署预测。预测先标准化
    只是五参数曲线的等价重参数化。projected_only 用于重处理已确认直接优化
    失败的轮次；拟合失败返回明确状态，不冒充原始预测。
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
    fits = []
    if not projected_only:
        linear, offset = np.polyfit(standardized, target, 1)
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
    if fits:
        best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
        parameters, nfev = best.x, best.nfev
        optimizer = 'least_squares'
    else:
        projected = _fit_projected_logistic5(standardized, target)
        if projected is None:
            return None, {'status': 'failed',
                          'reason': 'Direct and projected logistic optimization failed'}
        parameters, nfev = projected
        optimizer = 'variable_projection_differential_evolution'
    mapped = logistic5(standardized, parameters)
    return mapped, {
        'status': 'ok', 'fit_scope': 'test-set post-hoc reporting only',
        'prediction_center': center, 'prediction_scale': scale,
        'parameters': parameters.tolist(), 'nfev': nfev,
        'optimizer': optimizer,
        'parameter_order': ['amplitude', 'slope', 'midpoint', 'linear', 'offset'],
        'slope_bounds': [0.01, 100], 'midpoint_bounds': [-20, 20],
        'initial_slopes': [] if projected_only else [0.5, 2.0, 5.0],
    }
