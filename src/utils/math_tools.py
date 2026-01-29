import numpy as np
from scipy.special import gamma, gammaln
from scipy.linalg import expm
import numba as nb

def safe_gamma_ratio(D: float) -> float:
    """
    安全计算 Γ(D)/Γ(D/2)²，避免溢出
    使用对数Gamma函数
    """
    log_ratio = gammaln(D) - 2*gammaln(D/2)
    return np.exp(log_ratio)

@nb.njit
def fast_tanh(x: float) -> float:
    """快速tanh近似（Numba加速）"""
    if x > 5:
        return 1.0
    elif x < -5:
        return -1.0
    else:
        return np.tanh(x)

def rotation_matrix_11d(axis: int, angle: float) -> np.ndarray:
    """11维空间中的旋转矩阵"""
    R = np.eye(11)
    c, s = np.cos(angle), np.sin(angle)
    # 简化为2D旋转在选定平面
    idx = [(axis % 11), ((axis+1) % 11)]
    R[idx[0], idx[0]] = c
    R[idx[0], idx[1]] = -s
    R[idx[1], idx[0]] = s
    R[idx[1], idx[1]] = c
    return R

def iter_solve_fixed_point(func, x0, tol=1e-6, max_iter=100):
    """
    通用不动点迭代求解器
    """
    x = x0
    for i in range(max_iter):
        x_new = func(x)
        if abs(x_new - x) < tol:
            return x_new, True, i+1
        x = 0.5*x + 0.5*x_new  # 松弛
    return x, False, max_iter