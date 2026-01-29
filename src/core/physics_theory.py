import numpy as np
from scipy.special import gamma
from typing import Callable, Tuple, Optional
import numba as nb

class FractalSpacetime:
    """
    分形时空构造与测度理论
    
    实现GFTC-1第I部分的数学结构
    """
    
    def __init__(self, dimension: float, randomness_sigma: float = 0.01):
        self.D = dimension
        self.sigma = randomness_sigma
        self.epsilon = 1e-6  # 正则化截断
        
    def hausdorff_measure(self, sets: list, s: float) -> float:
        """
        计算s维豪斯多夫测度近似 (定义1.4)
        H^s(E) = lim_{δ→0} inf{Σ(diam U_i)^s}
        """
        measure = 0.0
        for subset in sets:
            diameter = subset['diameter']
            if diameter < self.epsilon:
                measure += diameter ** s
        return measure
    
    def fractal_integral(self, f: Callable, points: np.ndarray, 
                        measures: np.ndarray) -> float:
        """
        分形上的积分 (定义1.7)
        ∫_{F_D} f(x) dμ_D(x) ≈ Σ f(x_α) μ_D(Q_α)
        """
        values = f(points)
        integral = np.sum(values * measures)
        return integral
    
    def fractal_gradient(self, f: Callable, x: np.ndarray, 
                        neighbors: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """
        分形梯度计算 (定义1.8)
        ∇_{F_D} f(x) = lim [f(y)-f(x)] / d_D(x,y)^{D-1} * (y-x)/|y-x|
        """
        fx = f(x)
        gradients = []
        
        for y, d in zip(neighbors, distances):
            if d < 1e-10:
                continue
            fy = f(y)
            direction = (y - x) / np.linalg.norm(y - x)
            frac_grad = (fy - fx) / (d**(self.D - 1)) * direction
            gradients.append(frac_grad)
            
        return np.mean(gradients, axis=0) if gradients else np.zeros_like(x)
    
    def generate_fractal_points(self, n_points: int = 10000, 
                               iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成高维随机分形点集 (附录E.1)
        使用迭代函数系统(IFS)方法
        """
        # 初始超立方体 [0,1]^11
        dim = 11
        points = np.random.rand(n_points, dim)
        
        # IFS参数：模拟分形构造
        N_maps = 50  # 映射数量
        c_avg = np.exp(-np.log(N_maps) / self.D)  # 压缩因子
        
        fractal_points = []
        weights = []
        
        for _ in range(iterations):
            new_points = []
            for point in points:
                # 随机选择变换
                for _ in range(min(5, N_maps//10)):  # 采样部分映射
                    # 仿射变换: x' = c * R * x + b + ξ
                    rotation = self._random_rotation(dim)
                    translation = np.random.rand(dim) * (1 - c_avg)
                    noise = np.random.normal(0, self.sigma, dim)
                    
                    new_pt = c_avg * (rotation @ point) + translation + noise
                    new_points.append(new_pt)
                    
                    # 测度权重 ~ c^D
                    weight = c_avg ** self.D
                    weights.append(weight)
            
            points = np.array(new_points[:n_points])  # 控制点数
        
        weights = np.array(weights[:len(points)])
        weights /= np.sum(weights)  # 归一化
        
        return points, weights
    
    def _random_rotation(self, dim: int) -> np.ndarray:
        """生成随机旋转矩阵（正交矩阵）"""
        from scipy.stats import ortho_group
        return ortho_group.rvs(dim)
    
    def metric_tensor(self, x: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
        """
        计算分形度规张量 g^μν_D(x) (引理2.1)
        """
        dim = len(x)
        metric = np.zeros((dim, dim))
        
        for y in neighborhood:
            diff = y - x
            dist_sq = np.sum(diff**2)
            if dist_sq < self.epsilon:
                continue
            
            # g^μν = lim ∫ (y^μ-x^μ)(y^ν-x^ν)/|y-x|² dμ_D(y) / ε^D
            outer = np.outer(diff, diff) / dist_sq
            metric += outer
            
        metric /= len(neighborhood)
        return metric

class FieldTheoryFractal:
    """
    分形上的量子场论
    
    实现GFTC-1第II、IV部分的场论计算
    """
    
    def __init__(self, spacetime: FractalSpacetime):
        self.st = spacetime
        self.D = spacetime.D
        
    def propagator(self, r: float, m: float = 1.0) -> float:
        """
        分形上的自由传播子 (定理D.1)
        G_D(r) ~ 1/r^{D-2} (r << 1/m)
        """
        if r < 1e-10:
            return 0.0
        
        # 使用修正贝塞尔函数插值
        from scipy.special import kv
        prefactor = 1 / ((2*np.pi)**(self.D/2))
        x = m * r
        if x < 0.1:
            # 小距离行为: ~ r^{-(D-2)}
            return prefactor * (m/r)**(self.D/2 - 1) * \
                   0.5 * gamma(self.D/2 - 1) * (2/x)**(self.D/2 - 1)
        else:
            # 大距离行为
            return prefactor * (m/r)**(self.D/2 - 1) * kv(self.D/2 - 1, x)
    
    def one_loop_effective_potential(self, phi: np.ndarray, m: float, 
                                    lam: float) -> np.ndarray:
        """
        单圈有效势 (计算D.1)
        """
        # 树层级
        V_tree = 0.5 * m**2 * phi**2 + lam * phi**4 / 24
        
        # 单圈修正（维度正则化）
        coeff = lam / (2 * (4*np.pi)**(self.D/2))
        # 使用递推关系处理Gamma函数在负参数的行为
        D_eff = min(self.D, 3.99)  # 避免发散
        
        try:
            gamma_term = gamma(1 - D_eff/2) / (D_eff/2)
            correction = coeff * (m**2)**(D_eff/2) * gamma_term * \
                        (1 + lam*phi**2/(2*m**2))**(D_eff/2)
        except:
            correction = 0
            
        return V_tree + correction
    
    def rg_flow_equations(self, t: float, couplings: np.ndarray) -> np.ndarray:
        """
        RG流方程 (定理D.2)
        dy/dlnμ = (D-4)/2 * y + 3y³/(32π²)
        """
        y, g_A, theta = couplings
        
        beta_y = ((self.D - 4)/2) * y + (3*y**3)/(32*np.pi**2)
        beta_gA = ((self.D - 4)/2) * g_A + (g_A**3)/(16*np.pi**2)
        
        # θ的β函数
        beta_theta = -(self.D/4)*(y**2 * g_A / np.pi**4)*theta + \
                     (g_A**3/(3*np.pi**2)) * gamma(self.D/2)/gamma(2)
        
        return np.array([beta_y, beta_gA, beta_theta])
    
    def scale_dimension(self, d_base: int, eta_D: float) -> float:
        """
        分形上的标度维数 (定义4.1)
        Δ_φ(D) = (d-2+η)/2 * D/d
        """
        return ((d_base - 2 + eta_D)/2) * (self.D / d_base)
    
    def correlation_length_exponent(self, omega: float) -> float:
        """
        关联长度指数 ν = 1/ω (定理4.3)
        """
        if abs(omega) < 1e-10:
            return np.inf
        return 1.0 / omega