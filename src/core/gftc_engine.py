import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.special import gamma, betaln
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GFTCParameters:
    """GFTC ∞ 理论参数"""
    # 基础分形参数
    D0: float = 7.0              # 基础维数
    Delta_D: float = 4.0         # 维数涨落幅度  
    beta: float = 10.0           # 维数变化率
    
    # 耦合常数
    y0: float = 0.3              # 基础Yukawa耦合
    g_A0: float = 0.026          # 基础轴子耦合
    
    # 神经化学调制
    eta: np.ndarray = None       # y_eff调制系数
    zeta: np.ndarray = None      # g_Aeff调制系数
    
    def __post_init__(self):
        if self.eta is None:
            self.eta = np.array([0.02, -0.01, -0.005])
        if self.zeta is None:
            self.zeta = np.array([0.01, 0.005, -0.015])

class GFTCEngine:
    """
    GFTC ∞ 主计算引擎
    实现自洽方程组求解与分形动力学
    """
    
    def __init__(self, params: Optional[GFTCParameters] = None):
        self.params = params or GFTCParameters()
        self.solution_cache = {}
        self.iteration_count = 0
        
    def fractal_dimension(self, theta_star: float) -> float:
        """
        计算动态分形维数 D(θ*)
        D(θ*) = D0 + ΔD * tanh(β * θ*)
        """
        D = self.params.D0 + self.params.Delta_D * np.tanh(self.params.beta * theta_star)
        return D
    
    def theta_fixed_point_equation(self, theta_star: float, D: float, 
                                  y_eff: float, g_Aeff: float) -> float:
        """
        拓扑角固定点方程 (E2)
        θ* = (4π²/3D) * (g_A²/y²) * Γ(D)/Γ(D/2)²
        """
        prefactor = (4 * np.pi**2) / (3 * D)
        coupling_ratio = (g_Aeff**2) / (y_eff**2)
        gamma_ratio = gamma(D) / (gamma(D/2)**2)
        
        theta_calc = prefactor * coupling_ratio * gamma_ratio
        return theta_calc - theta_star  # 返回残差
    
    def effective_couplings(self, chi: np.ndarray, D: float) -> Tuple[float, float]:
        """
        计算有效耦合常数 (E4, E5)
        """
        chi_bar = np.array([5.0, 5.0, 5.0])  # 基准浓度
        
        y_eff = self.params.y0 + np.dot(self.params.eta, (chi - chi_bar))
        g_A_base = self.params.g_A0 * (D / 7.0)**1.5
        g_Aeff = g_A_base + np.dot(self.params.zeta, (chi - chi_bar))
        
        return y_eff, g_Aeff
    
    def solve_chemical_equilibrium(self, theta_star: float, 
                                  mu: np.ndarray = None) -> np.ndarray:
        """
        求解神经化学稳态 (E3)
        返回: [chi_DA, chi_5HT, chi_GABA]
        """
        if mu is None:
            mu = np.array([1.0, 1.0, 1.0])  # 默认化学势
            
        # Sigmoid激活函数参数
        k = np.array([30.0, 25.0, 40.0])
        theta_0 = np.array([0.04, 0.03, 0.02])
        
        S_theta = 1 / (1 + np.exp(-k * (theta_star - theta_0)))
        
        # 非线性方程组求解迭代
        chi = np.array([5.0, 5.0, 5.0])  # 初始猜测
        
        for _ in range(50):  # 最大迭代
            # 参数定义
            alpha = np.array([0.8, 0.6, 1.2])
            beta = np.array([0.1, 0.08, 0.15])
            gamma_coeff = np.array([0.05, 0.04, 0.08])
            delta = np.array([0.01, 0.005, 0.02])
            
            # 计算新稳态（考虑交叉相互作用）
            chi_DA_new = (alpha[0]*S_theta[0] + gamma_coeff[0]*mu[0]) / \
                        (beta[0] + delta[0]*chi[2])  # GABA抑制DA
            chi_5HT_new = (alpha[1]*S_theta[1] + gamma_coeff[1]*mu[1]) / \
                         (beta[1] - delta[1]*chi[0])  # DA促进5HT
            chi_GABA_new = (-beta[2] + np.sqrt(beta[2]**2 + \
                          4*delta[2]*(alpha[2]*S_theta[2] + gamma_coeff[2]*mu[2]))) / \
                          (2*delta[2])  # 自抑制项解析解
            
            chi_new = np.array([chi_DA_new, chi_5HT_new, chi_GABA_new])
            
            if np.linalg.norm(chi_new - chi) < 1e-6:
                break
                
            chi = 0.5 * chi + 0.5 * chi_new  # 松弛迭代
            
        return chi
    
    def solve_self_consistent(self, theta_init: float = 0.05, 
                            max_iter: int = 100, 
                            tol: float = 1e-6) -> Dict:
        """
        自洽求解完整方程组 (算法1)
        
        Returns:
            dict: 包含 θ*, D, χ, 耦合常数的解
        """
        theta = theta_init
        D = 8.0  # 初始猜测
        
        logger.info("开始自洽求解...")
        
        for iteration in range(max_iter):
            # Step a: 用当前θ*更新D (E1)
            D_new = self.fractal_dimension(theta)
            
            # Step b & c: 求解化学平衡
            chi = self.solve_chemical_equilibrium(theta)
            
            # Step d: 计算有效耦合
            y_eff, g_Aeff = self.effective_couplings(chi, D_new)
            
            # Step e: 用(E2)更新θ*
            # 解 θ* = f(θ*) 不动点
            def fixed_point_residual(th):
                D_th = self.fractal_dimension(th)
                y, gA = self.effective_couplings(chi, D_th)
                return self.theta_fixed_point_equation(th, D_th, y, gA)
            
            # 使用牛顿法求解
            try:
                theta_new = fsolve(fixed_point_residual, theta, xtol=tol)[0]
            except:
                # 回退到简单迭代
                D_temp = self.fractal_dimension(theta)
                y_temp, gA_temp = self.effective_couplings(chi, D_temp)
                prefactor = (4 * np.pi**2) / (3 * D_temp)
                ratio = (gA_temp**2) / (y_temp**2)
                gamma_r = gamma(D_temp) / (gamma(D_temp/2)**2)
                theta_new = prefactor * ratio * gamma_r
            
            # 检查收敛
            error = abs(theta_new - theta)
            if error < tol:
                logger.info(f"收敛于第 {iteration+1} 次迭代")
                self.iteration_count = iteration + 1
                break
                
            # 阻尼更新以保证稳定性
            theta = 0.7 * theta_new + 0.3 * theta
            D = D_new
            
            if iteration == max_iter - 1:
                logger.warning("达到最大迭代次数，可能未收敛")
        
        # 最终计算
        D_final = self.fractal_dimension(theta)
        chi_final = self.solve_chemical_equilibrium(theta)
        y_eff_final, g_Aeff_final = self.effective_couplings(chi_final, D_final)
        
        solution = {
            'theta_star': theta,
            'D': D_final,
            'chi': chi_final,
            'y_eff': y_eff_final,
            'g_Aeff': g_Aeff_final,
            'convergence': error < tol,
            'iterations': self.iteration_count,
            'consciousness_phase': self.identify_consciousness_phase(theta)
        }
        
        self.solution_cache = solution
        return solution
    
    def identify_consciousness_phase(self, theta_star: float) -> str:
        """根据θ*识别意识相"""
        if theta_star < 0.008:
            return "unconscious"
        elif theta_star < 0.015:
            return "drowsy"
        elif theta_star < 0.030:
            return "dreaming"
        elif theta_star < 0.060:
            return "awake"
        else:
            return "hyper_arousal"
    
    def compute_phase_boundaries(self) -> Dict[str, float]:
        """计算理论相边界"""
        boundaries = {
            'unconscious_drowsy': 0.008,
            'drowsy_dreaming': 0.015,
            'dreaming_awake': 0.030,
            'awake_hyper': 0.060,
            'hyper_pathological': 0.100
        }
        return boundaries
    
    def beta_function(self, theta: float, y: float, g_A: float, D: float) -> float:
        """
        计算分形上的β函数 (定理4.1)
        β_θ = -D/4 * y²g_A/π⁴ * θ + g_A³/(3π²) * Γ(D/2)/Γ(2) + Δβ
        """
        term1 = -(D/4) * (y**2 * g_A / np.pi**4) * theta
        term2 = (g_A**3 / (3*np.pi**2)) * gamma(D/2) / gamma(2)
        
        # 高阶修正项
        delta_beta = (y**4 * g_A / (64*np.pi**6)) * theta**2 * (D*(D-4)/12) - \
                     (y**2 * g_A**3 / (128*np.pi**6)) * (D**2 - 16)/D
        
        return term1 + term2 + delta_beta
    
    def critical_exponent_omega(self, D: float, y: float, g_A: float, theta_star: float) -> float:
        """
        计算临界指数 ω (定义4.3)
        """
        correction = 1 - (y**2 / (32*np.pi**2)) * theta_star * (D*(D-4)/3)
        omega = (D/4) * (y**2 * g_A / np.pi**4) * correction
        return omega