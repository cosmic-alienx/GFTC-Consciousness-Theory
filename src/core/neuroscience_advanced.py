import numpy as np
from scipy.integrate import odeint
from scipy.signal import welch
from typing import Dict, Tuple, Optional
import warnings

class ChemicalKinetics:
    """
    神经化学动力学系统 (附录B)
    """
    
    def __init__(self):
        # 基础参数
        self.alpha = np.array([0.8, 0.6, 1.2])    # 合成率
        self.beta = np.array([0.1, 0.08, 0.15])   # 降解率
        self.gamma = np.array([0.05, 0.04, 0.08]) # 化学势响应
        self.delta = np.array([0.01, 0.005, 0.02]) # 相互作用
        
        # Sigmoid参数
        self.k_params = np.array([30.0, 25.0, 40.0])
        self.theta_0 = np.array([0.04, 0.03, 0.02])
        
    def activation_function(self, theta_star: float) -> np.ndarray:
        """S_α(θ*)"""
        return 1 / (1 + np.exp(-self.k_params * (theta_star - self.theta_0)))
    
    def dynamics(self, chi: np.ndarray, t: float, theta_star: float, 
                mu: np.ndarray) -> np.ndarray:
        """
        化学动力学微分方程 (附录B.1)
        dχ_α/dt = α S_α(θ*) - β χ_α + γ μ_α ± δ 相互作用项
        """
        S = self.activation_function(theta_star)
        
        # DA方程: -δ_DA * χ_DA * χ_GABA
        dDA_dt = (self.alpha[0]*S[0] - self.beta[0]*chi[0] + 
                 self.gamma[0]*mu[0] - self.delta[0]*chi[0]*chi[2])
        
        # 5HT方程: +δ_5HT * χ_DA * χ_5HT (DA促进5HT)
        d5HT_dt = (self.alpha[1]*S[1] - self.beta[1]*chi[1] + 
                  self.gamma[1]*mu[1] + self.delta[1]*chi[0]*chi[1])
        
        # GABA方程: -δ_GABA * χ_GABA² (自抑制)
        dGABA_dt = (self.alpha[2]*S[2] - self.beta[2]*chi[2] + 
                   self.gamma[2]*mu[2] - self.delta[2]*chi[2]**2)
        
        return np.array([dDA_dt, d5HT_dt, dGABA_dt])
    
    def solve_trajectory(self, chi_init: np.ndarray, t_span: np.ndarray, 
                        theta_func: callable, mu: np.ndarray) -> np.ndarray:
        """
        求解化学动力学轨迹（时变θ*）
        """
        def ode_system(chi, t):
            th = theta_func(t)
            return self.dynamics(chi, t, th, mu)
        
        solution = odeint(ode_system, chi_init, t_span)
        return solution

class NeuroScienceMapper:
    """
    神经科学映射模块 (第V部分)
    连接分形理论与神经观测
    """
    
    def __init__(self, gftc_engine):
        self.engine = gftc_engine
        self.chem = ChemicalKinetics()
        
    def eeg_fractal_dimension(self, theta_star: float) -> float:
        """
        预测EEG分形维数 (定理5.2, 附录C)
        D_EEG = 1.52 + 7.83θ* - 28.6θ*² + 31.4θ*³
        """
        D_eeg = (1.52 + 7.83*theta_star - 28.6*theta_star**2 + 
                31.4*theta_star**3)
        return np.clip(D_eeg, 1.0, 2.0)
    
    def hurst_exponent(self, theta_star: float) -> float:
        """
        赫斯特指数与拓扑角关系 (定理5.2)
        H(θ*) = 0.5 + 0.3 * tanh(5θ*)
        """
        H = 0.5 + 0.3 * np.tanh(5 * theta_star)
        return H
    
    def fmri_bold_dimension(self, theta_star: float) -> float:
        """
        fMRI BOLD信号分形维数 (定义5.3)
        """
        boundaries = [0.008, 0.015, 0.03, 0.06]
        values = [1.1, 1.3, 1.5, 1.6, 1.7]
        
        if theta_star < boundaries[0]:
            return values[0]
        elif theta_star < boundaries[1]:
            return values[1]
        elif theta_star < boundaries[2]:
            return values[2]
        elif theta_star < boundaries[3]:
            return values[3]
        else:
            return values[4]
    
    def connectome_dimension(self, theta_star: float) -> float:
        """
        大脑连接组分形维数 (定理5.1)
        D_W(θ*) = 1.15 + 18.0θ* - 45.0θ*²
        """
        D_w = 1.15 + 18.0*theta_star - 45.0*(theta_star**2)
        return max(D_w, 1.0)  # 维数不低于1
    
    def power_spectral_density(self, f: np.ndarray, theta_star: float) -> np.ndarray:
        """
        BOLD信号功率谱密度 P(f) ~ f^{-β(θ*)} (定义5.3)
        β(θ*) = 2*D_BOLD(θ*) - 1
        """
        D_bold = self.fmri_bold_dimension(theta_star)
        beta_exp = 2*D_bold - 1
        psd = f**(-beta_exp)
        return psd / np.sum(psd)  # 归一化
    
    def simulate_eeg_signal(self, duration: float = 10.0, 
                           fs: float = 256.0, 
                           theta_star: float = 0.03) -> np.ndarray:
        """
        基于分形维数生成模拟EEG信号
        使用分数布朗运动模型
        """
        from scipy.special import gamma as gamma_func
        
        H = self.hurst_exponent(theta_star)
        n_points = int(duration * fs)
        t = np.arange(n_points) / fs
        
        # 使用Hosking方法生成分数布朗噪声
        # 简化实现：频域方法
        freqs = np.fft.rfftfreq(n_points, 1/fs)
        freqs[0] = 1e-10  # 避免除零
        
        # 功率谱 ~ 1/f^{2H+1}
        power = freqs**(-(2*H + 1))
        power[0] = 0
        
        # 随机相位
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        fft_signal = np.sqrt(power) * np.exp(1j*phases)
        
        signal = np.fft.irfft(fft_signal, n=n_points)
        return signal
    
    def compute_connectivity_matrix(self, n_regions: int = 68, 
                                   theta_star: float = 0.03) -> np.ndarray:
        """
        生成分形连接矩阵 (定义5.1)
        C_ij(θ*) = d_ij^{-(D_W-1)} * exp(-d_ij/l(θ*))
        """
        # 使用标准大脑距离（Desikan-Killiany图谱近似）
        np.random.seed(42)
        coords = np.random.randn(n_regions, 3)  # 简化：随机3D坐标
        
        # 计算欧氏距离
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        dist[dist == 0] = np.inf  # 对角线
        
        D_w = self.connectome_dimension(theta_star)
        l_theta = 10 + 100*theta_star  # 特征长度依赖θ*
        
        # 分形关联函数
        W = (dist**(-(D_w - 1))) * np.exp(-dist/l_theta)
        np.fill_diagonal(W, 0)
        
        # 归一化
        W = (W + W.T) / 2  # 对称
        return W
    
    def eeg_empirical_relation(self, D_eeg: float) -> float:
        """
        从EEG分形维数反推θ*（实验数据拟合反函数）
        """
        # 简化为查找表/数值反演
        theta_range = np.linspace(0, 0.1, 1000)
        D_pred = [self.eeg_fractal_dimension(th) for th in theta_range]
        
        # 最近邻查找
        idx = np.argmin(np.abs(np.array(D_pred) - D_eeg))
        return theta_range[idx]
    
    def validate_prediction(self, measured_D_eeg: float, 
                          measured_D_bold: float) -> Dict:
        """
        验证实验数据与理论预测的一致性 (第7.2节)
        """
        # 从EEG推断θ*
        theta_from_eeg = self.eeg_empirical_relation(measured_D_eeg)
        
        # 预测BOLD维数
        predicted_D_bold = self.fmri_bold_dimension(theta_from_eeg)
        
        # 一致性检查
        discrepancy = abs(predicted_D_bold - measured_D_bold)
        is_consistent = discrepancy < 0.1  # 阈值
        
        return {
            'theta_inferred': theta_from_eeg,
            'D_bold_predicted': predicted_D_bold,
            'D_bold_measured': measured_D_bold,
            'discrepancy': discrepancy,
            'is_consistent': is_consistent,
            'consciousness_state': self.engine.identify_consciousness_phase(theta_from_eeg)
        }