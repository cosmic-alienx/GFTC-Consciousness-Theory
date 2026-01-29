import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns

def plot_phase_diagram(engine, save_path=None):
    """
    绘制意识相图（θ* vs D）
    """
    theta_range = np.linspace(0, 0.12, 500)
    D_range = [engine.fractal_dimension(th) for th in theta_range]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 相边界
    boundaries = engine.compute_phase_boundaries()
    colors = ['#2c3e50', '#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']
    phases = ['Unconscious', 'Drowsy', 'Dreaming', 'Awake', 'Hyper-arousal', 'Pathological']
    
    prev_boundary = 0
    for i, (key, boundary) in enumerate(boundaries.items()):
        mask = (theta_range >= prev_boundary) & (theta_range < boundary)
        ax.fill_between(theta_range[mask], 6.5, D_range[mask], 
                       alpha=0.3, color=colors[i], label=phases[i])
        prev_boundary = boundary
    
    # 最后一段
    mask = theta_range >= prev_boundary
    ax.fill_between(theta_range[mask], 6.5, D_range[mask], 
                   alpha=0.3, color=colors[-1], label=phases[-1])
    
    ax.plot(theta_range, D_range, 'k-', linewidth=2, label='$D(\\theta^*)$')
    ax.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label='$D_0=7$')
    
    ax.set_xlabel('Topological Angle $\\\\theta^*$', fontsize=12)
    ax.set_ylabel('Fractal Dimension $D$', fontsize=12)
    ax.set_title('GFTC ∞ Consciousness Phase Diagram', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.12)
    ax.set_ylim(6.5, 11.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_neuro_correlations(mapper, save_path=None):
    """
    绘制神经指标与θ*的关系
    """
    theta_range = np.linspace(0, 0.1, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # EEG分形维数
    D_eeg = [mapper.eeg_fractal_dimension(th) for th in theta_range]
    axes[0,0].plot(theta_range, D_eeg, 'b-', linewidth=2)
    axes[0,0].set_xlabel('$\\\\theta^*$')
    axes[0,0].set_ylabel('$D_{EEG}$')
    axes[0,0].set_title('EEG Fractal Dimension')
    axes[0,0].grid(True)
    
    # fMRI BOLD维数
    D_bold = [mapper.fmri_bold_dimension(th) for th in theta_range]
    axes[0,1].plot(theta_range, D_bold, 'r-', linewidth=2)
    axes[0,1].set_xlabel('$\\\\theta^*$')
    axes[0,1].set_ylabel('$D_{BOLD}$')
    axes[0,1].set_title('fMRI BOLD Dimension')
    axes[0,1].grid(True)
    
    # 连接组维数
    D_conn = [mapper.connectome_dimension(th) for th in theta_range]
    axes[1,0].plot(theta_range, D_conn, 'g-', linewidth=2)
    axes[1,0].set_xlabel('$\\\\theta^*$')
    axes[1,0].set_ylabel('$D_{W}$')
    axes[1,0].set_title('Connectome Fractal Dimension')
    axes[1,0].grid(True)
    
    # 赫斯特指数
    H = [mapper.hurst_exponent(th) for th in theta_range]
    axes[1,1].plot(theta_range, H, 'm-', linewidth=2)
    axes[1,1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1,1].set_xlabel('$\\\\theta^*$')
    axes[1,1].set_ylabel('$H$')
    axes[1,1].set_title('Hurst Exponent')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig