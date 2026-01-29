import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

def plot_fractal_3d(points, weights=None, save_path=None):
    """
    3D分形点云可视化（前3维投影）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if weights is not None:
        colors = plt.cm.viridis(weights / np.max(weights))
    else:
        colors = 'b'
    
    ax.scatter(points[:,0], points[:,1], points[:,2], 
              c=colors, s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Fractal Spacetime Projection ($D_{eff}$ visualization)')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_connectome_brain(W, save_path=None):
    """
    绘制大脑连接图
    """
    G = nx.from_numpy_array(W)
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 边强度映射到颜色
    edges = G.edges()
    weights = [W[u,v] for u,v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                          alpha=0.3, edge_color=weights, 
                          edge_cmap=plt.cm.plasma, ax=ax)
    
    ax.set_title('Fractal Brain Connectome', fontsize=14)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_rg_flow(engine, save_path=None):
    """
    绘制重整化群流图
    """
    from scipy.integrate import odeint
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 参数空间流 (y, g_A)
    y_range = np.linspace(0.1, 0.5, 20)
    g_range = np.linspace(0.01, 0.05, 20)
    Y, G = np.meshgrid(y_range, g_range)
    
    D = 8.0  # 固定维数
    theta_fixed = 0.03
    
    dY = ((D-4)/2)*Y + (3*Y**3)/(32*np.pi**2)
    dG = ((D-4)/2)*G + (G**3)/(16*np.pi**2)
    
    # 归一化箭头
    M = np.sqrt(dY**2 + dG**2)
    M[M == 0] = 1
    
    axes[0].quiver(Y, G, dY/M, dG/M, M, cmap='coolwarm')
    axes[0].set_xlabel('$y$')
    axes[0].set_ylabel('$g_A$')
    axes[0].set_title('RG Flow in Coupling Space')
    axes[0].set_xlim(0.1, 0.5)
    axes[0].set_ylim(0.01, 0.05)
    
    # θ* 的beta函数
    theta_vals = np.linspace(0, 0.1, 100)
    beta_vals = [engine.beta_function(th, 0.3, 0.026, D) for th in theta_vals]
    
    axes[1].plot(theta_vals, beta_vals, 'b-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('$\\\\theta$')
    axes[1].set_ylabel('$\\\\beta_\\\\theta$')
    axes[1].set_title(f'Beta Function for $D={D}$')
    axes[1].grid(True)
    
    # 标记固定点
    zero_crossings = np.where(np.diff(np.sign(beta_vals)))[0]
    for zc in zero_crossings:
        axes[1].plot(theta_vals[zc], 0, 'ro', markersize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig