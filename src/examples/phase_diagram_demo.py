#!/usr/bin/env python
"""
相图演示：生成完整的意识相图与神经指标关系图
"""

import sys
sys.path.append('..')

from core.gftc_engine import GFTCEngine
from core.neuroscience_advanced import NeuroScienceMapper
from utils.visualization import plot_phase_diagram, plot_neuro_correlations
import matplotlib.pyplot as plt

def main():
    engine = GFTCEngine()
    mapper = NeuroScienceMapper(engine)
    
    print("生成相图...")
    fig1 = plot_phase_diagram(engine, save_path='phase_diagram.png')
    
    print("生成神经指标关联图...")
    fig2 = plot_neuro_correlations(mapper, save_path='neuro_correlations.png')
    
    print("生成高分辨率相图数据...")
    # 生成相边界详细数据
    boundaries = engine.compute_phase_boundaries()
    print("\n理论相边界:")
    for name, theta in boundaries.items():
        D = engine.fractal_dimension(theta)
        print(f"{name}: θ* = {theta:.3f}, D = {D:.2f}")
    
    # 验证相一致性
    print("\n验证相一致性:")
    test_thetas = [0.005, 0.012, 0.020, 0.040, 0.070]
    for th in test_thetas:
        sol = engine.solve_self_consistent(theta_init=th)
        print(f"初始θ={th:.3f} -> 收敛于 {sol['consciousness_phase']} "
              f"(θ*={sol['theta_star']:.3f}, D={sol['D']:.2f})")
    
    plt.show()
    print("\n图表已保存!")

if __name__ == "__main__":
    main()