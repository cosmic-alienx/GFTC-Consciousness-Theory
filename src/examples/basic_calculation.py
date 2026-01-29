#!/usr/bin/env python
"""
基础计算示例：求解自洽方程并输出意识状态
"""

import sys
sys.path.append('..')

from core.gftc_engine import GFTCEngine, GFTCParameters
from core.neuroscience_advanced import NeuroScienceMapper

def main():
    print("="*60)
    print("GFTC ∞ 基础计算演示")
    print("="*60)
    
    # 初始化引擎
    params = GFTCParameters(
        D0=7.0,
        Delta_D=4.0,
        beta=10.0,
        y0=0.3,
        g_A0=0.026
    )
    
    engine = GFTCEngine(params)
    mapper = NeuroScienceMapper(engine)
    
    # 求解不同初始条件下的自洽解
    initial_conditions = [0.01, 0.03, 0.05, 0.08]
    
    print("\n自洽方程求解结果:")
    print("-"*60)
    print(f"{'Initial θ*':<12} {'Final θ*':<12} {'D':<8} {'Phase':<15} {'D_EEG':<8}")
    print("-"*60)
    
    for th_init in initial_conditions:
        sol = engine.solve_self_consistent(theta_init=th_init)
        D_eeg = mapper.eeg_fractal_dimension(sol['theta_star'])
        
        print(f"{th_init:<12.4f} {sol['theta_star']:<12.4f} "
              f"{sol['D']:<8.2f} {sol['consciousness_phase']:<15} "
              f"{D_eeg:<8.2f}")
    
    # 详细分析清醒状态
    print("\n" + "="*60)
    print("清醒状态详细分析:")
    sol_awake = engine.solve_self_consistent(theta_init=0.05)
    
    print(f"拓扑角 θ*: {sol_awake['theta_star']:.6f}")
    print(f"分形维数 D: {sol_awake['D']:.4f}")
    print(f"神经化学浓度:")
    print(f"  - DA:    {sol_awake['chi'][0]:.3f} μM")
    print(f"  - 5HT:   {sol_awake['chi'][1]:.3f} μM")
    print(f"  - GABA:  {sol_awake['chi'][2]:.3f} μM")
    print(f"有效耦合:")
    print(f"  - y_eff: {sol_awake['y_eff']:.4f}")
    print(f"  - g_Aeff: {sol_awake['g_Aeff']:.6f}")
    
    # 神经指标预测
    print(f"\n神经指标预测:")
    print(f"  - EEG分形维数: {mapper.eeg_fractal_dimension(sol_awake['theta_star']):.2f}")
    print(f"  - BOLD分形维数: {mapper.fmri_bold_dimension(sol_awake['theta_star']):.2f}")
    print(f"  - 连接组维数: {mapper.connectome_dimension(sol_awake['theta_star']):.2f}")
    print(f"  - 赫斯特指数: {mapper.hurst_exponent(sol_awake['theta_star']):.3f}")

if __name__ == "__main__":
    main()