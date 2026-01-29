#!/usr/bin/env python
"""
验证与基准测试：检验理论预测与模拟的一致性
"""

import sys
sys.path.append('..')

import numpy as np
from core.gftc_engine import GFTCEngine
from core.physics_theory import FractalSpacetime, FieldTheoryFractal
from core.neuroscience_advanced import NeuroScienceMapper

def test_self_consistency():
    """测试自洽方程收敛性"""
    print("测试1: 自洽方程收敛性")
    engine = GFTCEngine()
    
    converged = 0
    total = 20
    for i in range(total):
        th_init = np.random.uniform(0.001, 0.1)
        sol = engine.solve_self_consistent(theta_init=th_init, max_iter=100)
        if sol['convergence']:
            converged += 1
    
    print(f"  收敛率: {converged}/{total} ({100*converged/total:.1f}%)")
    assert converged/total > 0.9, "收敛率过低!"
    
def test_fractal_dimension_bounds():
    """测试分形维数范围 (7, 11)"""
    print("\n测试2: 分形维数范围约束")
    engine = GFTCEngine()
    
    test_thetas = np.linspace(0, 0.2, 100)
    for th in test_thetas:
        D = engine.fractal_dimension(th)
        assert 7.0 <= D < 11.0, f"维数越界: D={D} at θ={th}"
    
    print("  ✓ 所有维数在 [7, 11) 范围内")

def test_chemical_stability():
    """测试化学动力学稳定性"""
    print("\n测试3: 化学动力学稳态稳定性")
    engine = GFTCEngine()
    chem = engine.chem if hasattr(engine, 'chem') else None
    
    if chem is None:
        from core.neuroscience_advanced import ChemicalKinetics
        chem = ChemicalKinetics()
    
    #测试不同θ*下的稳态
    for th in [0.01, 0.03, 0.05]:
        chi_eq = engine.solve_chemical_equilibrium(th)
        # 检查浓度非负
        assert np.all(chi_eq >= 0), f"负浓度 at θ={th}"
        # 检查有限性
        assert np.all(np.isfinite(chi_eq)), f"非有限浓度 at θ={th}"
    
    print("  ✓ 化学稳态物理合理")

def test_experimental_predictions():
    """测试实验预测范围"""
    print("\n测试4: 实验预测范围验证")
    engine = GFTCEngine()
    mapper = NeuroScienceMapper(engine)
    
    theta_samples = np.linspace(0.001, 0.1, 50)
    
    # EEG维数应在合理范围(1,2)
    D_eegs = [mapper.eeg_fractal_dimension(th) for th in theta_samples]
    assert all(1.0 <= d <= 2.0 for d in D_eegs), "EEG维数越界"
    
    # 连接组维数应>1
    D_conns = [mapper.connectome_dimension(th) for th in theta_samples]
    assert all(d >= 1.0 for d in D_conns), "连接组维数<1"
    
    # 赫斯特指数应在(0,1)
    Hs = [mapper.hurst_exponent(th) for th in theta_samples]
    assert all(0 < h < 1 for h in Hs), "赫斯特指数越界"
    
    print("  ✓ 所有神经指标在生理合理范围")

def test_field_theory_consistency():
    """测试场论计算一致性"""
    print("\n测试5: 场论计算一致性")
    
    for D in [7.0, 8.0, 9.0]:
        st = FractalSpacetime(D)
        ft = FieldTheoryFractal(st)
        
        # 测试传播子正定性（短距离）
        r_test = 0.1
        G = ft.propagator(r_test, m=1.0)
        assert G > 0, f"传播子非正 D={D}"
        
        # 测试RG流方程有限性
        couplings = np.array([0.3, 0.026, 0.03])
        beta = ft.rg_flow_equations(0, couplings)
        assert np.all(np.isfinite(beta)), f"RG流发散 D={D}"
    
    print("  ✓ 场论计算数值稳定")

def run_benchmarks():
    """运行性能基准测试"""
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)
    
    import time
    engine = GFTCEngine()
    
    # 单次求解速度
    start = time.time()
    for _ in range(10):
        engine.solve_self_consistent()
    elapsed = time.time() - start
    print(f"平均单次求解时间: {elapsed/10*1000:.2f} ms")
    
    # 大规模模拟
    start = time.time()
    thetas = np.random.rand(100) * 0.1
    results = [engine.fractal_dimension(th) for th in thetas]
    elapsed = time.time() - start
    print(f"100次维数计算时间: {elapsed*1000:.2f} ms")

def main():
    print("="*60)
    print("GFTC ∞ 验证与基准测试套件")
    print("="*60)
    
    try:
        test_self_consistency()
        test_fractal_dimension_bounds()
        test_chemical_stability()
        test_experimental_predictions()
        test_field_theory_consistency()
        
        print("\n" + "="*60)
        print("✓ 所有测试通过!")
        print("="*60)
        
        run_benchmarks()
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
    except Exception as e:
        print(f"\n✗ 错误: {e}")

if __name__ == "__main__":
    main()