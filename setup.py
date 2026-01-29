#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GFTC ∞ (Geometric-Field Theory of Consciousness)
Setup configuration for fractal spacetime consciousness modeling framework
"""

import os
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# 确保系统使用UTF-8编码（Windows兼容性）
if sys.version_info < (3, 8):
    raise RuntimeError("GFTC ∞ requires Python 3.8 or higher")

# 项目根目录
here = os.path.abspath(os.path.dirname(__file__))

def read_file(filepath):
    """读取文件内容，处理编码"""
    with open(os.path.join(here, filepath), 'r', encoding='utf-8') as f:
        return f.read()

def parse_requirements(filepath):
    """解析requirements.txt，忽略注释和空行"""
    requirements = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    requirements.append(line)
    return requirements

# 读取文档
try:
    long_description = read_file("README.md")
    long_description_content_type = "text/markdown"
except FileNotFoundError:
    long_description = "Geometric-Field Theory of Consciousness with Fractal Spacetime"
    long_description_content_type = "text/plain"

# 基础依赖
install_requires = parse_requirements("requirements.txt")

# 可选依赖组
extras_require = {
    # 开发工具
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "pylint>=2.12.0",
        "jupyter>=1.0.0",
        "ipython>=8.0.0",
    ],
    # 高级可视化
    "viz": [
        "plotly>=5.0.0",
        "bokeh>=2.4.0",
        "mayavi>=4.7.0",  # 3D brain visualization
        "nilearn>=0.9.0",  # Neuroimaging visualization
        "pymatbridge>=0.5.0",  # MATLAB integration for validation
    ],
    # 高性能计算
    "hpc": [
        "mpi4py>=3.1.0",  # Parallel computing for large-scale simulations
        "dask>=2022.0.0",  # Parallel computing
        "cupy-cuda11x>=10.0.0",  # GPU acceleration (optional)
    ],
    # 神经科学数据接口
    "neurodata": [
        "mne>=1.0.0",  # EEG/MEG processing
        "nibabel>=3.2.0",  # Neuroimaging I/O
        "h5py>=3.6.0",  # HDF5 support for connectome data
    ],
    # 文档构建
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autodoc-typehints>=1.18.0",
        "myst-parser>=0.17.0",  # Markdown support for Sphinx
        "nbsphinx>=0.8.8",  # Jupyter notebook integration
    ],
    # 所有可选依赖
    "all": [],
}

# 合并所有可选依赖到 "all"
for deps in extras_require.values():
    if deps is not extras_require["all"]:
        extras_require["all"].extend(deps)

# 自定义构建命令（处理Numba预编译等）
class BuildExt(_build_ext):
    """Custom build extension to handle Numba AOT compilation if needed"""
    def run(self):
        # 可以在这里添加自定义构建步骤，如LLVM优化等
        _build_ext.run(self)

# 控制台脚本入口点
entry_points = {
    'console_scripts': [
        'gftc-solve=examples.basic_calculation:main',  # 基础求解
        'gftc-phase-diagram=examples.phase_diagram_demo:main',  # 生成相图
        'gftc-validate=examples.validation_benchmarks:main',  # 运行验证
        'gftc-benchmark=examples.validation_benchmarks:run_benchmarks',  # 性能测试
    ],
}

# 数据文件配置（包含YAML, JSON等非Python文件）
package_data = {
    'core': ['*.yaml', '*.json', '*.txt'],  # 如果核心模块有配置文件
    'data': ['*.yaml', '*.json', '*.csv', '*.npy'],  # 实验数据
}

# 排除测试文件和缓存
exclude_package_data = {
    '': ['tests', 'tests.*', '*.pyc', '__pycache__'],
}

setup(
    # 基础元数据
    name="gftc-infinite",
    version="2.0.0",
    author="GFTC Research Group",
    author_email="research@gftc-theory.org",
    maintainer="GFTC Core Team",
    maintainer_email="core@gftc-theory.org",
    description="Geometric-Field Theory of Consciousness with Infinite-Dimensional Fractal Spacetime",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    
    # 项目URL
    url="https://github.com/gftc/GFTC-infinite",
    project_urls={
        "Bug Tracker": "https://github.com/gftc/GFTC-infinite/issues",
        "Documentation": "https://gftc-infinite.readthedocs.io/",
        "Source Code": "https://github.com/gftc/GFTC-infinite",
        "Research Paper": "https://arxiv.org/abs/gftc.2024",
    },
    
    # 包发现与结构
    packages=find_packages(
        where="src",
        exclude=["tests", "tests.*", "docs", "docs.*", "examples.*"]
    ),
    package_dir={"": "src"},
    package_data=package_data,
    exclude_package_data=exclude_package_data,
    include_package_data=True,  # 包含MANIFEST.in中列出的数据文件
    
    # 依赖管理
    python_requires=">=3.8, <4.0",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # 入口点
    entry_points=entry_points,
    
    # 构建配置
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,  # 科学计算包通常设为False以便正确加载动态库
    
    # 分类元数据
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Typing :: Typed",  # 包含类型注解
    ],
    
    # 关键词
    keywords=[
        "consciousness",
        "fractal",
        "quantum-field-theory",
        "neuroscience",
        "spacetime",
        "renormalization-group",
        "connectome",
        "mathematical-physics",
        "high-dimensional-geometry"
    ],
    
    # 下载地址
    download_url="https://github.com/gftc/GFTC-infinite/archive/refs/tags/v2.0.0.tar.gz",
    
    # 平台支持
    platforms=["any"],
    
    # 许可证
    license="MIT",
    
    # 测试套件（如果使用pytest，这里设为pytest）
    test_suite="pytest",
    tests_require=["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    
    # 额外的setup参数
    options={
        'bdist_wheel': {'universal': False}  # 不是纯Python包（有Numba等依赖）
    },
)