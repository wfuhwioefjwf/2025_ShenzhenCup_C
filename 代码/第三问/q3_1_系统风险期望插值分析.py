import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.font_manager import FontProperties
import scienceplots  # 导入SciencePlots库
from pathlib import Path
HERE = Path(__file__).resolve().parent

# 设置中文字体
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)
# 使用科学绘图样式
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',  # 使用 STIX 数学字体
    'text.usetex': False,  # 不使用 LaTeX 渲染所有文本，仅用于数学表达式
})
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
CACHE_FILE = str((HERE.parent / "pchip_cache.pkl").resolve())

def analyze_risk(folder_path):
    # 先看缓存
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    # 读数据并缩放
    records = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(folder_path, fn))
            cap = float(df['capacity'].iloc[0])
            loss = float(df['expected_risk'].iloc[-1]) / 10000
            records.append((cap, loss))
    if not records:
        raise RuntimeError("找不到任何 .xlsx 文件！")
    df = pd.DataFrame(records, columns=['capacity', 'loss'])
    df.sort_values('capacity', inplace=True)
    x = df['capacity'].values
    y = df['loss'].values

    pchip = PchipInterpolator(x, y)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(pchip, f)
    return pchip

def plot_pchip(folder_path):
    # 拿到插值对象
    pchip = analyze_risk(folder_path)
    # 重新读数据
    records = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(folder_path, fn))
            cap = float(df['capacity'].iloc[0])
            loss = float(df['expected_risk'].iloc[-1]) / 10000
            records.append((cap, loss))
    df = pd.DataFrame(records, columns=['capacity', 'loss']).sort_values('capacity')
    # 插值曲线
    xs = np.linspace(df['capacity'].min(), df['capacity'].max(), 500)
    ys = pchip(xs)

    for i in range(len(xs)):
        print(f"x = {xs[i]}, y = {ys[i]}")

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.scatter(df['capacity'], df['loss'], marker='o', s=40, c='#0087BD',
               edgecolor='#191970', linewidths=1.5, zorder=3, label='原始观测值')
    ax.plot(xs, ys,
            color='#191970',  # 线色
            linewidth=2.5,
            label='PCHIP 分段 Hermite 插值')
    # 设置坐标轴标签和标题
    ax.set_xlabel('DG容量(kW)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.set_ylabel('系统风险', fontsize=14, fontweight='bold', fontproperties=font)
    # 设置坐标轴范围
    ax.set_xlim(df['capacity'].min() * 0.95, df['capacity'].max() * 1.05)
    ax.set_ylim(df['loss'].min() * 0.98, df['loss'].max() * 1.02)
    ax.grid(True, linestyle='--', alpha=0.4)

    # 添加图例
    ax.legend(loc='best', prop=font, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig('期望损失_PCHIP插值分析.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == '__main__':
    folder = str((HERE.parent /"第二问"/ "问题二结果_预先运行的结果").resolve())
    plot_pchip(folder)