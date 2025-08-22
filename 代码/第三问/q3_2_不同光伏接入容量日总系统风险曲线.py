# -*- coding: utf-8 -*-
import os, pickle, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # 导入SciencePlots库
from scipy.interpolate import UnivariateSpline
from matplotlib.font_manager import FontProperties
from pathlib import Path
HERE = Path(__file__).resolve().parent

# 设置中文字体
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)
# ===================== 可调参数 =====================
SMOOTH_S = 1e-3  # 样条平滑
SEED = 2025

# 设置科学绘图样式
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'text.usetex': False,
})
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(SEED)
CACHE_FILE = str((HERE.parent / "pchie_cache.pkl").resolve())

# 加载样条
def analyze_risk(folder_path, smooth_factor=SMOOTH_S):
    rec = []
    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".xlsx"):
            df = pd.read_excel(os.path.join(folder_path, fn))
            rec.append({"capacity": float(df["capacity"][0]),
                        "total_loss": float(df["expected_risk"].iloc[-1])})
    if not rec:
        raise RuntimeError("未找到 xlsx 数据")
    df = pd.DataFrame(rec).sort_values("capacity")
    return UnivariateSpline(df["capacity"], df["total_loss"], k=3, s=smooth_factor)


def load_or_compute_spline(folder, s=SMOOTH_S):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f: return pickle.load(f)
    sp = analyze_risk(folder, s);
    pickle.dump(sp, open(CACHE_FILE, "wb"))
    return sp


if __name__ == "__main__":
    print(f"[平台] {platform.system()}  Python {platform.python_version()}  SciPy ≥1.9 推荐")
    data_folder = str((HERE.parent /"第二问"/ "问题二结果_预先运行的结果").resolve())
    spline = load_or_compute_spline(data_folder)
    Ms = np.arange(30, 901, 10)
    risks = []

    # 基础功率配置
    Gp = np.array([0.29, 3.10, 7.19, 10.50, 13.94, 11.14,
                   16.41, 10.14, 10.93, 8.24, 5.65, 2.05])

    for M in Ms:
        print(f"· M={M:>3} kW …", end=" ", flush=True)
        # 计算基础容量配置
        c = (Gp / Gp.max()) * M

        # 计算所有决策变量为0时的风险 (x_i=0)
        total_risk = np.sum(spline(c))
        risks.append(total_risk)
        print(f"risk {total_risk:.2f}")

    # —— 输出汇总表 ——
    df_out = pd.DataFrame({
        "M (kW)": Ms,
        "Risk (x=0)": risks
    })
    print("\n================ 每个功率下的风险值 (x_i=0) ================\n")
    print(df_out.to_string(index=False, formatters={"Risk (x=0)": "{:.2f}".format}))

    # 作图
    DEEP_BLUE_PURPLE = '#483D8B'
    # 将风险值除以10000.0用于绘图
    risks_plot = [r / 10000.0 for r in risks]

    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(Ms, risks_plot, marker='o', markersize=3,
             color=DEEP_BLUE_PURPLE, linestyle='-', linewidth=1.5)
    plt.xlabel("光伏最大接入容量(kW)", fontsize=14, fontweight='bold', fontproperties=font)
    plt.ylabel("日总系统风险", fontsize=14, fontweight='bold',fontproperties=font)
    plt.grid(alpha=.4, linestyle="--")
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylim(85, 115)  # 固定纵轴范围为 85-115
    plt.tight_layout()
    plt.savefig('各M对应的风险曲线(决策变量全为0).pdf', bbox_inches='tight', facecolor='white')
    plt.show()