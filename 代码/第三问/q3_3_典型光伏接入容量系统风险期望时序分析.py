import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from sklearn.metrics import r2_score
import scienceplots  # 导入SciencePlots库
from matplotlib.font_manager import FontProperties
from pathlib import Path
HERE = Path(__file__).resolve().parent

# 设置中文字体
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)

# —— 全局设置 ——
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'text.usetex': False,
})
plt.rcParams["font.family"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False

CACHE_FILE = str((HERE.parent / "pchip_cache.pkl").resolve())
PV_DEGREE = 6  # 光伏曲线多项式拟合阶数

# 定义颜色方案
COLORS = {
    'deep_blue': '#4DABD1',
    'gold_yellow': '#B7950B',
    'cyan': '#6F9C71',
    'deep_purple': '#4A5490'
}

# ------------------- 公共函数 -------------------
def read_capacity_and_risk(folder_path, divide_risk_by=1.0):
    """从文件夹中读取所有 .xlsx 的 capacity 与 expected_risk。
    divide_risk_by: 是否需要对 risk 做缩放（如/10000）。"""
    records = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.xlsx'):
            path = os.path.join(folder_path, fname)
            try:
                df = pd.read_excel(path)
                cap = float(df['capacity'].iloc[0])
                risk = float(df['expected_risk'].iloc[-1]) / divide_risk_by
                records.append((cap, risk))
            except Exception as e:
                print(f"跳过 {fname}: {e}")
    if not records:
        raise RuntimeError("未找到有效的 .xlsx 文件或数据列名不匹配。")
    records.sort(key=lambda x: x[0])
    x = np.array([r[0] for r in records])
    y = np.array([r[1] for r in records])
    return x, y

def analyze_risk(folder_path, smooth_factor=0):
    x, y = read_capacity_and_risk(folder_path, divide_risk_by=1.0)
    return UnivariateSpline(x, y, k=3, s=smooth_factor)

def analyze_and_fit_pchip(folder_path):
    x, y = read_capacity_and_risk(folder_path, divide_risk_by=10000.0)
    return PchipInterpolator(x, y)

def load_or_compute_spline(folder_path, smooth_factor=0):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            spline = pickle.load(f)
    else:
        spline = analyze_risk(folder_path, smooth_factor)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(spline, f)
    return spline

def load_or_compute_pchip(folder_path):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            pchip = pickle.load(f)
        print(f"已从缓存加载 PCHIP 插值：{CACHE_FILE}")
    else:
        pchip = analyze_and_fit_pchip(folder_path)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(pchip, f)
        print(f"已计算并缓存 PCHIP 插值到：{CACHE_FILE}")
    return pchip

def fit_pv_curve():
    # 原始百分比转为 [0,1]
    G_percent = np.array([0.29, 3.10, 7.19, 10.50, 13.94, 11.14,
                          16.41, 10.14, 10.93, 8.24, 5.65, 2.05,
                          0.39, 0.02])
    G_frac_raw = G_percent / 100.0
    t = np.arange(6, 6 + len(G_frac_raw))
    # 先对原始比例进行多项式拟合
    coeffs = np.polyfit(t, G_frac_raw, deg=PV_DEGREE)
    G_fit_raw = np.polyval(coeffs, t)
    r2 = r2_score(G_frac_raw, G_fit_raw)
    # 再做归一化处理：拟合曲线除以其最大值
    peak = np.max(G_fit_raw)

    def g_poly(x):
        return np.polyval(coeffs, x) / peak

    def g_safe(x):
        return np.clip(g_poly(x), 0.0, 1.0)

    print(f"光伏多项式拟合 R² = {r2:.4f}")
    print("原始拟合系数（高次→低次）:", coeffs)
    return g_safe, t, coeffs, r2

# ------------------- 热图分析（原 UnivariateSpline 部分） -------------------
def plot_heatmap(folder):
    spline = load_or_compute_spline(folder, smooth_factor=0)
    # 重新读取所有 capacity，用于确定细网格范围
    caps = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(folder, fname))
                caps.append(float(df["capacity"].iloc[0]))
            except:
                pass
    min_cap, max_cap = min(caps), max(caps)
    # —— 发电比例和时间点 ——
    G_percent = [0.29, 3.10, 7.19, 10.50, 13.94, 11.14,
                 16.41, 10.14, 10.93,  8.24,  5.65,  2.05,
                  0.39,  0.02]
    ratios = np.array(G_percent) / max(G_percent)
    t_hours = np.arange(6, 6 + len(ratios))
    # —— 构建更细的 Crated 网格 ——
    n_fine = 300
    Crated_fine = np.linspace(min_cap, max_cap, n_fine)
    # —— 计算风险矩阵 ——
    risk_matrix = np.zeros((n_fine, len(t_hours)))
    for i, cap in enumerate(Crated_fine):
        inst = cap * ratios
        risk_matrix[i, :] = spline(inst)

    plt.figure(figsize=(8, 6))
    plt.imshow(risk_matrix, origin='lower', aspect='auto',
        extent=[t_hours.min(), t_hours.max(), min_cap, max_cap],
        interpolation='bilinear', cmap='viridis'
    )
    plt.grid(False)
    plt.xlabel('时间(h)', fontsize=14, fontweight='bold', fontproperties=font)
    plt.ylabel('最大接入容量(kW)', fontsize=14, fontweight='bold', fontproperties=font)
    plt.colorbar(label='系统风险').set_label('系统风险', fontsize=12, fontproperties=font)
    plt.tight_layout()
    plt.savefig('不同时间功率下期望损失热图.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

# ------------------- 时序期望损失分析 (原 Pchip 部分) -------------------
def plot_and_export(pchip, excel_path):
    # 1) 总期望损失曲线
    capacities = np.arange(0, 901, 30)
    losses_total = pchip(capacities) * 10000
    df_tot = pd.DataFrame({
        'Crated (kW)': capacities,
        '总期望损失 (元)': losses_total
    })
    # 2) PV 多项式拟合并归一化
    g_safe, t_hours, *_ = fit_pv_curve()
    # 3) 时序损失
    scenario_caps = [100, 300, 500, 700]
    scenario_colors = [
        COLORS['deep_blue'],
        COLORS['gold_yellow'],
        COLORS['cyan'],
        COLORS['deep_purple']
    ]
    df_time = pd.DataFrame({'时间 (小时)': t_hours})
    for C, color in zip(scenario_caps, scenario_colors):
        P_inst = g_safe(t_hours) * C
        df_time[f'{C} kW'] = pchip(P_inst)

    # 导出 Excel
    with pd.ExcelWriter(excel_path) as writer:
        df_tot.to_excel(writer, sheet_name='总期望损失', index=False)
        df_time.to_excel(writer, sheet_name='时序期望损失', index=False)
    print(f"结果已导出到: {excel_path}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    markers = ['o', '^', 's', 'p']  # 圆形、三角形、正方形、五角星
    for i, C in enumerate(scenario_caps):
        ax.plot(t_hours, df_time[f'{C} kW'], '-', color=scenario_colors[i],
                linewidth=2, marker=markers[i], markersize=6, markeredgewidth=1,
                markerfacecolor=scenario_colors[i], markeredgecolor=scenario_colors[i],
                label=f'{C} kW')
    ax.set_xlabel('时间(h)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.set_ylabel('系统风险', fontsize=14, fontweight='bold', fontproperties=font)
    ax.legend(fontsize=11, loc='best', prop=font)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('时序期望损失曲线.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

# ------------------- 主程序入口 -------------------
if __name__ == '__main__':
    folder = str((HERE.parent /"第二问"/ "问题二结果_预先运行的结果").resolve())
    # 热图分析
    plot_heatmap(folder)
    # 时序期望损失分析
    pchip = load_or_compute_pchip(folder)
    plot_and_export(pchip, str((HERE.parent /"risk_vs_capacity_and_time.xlsx").resolve()))
