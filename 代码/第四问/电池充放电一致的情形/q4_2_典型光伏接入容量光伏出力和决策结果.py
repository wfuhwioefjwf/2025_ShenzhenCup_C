# -*- coding: utf-8 -*-
import os, pickle, random, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, LinearConstraint
import scienceplots
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
from pathlib import Path
HERE = Path(__file__).resolve().parent

font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)

# ========= 参数 =========
M_TARGETS = [100.0, 300.0, 500.0, 700.0]   # 一次性计算 4 个容量
N_SAMPLE = 2000
TOP_K = 10
RHO_PAIR = (0.3, 0.05)
MAXITER = 30000
N_RESTARTS = 2
SMOOTH_S = 1e-3
SEED = 205

# 科学绘图样式（与原版一致）
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

# 与附件一致的颜色表（容量: 100/300/500/700）
COLORS = {
    'deep_blue':   '#4DABD1',  # 100 kW
    'gold_yellow': '#B7950B',  # 300 kW
    'cyan':        '#6F9C71',  # 500 kW
    'deep_purple': '#4A5490',  # 700 kW
}
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)

# 固定顺序与标记（与附件一致）
scenario_caps    = [100, 300, 500, 700]
scenario_colors  = [COLORS['deep_blue'], COLORS['gold_yellow'],
                    COLORS['cyan'],      COLORS['deep_purple']]
scenario_markers = ['o', '^', 's', 'p']  # 圆、三角、方块、五角星

random.seed(SEED)
np.random.seed(SEED)
CACHE_FILE = str((HERE.parent.parent / "spline_cache.pkl").resolve())

def analyze_risk(folder, s=SMOOTH_S):
    rec = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".xlsx"):
            df = pd.read_excel(os.path.join(folder, fn))
            rec.append({"capacity": float(df["capacity"][0]),
                        "total_loss": float(df["expected_risk"].iloc[-1])})
    if not rec:
        raise RuntimeError("未找到 xlsx 数据")
    df = pd.DataFrame(rec).sort_values("capacity")
    return UnivariateSpline(df["capacity"], df["total_loss"], k=3, s=s)

def load_or_compute_spline(folder):
    if os.path.exists(CACHE_FILE):
        return pickle.load(open(CACHE_FILE, "rb"))
    sp = analyze_risk(folder)
    pickle.dump(sp, open(CACHE_FILE, "wb"))
    return sp

def random_feasible_z(lo, up):
    """生成满足逐段前缀和≤0、且逐段盒约束的可行 z（长度=12）"""
    z = np.empty(12)
    cum = 0.0
    for i in range(12):
        hi = min(up[i], -cum)
        lo_i = lo[i]
        if hi < lo_i:
            hi = lo_i
        z[i] = np.random.uniform(lo_i, hi)
        cum += z[i]
    return z

def solve_robust_for_target(M_target, spline):
    """把原先的 solve_robust_300 泛化为任意容量 M_target 的稳健优化."""
    # 与原脚本一致的基准出力形状（6~18 点，共 12 段）
    Gp = np.array([0.29, 3.10, 7.19, 10.50, 13.94, 11.14,
                   16.41, 10.14, 10.93, 8.24, 5.65, 2.05])
    c = (Gp / Gp.max()) * M_target
    Delta = 0.15 * M_target     # 盒半径随容量线性缩放
    lo_x = np.maximum(-c, -Delta)
    up_x = np.full_like(c, Delta)
    lo_z, up_z = lo_x / Delta, up_x / Delta

    def obj_z(z):
        # 目标：∑ R(c + x)；其中 x = z * Delta
        return float(np.sum(spline(c + z * Delta)))

    # 线性约束：前缀和 ≤ 0 以及逐段盒约束（与原脚本一致）
    A = np.tril(np.ones((12, 12))) * Delta
    prefix = LinearConstraint(A, -np.inf, 0)
    I = np.eye(12)
    bound_hi = LinearConstraint(I, -np.inf, up_z)
    bound_lo = LinearConstraint(-I, -np.inf, -lo_z)
    cons = [prefix, bound_hi, bound_lo]

    # 1) 随机粗搜
    cand = []
    for _ in range(N_SAMPLE):
        z = random_feasible_z(lo_z, up_z)
        cand.append((obj_z(z), z))
    cand.sort(key=lambda t: t[0])
    starts = [z for _, z in cand[:TOP_K]]

    # 2) 双阶段 COBYLA 局部精化
    best_R = np.inf
    best_z = None
    for z0 in starts:
        z_cur = z0.copy()
        ok = True
        for rho in RHO_PAIR:
            try:
                res = minimize(obj_z, z_cur, method="COBYLA",
                               constraints=cons,
                               options=dict(rhobeg=rho,
                                            maxiter=MAXITER // len(RHO_PAIR),
                                            tol=1e-6, disp=False))
                if not res.success:
                    raise RuntimeError
                z_cur = res.x
            except Exception:
                ok = False
                break
        if ok:
            val = obj_z(z_cur)
            if val < best_R:
                best_R, best_z = val, z_cur

    # 3) 随机重启兜底
    tries = 0
    while best_z is None and tries < N_RESTARTS:
        tries += 1
        z0 = random_feasible_z(lo_z, up_z)
        try:
            res = minimize(obj_z, z0, method="COBYLA",
                           constraints=cons,
                           options=dict(rhobeg=0.2, maxiter=MAXITER, tol=1e-6))
            if res.success and obj_z(res.x) < best_R:
                best_R, best_z = obj_z(res.x), res.x
        except Exception:
            pass

    if best_z is None:
        raise RuntimeError(f"{M_target} kW: 全部启动点均未收敛")

    x_opt = best_z * Delta          # 决策变量（kW）
    c_plus_x = c + x_opt            # 光伏出力调整后（kW）
    return x_opt, c_plus_x, best_R

if __name__ == "__main__":
    print(f"[平台] {platform.system()}  Python {platform.python_version()}  SciPy ≥1.9 推荐")
    folder = str((HERE.parent.parent /"第二问"/ "问题二结果_预先运行的结果").resolve())
    spline = load_or_compute_spline(folder)

    # 计算四个容量
    results = {}   # M -> dict(x, cpx, R)
    for M in M_TARGETS:
        x_opt, cpx, R_min = solve_robust_for_target(M, spline)
        results[M] = dict(x=x_opt, cpx=cpx, R=R_min)
        print(f"\n【{int(M)} kW】最优目标 ∑R(c+x) = {R_min:.4f}")
        print("最优 x =", np.round(x_opt, 3))

    # === 绘图（两张） ===
    hours = np.arange(6, 19)  # 6~18 共 13 个点

    # 图1：光伏出力 c_i + x_i（每个容量一条曲线）
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    all_cpx_plots = []

    # 与附件一致的线宽/点样式/网格/图例标签
    for i, (C, color, mk) in enumerate(zip(scenario_caps, scenario_colors, scenario_markers)):
        cpx_plot = np.concatenate([results[C]["cpx"], [0.0]])  # 原逻辑：末尾补0
        all_cpx_plots.append(cpx_plot)
        ax.plot(hours, cpx_plot, '-', color=color,
                linewidth=2, marker=mk, markersize=6, markeredgewidth=1,
                markerfacecolor=color, markeredgecolor=color,
                label=f"{C} kW")

    ax.set_xlabel('时间(h)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.set_ylabel('出力(kW)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=11, prop=font)  # 不再加 legend 标题，保持附件风格
    ax.set_xlim(5.5, 18.5)

    # （保留你原先的 y 轴范围逻辑）
    ymin = min([p.min() for p in all_cpx_plots])
    ymax = max([p.max() for p in all_cpx_plots])
    ax.set_ylim(ymin * 0.90 - 10, ymax * 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig('光伏出力_c_i+x_i_多容量.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

    # 图2：决策变量 x_i（每个容量一条曲线）
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    all_x_plots = []

    for i, (C, color, mk) in enumerate(zip(scenario_caps, scenario_colors, scenario_markers)):
        x_plot = np.concatenate([results[C]["x"], [0.0]])  # 原逻辑：末尾补0
        all_x_plots.append(x_plot)
        ax.plot(hours, x_plot, '-', color=color,
                linewidth=2, marker=mk, markersize=6, markeredgewidth=1,
                markerfacecolor=color, markeredgecolor=color,
                label=f"{C} kW")

    ax.set_xlabel('时间(h)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.set_ylabel('$x_i$(kW)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=11, prop=font)  # 与附件一致
    ax.set_xlim(5.5, 18.5)

    # （保留你原先的 y 轴范围逻辑）
    xmin = min([p.min() for p in all_x_plots])
    xmax = max([p.max() for p in all_x_plots])
    ax.set_ylim(xmin * 1.1, xmax * 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig('决策变量_x_i_多容量.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

