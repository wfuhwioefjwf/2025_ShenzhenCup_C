# -*- coding: utf-8 -*-
import os, pickle, random, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # 导入SciencePlots库
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, LinearConstraint
from matplotlib.font_manager import FontProperties
from pathlib import Path
HERE = Path(__file__).resolve().parent

# 设置中文字体
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)
# ===================== 可调参数 =====================
N_SAMPLE = 2000
TOP_K = 10
RHO_PAIR = (0.3, 0.05)
MAXITER = 30000
N_RESTARTS = 2
SMOOTH_S = 1e-3
SEED = 2025
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'text.usetex': False,
})
plt.rcParams["axes.unicode_minus"] = False

random.seed(SEED)
np.random.seed(SEED)
CACHE_FILE1 = str((HERE.parent.parent / "pchip_cache.pkl").resolve())
CACHE_FILE2 = str((HERE.parent.parent / "pchie_cache.pkl").resolve())

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


def load_or_compute_spline(folder, cache_file, s=SMOOTH_S):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f: return pickle.load(f)
    sp = analyze_risk(folder, s)
    pickle.dump(sp, open(cache_file, "wb"))
    return sp


def random_feasible_z(lo, up):
    z = np.empty(12)
    cum = 0.0
    for i in range(12):
        hi = min(up[i], -cum)
        lo_i = lo[i]
        if hi < lo_i: hi = lo_i
        z[i] = np.random.uniform(lo_i, hi)
        cum += z[i]
    return z


def solve_global(spline, M):
    Gp = np.array([0.29, 3.10, 7.19, 10.50, 13.94, 11.14,
                   16.41, 10.14, 10.93, 8.24, 5.65, 2.05])
    c = (Gp / Gp.max()) * M
    Delta = 0.15 * M
    lo_x = np.maximum(-c, -Delta)
    up_x = np.full_like(c, Delta)
    lo_z, up_z = lo_x / Delta, up_x / Delta

    def obj_z(z):
        return float(np.sum(spline(c + z * Delta)))

    A = np.tril(np.ones((12, 12))) * Delta
    prefix = LinearConstraint(A, -np.inf, 0)
    I = np.eye(12)
    bound_hi = LinearConstraint(I, -np.inf, up_z)
    bound_lo = LinearConstraint(-I, -np.inf, -lo_z)
    cons = [prefix, bound_hi, bound_lo]
    cand = []
    for _ in range(N_SAMPLE):
        z = random_feasible_z(lo_z, up_z)
        cand.append((obj_z(z), z))
    cand.sort(key=lambda t: t[0])
    starts = [z for _, z in cand[:TOP_K]]
    best_R, best_z = np.inf, None
    for idx, z0 in enumerate(starts, 1):
        z_cur = z0.copy()
        ok = True
        for rho in RHO_PAIR:
            try:
                res = minimize(obj_z, z_cur, method="COBYLA",
                               constraints=cons,
                               options=dict(rhobeg=rho,
                                            maxiter=MAXITER // len(RHO_PAIR),
                                            tol=1e-6, disp=False))
                if not res.success: raise RuntimeError(res.message)
                z_cur = res.x
            except Exception:
                ok = False
                break
        if ok:
            R = obj_z(z_cur)
            if R < best_R:
                best_R, best_z = R, z_cur
    tries = 0
    while (best_z is None) and tries < N_RESTARTS:
        tries += 1
        z0 = random_feasible_z(lo_z, up_z)
        try:
            res = minimize(obj_z, z0, method="COBYLA",
                           constraints=cons,
                           options=dict(rhobeg=0.2, maxiter=MAXITER, tol=1e-6))
            if res.success:
                best_R, best_z = obj_z(res.x), res.x
        except Exception:
            pass

    if best_z is None:
        raise RuntimeError("全部启动点均未收敛")
    return best_R, best_z * Delta


if __name__ == "__main__":
    print(f"[平台] {platform.system()}  Python {platform.python_version()}  SciPy ≥1.9 推荐")
    # 路径一：决策变量全为0
    data_folder_1 = str((HERE.parent.parent /"第二问"/ "问题二结果_预先运行的结果").resolve())
    spline1 = load_or_compute_spline(data_folder_1, CACHE_FILE1)
    Ms = np.arange(30, 901, 10)
    risks_0 = []
    Gp = np.array([0.29, 3.10, 7.19, 10.50, 13.94, 11.14,
                   16.41, 10.14, 10.93, 8.24, 5.65, 2.05])
    for M in Ms:
        c = (Gp / Gp.max()) * M
        total_risk = np.sum(spline1(c))
        risks_0.append(total_risk)
    risks_0_plot = [r / 10000.0 for r in risks_0]

    # 路径二：COBYLA优化
    data_folder_2 = str((HERE.parent.parent /"第二问"/ "问题二结果_预先运行的结果").resolve())
    spline2 = load_or_compute_spline(data_folder_2, CACHE_FILE2)
    risks_opt = []
    for M in Ms:
        try:
            R, x = solve_global(spline2, M)
            risks_opt.append(R)
        except Exception as e:
            risks_opt.append(np.nan)
    risks_opt_plot = [r / 10000.0 for r in risks_opt]

    # ---- 绘图 ----
    DEEP_BLUE_PURPLE = '#483D8B'
    DARK_YELLOW_GREEN = '#A259BF'
    SHADOW_COLOR = (166/255, 166/255, 237/255, 0.3)

    plt.figure(figsize=(8, 4), dpi=150)
    # 添加阴影：fill_between 两条线
    plt.fill_between(Ms, risks_opt_plot, risks_0_plot,
                     where=(np.array(risks_0_plot) > np.array(risks_opt_plot)),
                     color=SHADOW_COLOR, alpha=0.3, label='优化带来的风险降低')

    plt.plot(Ms, risks_0_plot, marker='o', markersize=0,
             color=DEEP_BLUE_PURPLE, linestyle='-', linewidth=1.5, label='无储能电池')
    plt.plot(Ms, risks_opt_plot, marker='s', markersize=0,
             color=DARK_YELLOW_GREEN, linestyle='-', linewidth=1.5, label='有储能电池')
    plt.xlabel("光伏最大接入容量(kW)", fontsize=14, fontproperties=font)
    plt.ylabel("日总系统风险", fontsize=14, fontproperties=font)
    plt.ylim(85, 115)
    plt.grid(alpha=.4, linestyle="--")
    plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.legend(fontsize=11, prop=font)
    plt.savefig('COBYLA双曲线对比_阴影.pdf', bbox_inches='tight', facecolor='white')
    plt.show()