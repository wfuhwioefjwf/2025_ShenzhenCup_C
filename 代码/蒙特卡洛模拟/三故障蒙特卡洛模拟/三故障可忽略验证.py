#-*- coding: utf-8 -*- 

import math
from numba import njit
from collections import deque
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
import multiprocessing as mp
import os, copy

from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from BASE_用于构建基础网络和封装故障处理类 import NetworkOperator, device_category, G

# 初始化网络（修改init_worker，添加按概率抽样所需参数）
def init_worker(global_base_net, global_p_cond, global_P0, global_events, global_cdf, global_total_prob):
    global BASE_NET, op, p_cond, P0, EVENTS, CDF, TOTAL_PROB, RNG
    BASE_NET = global_base_net
    op       = NetworkOperator(copy.deepcopy(BASE_NET))
    p_cond   = global_p_cond
    P0       = global_P0
    EVENTS   = global_events
    CDF      = global_cdf
    TOTAL_PROB = global_total_prob
    RNG      = np.random.default_rng()

# 按概率抽样函数
def _draw_triple():
    """按概率抽样一个三故障组合"""
    r = RNG.random()  # 生成0-1之间的随机数
    idx = np.searchsorted(CDF, r * TOTAL_PROB, side='right')  # 根据CDF查找对应事件
    return EVENTS[min(idx, len(EVENTS)-1)]

# simulate_triple：simulate_three_faults接口函数
def simulate_triple(triple):
    loc1, loc2, loc3 = triple
    op.G = copy.deepcopy(BASE_NET)
    op.original_G = BASE_NET

    lost, over, tot, strat = op.simulate_three_faults(loc1, loc2, loc3)
    w = (p_cond[loc1] * p_cond[loc2] * p_cond[loc3]/(P0 * P0))
    return loc1, loc2, loc3, lost, over, tot, strat, w

# 蒙特卡洛抽样函数（替代原来的simulate_triple）
def run_triple_scenario(_dummy):
    """按概率抽样并运行一个三故障场景"""
    triple = _draw_triple()
    loc1, loc2, loc3, lost, over, tot, strat, w = simulate_triple(triple)
    return lost, over, tot

# ============ 三故障可忽略验证（按概率抽样）主程序 =============== 
if __name__ == '__main__':
    import time
    
    CAP_SEQ        = [300]          # 固定DG容量为300KW
    TRIPLE_SAMPLE  = 5000           # 按概率抽样 5000 个三故障
    PROGRESS_INT   = 100
    BASE_XLSX_PATH = str((HERE).resolve())
    os.makedirs(BASE_XLSX_PATH, exist_ok=True)

    dg_nodes = [n for n, d in G.nodes(data=True)
                if d.get('type') == '分布式能源']
    cpu_num = max(1, mp.cpu_count() - 1)

    for idx, cap in enumerate(CAP_SEQ, 1):
        print(f"\n===== 评估 {cap:.0f} kW ({idx}/{len(CAP_SEQ)}) - 按概率抽样 =====")
        t_start = time.perf_counter()

        # 生成一次已初始化好的基准网络BASE_NET
        G_cap = copy.deepcopy(G)
        for dg in dg_nodes:
            G_cap.nodes[dg]['容量(kW)'] = cap
        op0 = NetworkOperator(G_cap)
        BASE_NET = copy.deepcopy(op0.G)

        # 计算概率
        p_fail, p_safe = {}, {}
        for n, d in op0.G.nodes(data=True):
            typ = d['type']
            if typ == '导线':
                p_fail[n] = d['长度(km)'] * 0.002
            elif typ in ('用户', '分布式能源'):
                p_fail[n] = 0.005
            elif typ in ('分段开关', '联络开关', '断路器'):
                p_fail[n] = 0.002
            else:
                p_fail[n] = 0.0
            p_safe[n] = 1.0 - p_fail[n]

        print(f"[{time.strftime('%H:%M:%S')}] 基准网络生成完毕 ({time.perf_counter()-t_start:.1f}s)")

        # 故障全集：三组合
        all_faults = (
              [f"L{i}"  for i in range(1, 60)]
            + [f"U{i}"  for i in range(1, 63)]
            + [f"S{i}"  for i in range(1, 30)] + ["S13-1", "S29-2", "S62-3"]
            + ["CB1", "CB2", "CB3"]
            + [f"DG{i}" for i in range(1, 9)]
        )

        # 根据"同类故障最多发生一个"的约束条件计算条件概率
        # 按设备类型分组
        devices_by_category = {}
        for j in all_faults:
            if j in op0.G.nodes:
                category = device_category(j)
                if category not in devices_by_category:
                    devices_by_category[category] = []
                devices_by_category[category].append(j)
        # 计算每个类别的概率字典
        category_probs = {}
        for category, devices in devices_by_category.items():
            # 该类别全不发生故障的概率
            prob_no_fault = np.prod([p_safe[d] for d in devices])
            # 该类别中只发生一个故障的概率
            prob_single_fault = sum(
                p_fail[d] * np.prod([p_safe[other] for other in devices if other != d])
                for d in devices
            )
            category_probs[category] = {
                'no_fault': prob_no_fault,
                'single_fault': prob_single_fault,
                'devices': devices
            }
        
        # 计算条件概率字典
        p_cond_category = {}
        for j in all_faults:
            if j in op0.G.nodes:
                category = device_category(j)
                cat_info = category_probs[category]
                # 该位置发生故障的初始概率
                p_j_fail = p_fail[j]
                # 该类其他位置不发生故障的初始概率
                other_devices = [d for d in cat_info['devices'] if d != j]
                p_others_safe = np.prod([p_safe[d] for d in other_devices])
                # 分母：该类别全不发生故障概率 + 该类别中只发生一个故障的概率
                denominator = cat_info['no_fault'] + cat_info['single_fault']
                # 条件概率 = 该位置发生故障的初始概率 * 该类其他位置不发生故障的初始概率 / 分母
                p_cond_category[j] = (p_j_fail * p_others_safe) / denominator

        # 单故障全局条件概率
        # 计算各类型不发生故障的概率
        category_no_fault_probs = {}
        for category, cat_info in category_probs.items():
            category_no_fault_probs[category] = cat_info['no_fault'] / (cat_info['no_fault'] + cat_info['single_fault'])
        
        # 计算全网安全概率为所有类别不发生故障概率的乘积
        P0 = np.prod(list(category_no_fault_probs.values()))
        
        # 计算全局单故障条件概率
        p_cond = {}
        for j in all_faults:
            if j in op0.G.nodes:
                category = device_category(j)
                # 计算其他类别都不发生故障的概率
                other_cats_prob = 1.0
                for other_cat in category_no_fault_probs.keys():
                    if other_cat != category:
                        other_cats_prob *= category_no_fault_probs[other_cat]
                # 全局条件概率 = 该类型中的条件概率 * 其他类别都不发生故障的概率
                p_cond[j] = p_cond_category[j] * other_cats_prob

        # 生成所有合法的三故障组合
        t_start = time.perf_counter()
        triple_args_all = [
            (a, b, c)
            for a, b, c in combinations(all_faults, 3)
            if len({device_category(a), device_category(b), device_category(c)}) == 3
        ]
        print(f"[{time.strftime('%H:%M:%S')}] 三故障组合生成完毕: {len(triple_args_all)}组 ({time.perf_counter()-t_start:.1f}s)")

        # 计算每个三故障组合的概率权重
        t_start = time.perf_counter()
        weights = []
        for a, b, c in triple_args_all:
            w = (p_cond[a] * p_cond[b] * p_cond[c]/(P0 * P0))
            weights.append(w)
        
        weights = np.array(weights)
        cdf = np.cumsum(weights)  # 累积分布函数
        total_prob = cdf[-1]      # 总概率
        
        print(f"[{time.strftime('%H:%M:%S')}] 概率权重计算完毕: 总概率={total_prob:.6f} ({time.perf_counter()-t_start:.1f}s)")

        # 进程池初始化
        print(f"[{time.strftime('%H:%M:%S')}] 启动 {cpu_num} 个 worker...")
        
        # 蒙特卡洛抽样
        cum_loss = cum_over = cum_tot = 0.0
        done = 0
        t_start = time.perf_counter()

        with mp.Pool(
            processes=cpu_num,
            initializer=init_worker,
            initargs=(BASE_NET, p_cond, P0, triple_args_all, cdf, total_prob)
        ) as pool:
            
            print(f"[{time.strftime('%H:%M:%S')}] workers ready.")

            for lost, over, tot in pool.imap_unordered(
                    run_triple_scenario,
                    range(TRIPLE_SAMPLE),
                    chunksize=50):

                done += 1
                cum_loss += lost
                cum_over += over
                cum_tot += tot

                if done % PROGRESS_INT == 0:
                    # 计算当前期望值（乘以总概率进行缩放）
                    current_E_loss = cum_loss / done * total_prob
                    current_E_over = cum_over / done * total_prob
                    current_E_risk = cum_tot / done * total_prob
                    
                    speed = done / (time.perf_counter() - t_start)
                    eta = (TRIPLE_SAMPLE - done) / speed if speed > 0 else 0
                    
                    print(f"三故障MC({done:>6}/{TRIPLE_SAMPLE})  "
                          f"E(loss)={current_E_loss:,.6f}  "
                          f"E(over)={current_E_over:,.6f}  "
                          f"E(risk)={current_E_risk:,.6f}  "
                          f"速度: {speed:.1f}样本/秒  "
                          f"预计剩余: {eta/60:.1f}分钟",
                          flush=True)

        # 最终期望值计算
        E_loss = cum_loss / TRIPLE_SAMPLE * total_prob
        E_over = cum_over / TRIPLE_SAMPLE * total_prob
        E_risk = cum_tot / TRIPLE_SAMPLE * total_prob

        print(f"\n[{cap:>4} kW] 三故障按概率抽样期望值:")
        print(f"  E(loss) = {E_loss:.6f}")
        print(f"  E(over) = {E_over:.6f}")
        print(f"  E(risk) = {E_risk:.6f}")
        print(f"  总概率  = {total_prob:.6f}")

        # 保存结果到Excel
        results_data = {
            "capacity": [cap],
            "sampling_method": ["probability_sampling"],
            "total_samples": [TRIPLE_SAMPLE],
            "total_combinations": [len(triple_args_all)],
            "total_probability": [total_prob],
            "expected_loss": [E_loss],
            "expected_over": [E_over],
            "expected_risk": [E_risk]
        }
        
        df = pd.DataFrame(results_data)
        
        out_dir = BASE_XLSX_PATH
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"三故障_prob_{int(cap)}kW_test.xlsx")

        with pd.ExcelWriter(out_path, engine='xlsxwriter') as w:
            df.to_excel(w, sheet_name=f"{cap}kW_prob_sampling", index=False)

        print(f"按概率抽样结果已保存：{out_path}")
        print(f"模拟完成，用时: {(time.perf_counter() - t_start)/60:.1f}分钟")