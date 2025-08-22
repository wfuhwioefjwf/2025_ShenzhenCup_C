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

# 初始化网络（修改init_worker，增加蒙特卡洛模拟传递的参数）
def init_worker(base_net, p_cond, prod_safe, events, cdf, total_prob, rng_seed=2025):
    global op, P_COND, PROD_SAFE, EVENTS, CDF, TOTAL_PROB, RNG, TEMPLATE_G
    
    TEMPLATE_G = copy.deepcopy(base_net)
    op = NetworkOperator(base_net)
    P_COND = p_cond
    PROD_SAFE = prod_safe

    EVENTS = events
    CDF = cdf
    TOTAL_PROB = total_prob
    
    # 为每个进程设置不同但确定的种子
    import os
    process_id = os.getpid()
    seed = rng_seed + process_id % 1000  # 使用进程ID的后3位避免种子过大
    RNG = np.random.default_rng(seed)

# simulate_single：simulate_fault接口函数
def simulate_single(loc):
    _reset_graph()

    ftype = op._determine_fault_type(loc)
    lost, over, tot, strat, overload = op.simulate_fault(
        ftype, loc, reset=False)
    w = P_COND[loc]      # 计算概率         
    return loc, lost, over, tot, strat, overload, w

# simulate_double：simulate_two_faults接口函数
def simulate_double(pair):
    loc1, loc2 = pair
    _reset_graph()

    lost, over, tot, strat = op.simulate_two_faults(loc1, loc2)
    w = P_COND[loc1] * P_COND[loc2] / PROD_SAFE     # 计算概率  
    return loc1, loc2, lost, over, tot, strat, w

# 重置网络
def _reset_graph():
    g = op.G
    for n, d0 in TEMPLATE_G.nodes(data=True):
        d  = g.nodes[n]
        d.clear();   d.update(d0)          
    g.remove_edges_from(list(g.edges))      
    g.add_edges_from(TEMPLATE_G.edges(data=True))

def _draw():
    r = RNG.random()  # 直接使用进程本地RNG，避免重复初始化
    idx = np.searchsorted(CDF, r * TOTAL_PROB, side='right')  # 乘以总概率缩放
    return EVENTS[min(idx, len(EVENTS)-1)]

# _dummy 只是一个占位计数（imap_unordered 必须有“任务”参数）
# 每调用一次就抽1个随机场景并返回3个标量：lost, over_penalty, total
def run_scenario(_dummy):
    s = _draw()
    if isinstance(s, tuple):                 # 双故障
        loc1, loc2 = s
        _, _, lost, over, tot, *_ = simulate_double((loc1, loc2))
    else:                                    # 单故障
        loc = s
        _, lost, over, tot, *_ = simulate_single(loc)
    return lost, over, tot

# ============ 用真实概率抽样的 Monte-Carlo 主程序 =============== 
if __name__ == '__main__':
    import openpyxl,time
    from openpyxl import Workbook
    mp.set_start_method('spawn', force=True)
    
    # 设置主进程的随机种子（可选，用于主程序中的随机操作）
    np.random.seed(2025)

    CAP_SEQ        = [300]        # DG容量固定为 300 KW
    TOTAL_SAMPLES  = 8000       # Monte-Carlo 总样本数
    START_RECORD   = 5           # 从第几次样本开始统计
    RECORD_STEP    = 100          # 每隔多少次写一次 Excel
    RNG_SEED       = 2026
    CHUNK          = 30          # 进程池 chunksize
    OUT_DIR        = str((HERE).resolve())
    cpu_num        = max(1, mp.cpu_count() - 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    # 定义 Excel 工具函数
    def init_workbook(path):
        wb = Workbook()
        ws = wb.active
        ws.title = 'MC_summary'
        ws.append(["samples", "E_loss", "E_over", "E_risk"])
        wb.save(path)
    def append_row(path, row):
        wb = openpyxl.load_workbook(path)
        ws = wb['MC_summary']
        ws.append(row)
        wb.save(path)
    
    for cap in CAP_SEQ:
        print(f"\n===== Monte-Carlo 评估 {cap} kW × {TOTAL_SAMPLES} =====")
        t_start = time.perf_counter()
        # 生成基准网络
        G_cap = copy.deepcopy(G)
        dg_nodes = [n for n, d in G_cap.nodes(data=True) if d['type'] == '分布式能源']
        for dg in dg_nodes:
            G_cap.nodes[dg]['容量(kW)'] = cap
        op0 = NetworkOperator(G_cap)
        BASE_NET = copy.deepcopy(op0.G)
        print(f"[{time.strftime('%H:%M:%S')}] 基准网络生成完毕 ({time.perf_counter()-t_start:.1f}s)")

        # 计算概率
        p_fail = {}
        for n, d in BASE_NET.nodes(data=True):
            t = d['type']
            if t == '导线':
                p_fail[n] = d['长度(km)'] * 0.002  
            elif t in ('分段开关', '联络开关', '断路器'): 
                p_fail[n] = 0.002
            elif t in ('用户', '分布式能源'):
                p_fail[n] = 0.005
            else:
                p_fail[n] = 0.0
        p_safe = {n: 1 - p for n, p in p_fail.items()}
        t_start = time.perf_counter()
        all_faults = (
              [f"L{i}"  for i in range(1, 60)]
            + [f"U{i}"  for i in range(1, 63)]
            + [f"S{i}"  for i in range(1, 30)] + ["S13-1", "S29-2", "S62-3"]
            + ["CB1", "CB2", "CB3"]
            + [f"DG{i}" for i in range(1, 9)]
        )
        single_args = [x for x in all_faults if x in BASE_NET.nodes]
        double_args = [
            (a, b) for a, b in combinations(all_faults, 2)
            if device_category(a) != device_category(b)
            and a in BASE_NET.nodes and b in BASE_NET.nodes
        ]
        print(f"[{time.strftime('%H:%M:%S')}] 故障列表生成完毕: {len(single_args)}单/{len(double_args)}双 ({time.perf_counter()-t_start:.1f}s)")

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
        
        # 更新全网安全概率为所有类别不发生故障概率的乘积
        prod_safe_all = np.prod(list(category_no_fault_probs.values()))
        
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
        
        # 采样权重 = 真实概率
        t_start = time.perf_counter()
        # 单故障
        event_single  = single_args
        w_single      = np.array([p_cond[x] for x in event_single])
        # 双故障
        event_double  = double_args
        w_double      = np.array([
            p_cond[a] * p_cond[b] / prod_safe_all
            for a, b in event_double
        ])

        events   = event_single + event_double
        weights  = np.concatenate([w_single, w_double])
        cdf      = np.cumsum(weights)     # 累计权重
        total_prob = cdf[-1]              # 总概率
        print(f"[{time.strftime('%H:%M:%S')}] 权重计算完毕: 总概率={total_prob:.4f} ({time.perf_counter()-t_start:.1f}s)")

        # 写入Excel
        xlsx_path = os.path.join(OUT_DIR, f"MonteCarlo_{cap}kW.xlsx")
        if os.path.exists(xlsx_path):
            os.remove(xlsx_path)
        init_workbook(xlsx_path)

        # 进程池
        print(f"[{time.strftime('%H:%M:%S')}] 启动 {cpu_num} 个 worker...")
        with mp.Pool(
            cpu_num,
            initializer=init_worker,
            initargs=(BASE_NET, p_cond, prod_safe_all, events, cdf, total_prob, RNG_SEED)
        ) as pool:
            print(f"[{time.strftime('%H:%M:%S')}] workers ready.")

            cum_loss = cum_over = cum_tot = 0.0
            done     = 0
            t_start  = time.perf_counter()

            for lost, over, tot in pool.imap_unordered(
                    run_scenario,
                    range(TOTAL_SAMPLES),
                    chunksize=CHUNK):

                done      += 1
                cum_loss  += lost
                cum_over  += over
                cum_tot   += tot

                if done >= START_RECORD and (done - START_RECORD) % RECORD_STEP == 0:
                    row = [done,
                        cum_loss / done * total_prob,
                        cum_over / done * total_prob,
                        cum_tot  / done * total_prob]
                    
                    speed = done / (time.perf_counter() - t_start)
                    eta = (TOTAL_SAMPLES - done) / speed if speed > 0 else 0
                    
                    print(f"MC({done:>6}/{TOTAL_SAMPLES})  "
                        f"E(loss)={row[1]:,.2f}  "
                        f"E(over)={row[2]:,.2f}  "
                        f"E(risk)={row[3]:,.2f}  "
                        f"速度: {speed:.1f}样本/秒  "
                        f"预计剩余: {eta/60:.1f}分钟",
                        flush=True)

                    append_row(xlsx_path, row)

        print(f"模拟完成，统计已写入：{xlsx_path}  "
              f"(从 {START_RECORD} 开始，每 {RECORD_STEP} 条记录一次)")