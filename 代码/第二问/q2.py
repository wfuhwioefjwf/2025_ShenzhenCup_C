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
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from BASE_用于构建基础网络和封装故障处理类 import NetworkOperator, device_category, G

# 初始化网络
def init_worker(base_net, p_cond, prod_safe):
    global TEMPLATE_G, op, P_COND, PROD_SAFE
    TEMPLATE_G   = base_net
    op           = NetworkOperator(base_net)
    P_COND       = p_cond          
    PROD_SAFE    = prod_safe       

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

# ============ 主程序 ===============
if __name__ == '__main__':
    CAP_SEQ       = [i for i in range(30, 901, 30)]
    PROGRESS_INT  = 100
    BASE_XLSX_DIR = str((HERE / "问题二结果" ).resolve())
    
    # 创建问题二结果文件夹（如果不存在）
    os.makedirs(BASE_XLSX_DIR, exist_ok=True)
    print(f"结果文件夹已准备: {BASE_XLSX_DIR}")

    dg_nodes = [n for n, d in G.nodes(data=True)
                if d.get('type') == '分布式能源']
    cpu_num  = max(1, mp.cpu_count() - 1)

    for idx, cap in enumerate(CAP_SEQ, 1):
        print(f"\n===== 评估 {cap:.0f} kW ({idx}/{len(CAP_SEQ)}) =====")

        # 生成一次已初始化好的基准网络BASE_NET
        G_cap = copy.deepcopy(G)
        for dg in dg_nodes:
            G_cap.nodes[dg]['容量(kW)'] = cap
        op0 = NetworkOperator(G_cap)
        BASE_NET = copy.deepcopy(op0.G)

        # 初始故障概率
        p_fail = {}
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
        p_safe = {n: 1.0 - p for n, p in p_fail.items()}

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

        # 准备Excel文件和写入器
        fn   = f"结果_{int(cap)}kW.xlsx"
        path = os.path.join(BASE_XLSX_DIR, fn)
        os.makedirs(os.path.dirname(path), exist_ok=True)
                # 创建Excel文件并写入表头
        import openpyxl
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = f"cap{cap}"
        headers = [
            "capacity", "type", "loc", "best_strategy", 
            "lost", "over_penalty", "tot", "overload_lines", "w",
            "expected_loss", "expected_over", "expected_risk"
        ]
        sheet.append(headers)
        workbook.save(path)
        
        # 初始化累加器
        sum_risk = 0.0
        sum_loss = 0.0
        sum_over = 0.0
        row_count = 1  # 表头已占1行

        # 定义写入行的函数（替代方案）
        def write_row(data, sheet, workbook, path, accumulators):
            sum_risk, sum_loss, sum_over, row_count = accumulators
            
            # 更新累加器
            weight = data["w"]
            sum_risk += data["tot"] * weight
            sum_loss += data["lost"] * weight
            sum_over += data["over_penalty"] * weight
            
            # 创建行数据
            row = [
                data["capacity"],
                data["type"],
                data["loc"],
                data["best_strategy"],
                data["lost"],
                data["over_penalty"],
                data["tot"],
                data["overload_lines"],
                weight,
                sum_loss,
                sum_over,
                sum_risk
            ]
            
            # 写入行
            sheet.append(row)
            row_count += 1
            
            # 定期保存
            if row_count % PROGRESS_INT == 0:
                workbook.save(path)
                print(f"  已保存 {row_count} 行结果")
            
            return (sum_risk, sum_loss, sum_over, row_count)

        # 并行计算
        single_args = [loc for loc in all_faults if loc in op0.G.nodes]
        double_args = [
            (a, b) for a, b in combinations(all_faults, 2)
            if device_category(a) != device_category(b)
            and a in op0.G.nodes and b in op0.G.nodes
        ]

        with mp.Pool(processes=cpu_num,
                     initializer=init_worker,
                     initargs=(op0.G, p_cond, prod_safe_all)) as pool:

            print(f"启动任务：{len(single_args)} 单故障 + {len(double_args)} 双故障")

            # 单故障计算
            cnt = 0
            for result in pool.imap_unordered(simulate_single, single_args, chunksize=50):
                loc, lost, over, tot, strat, overload, w = result
                cnt += 1
                
                if cnt % PROGRESS_INT == 0:
                    print(f"  单故障进度 {cnt}/{len(single_args)}")
                
                accumulators = write_row({
                    "capacity": cap,
                    "type": "single",
                    "loc": loc,
                    "best_strategy": strat,
                    "lost": lost,
                    "over_penalty": over,
                    "tot": tot,
                    "overload_lines": ";".join(
                        f"{bid}:{I:.1f}A" for bid, I in overload
                    ),
                    "w": w
                }, sheet, workbook, path, (sum_risk, sum_loss, sum_over, row_count))
                sum_risk, sum_loss, sum_over, row_count = accumulators

                if cnt % PROGRESS_INT == 0:
                    print(f"[{cap} kW] E(loss)={sum_loss:,.2f} E(over)={sum_over:,.2f} "f"E(risk)={sum_risk:,.2f}")

            # 双故障计算
            cnt = 0
            for result in pool.imap_unordered(simulate_double, double_args, chunksize=30):
                loc1, loc2, lost, over, tot, strat, w = result
                cnt += 1
                
                if cnt % PROGRESS_INT == 0:
                    print(f"  双故障进度 {cnt}/{len(double_args)}")
                
                accumulators = write_row({
                    "capacity": cap,
                    "type": "double",
                    "loc": f"{loc1}+{loc2}",
                    "best_strategy": strat,
                    "lost": lost,
                    "over_penalty": over,
                    "tot": tot,
                    "overload_lines": "-",
                    "w": w
                }, sheet, workbook, path, (sum_risk, sum_loss, sum_over, row_count))
                sum_risk, sum_loss, sum_over, row_count = accumulators
                
                if cnt % PROGRESS_INT == 0:
                    print(f"[{cap} kW] E(loss)={sum_loss:,.2f} E(over)={sum_over:,.2f} "f"E(risk)={sum_risk:,.2f}")

        # 最终保存并关闭
        workbook.save(path)
        workbook.close()

        print(f"[{cap} kW] E(loss)={sum_loss:,.2f} E(over)={sum_over:,.2f} "
              f"E(risk)={sum_risk:,.2f} → 已写入 {fn}")