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
from bisect import bisect_left
from pathlib import Path
HERE = Path(__file__).parent

# ============ 常量 ============

# === 段缓存目录（与预计算脚本一致） ===
SEG_CACHE_DIR = str((HERE / "缓存").resolve())
BASE_DIR = str((HERE).resolve())
os.makedirs(SEG_CACHE_DIR, exist_ok=True)

EXCEL = str((HERE.parent.parent / "C题附件：增加导线中间的开关和是否为干路两列.xlsx").resolve()) # 基本参数excel
V_BASE = 10e3        
I_RATE = RATED_CURRENT = 220    # 额定电流 单位A
DG_BASE = 300         # DG初始容量 单位kW
TYPE_W = {'居民':0.067,'商业':0.226,'政府和机构':0.354,'办公和建筑':0.354}   # 不同用户权重
main_switches = ['S1','S2','S3','S4','S5','S8','S27','S29','S11','S12'] # 干路开关表

# ==================== 初始图节点列表 ====================
# ==================== 出线开关列表 ====================
circuit_breakers = [
    {'名称': f'CB{i}', '故障状态': False} for i in range(1, 4)  # CB1~CB3
]
# ==================== 分段开关列表 ====================
segment_switches = [
    {'名称': f'S{i}',
     '开关状态': "开",  
     '故障状态': False}
    for i in range(1, 30)  # S1~S29
]
# ==================== 联络开关列表 ====================
tie_switches = [
    {'名称': 'S13-1', '开关状态': '关', '故障状态': False},
    {'名称': 'S29-2', '开关状态': '关', '故障状态': False},
    {'名称': 'S62-3', '开关状态': '关', '故障状态': False}
]
# ==================== 用户列表 ========================
# 类型字典
type_dict = {
    1: '居民',
    2: '商业',
    3: '政府和机构',
    4: '办公和建筑'
}
# 用户类型映射表
user_type_mapping = [
    # 用户1-10
    1, 1, 1, 1, 1, 1, 4, 1, 3, 1,
    # 用户11-20
    2, 4, 1, 4, 1, 2, 1, 4, 1, 1,
    # 用户21-30
    3, 1, 1, 1, 1, 1, 2, 1, 3, 1,
    # 用户31-40
    2, 4, 2, 2, 1, 3, 1, 2, 1, 4,
    # 用户41-50
    1, 2, 1, 1, 3, 1, 4, 1, 2, 1,
    # 用户51-60
    1, 1, 2, 1, 1, 2, 1, 3, 1, 1,
    # 用户61-62
    3, 1
]
users=pd.read_excel(EXCEL,sheet_name=0)
active_power=users['有功P/kW']
customers = [
    {'名称': f'U{i}',
     '类型': type_dict[user_type_mapping[i-1]],
     '用电功率(kW)': active_power[i-1],  
     '故障状态': False}
    for i in range(1, 63) # U1~U62
]
# ==================== 线路列表 =======================
df_topo = pd.read_excel(EXCEL, sheet_name=1, header=0,converters={'导线中间的开关': str,'是否是干路':str})   

# 创建导线编号到起点终点用户的字典
line_dict = {f'L{row["编号"]}': (f'U{row["起点"]}', f'U{row["终点"]}') for _, row in df_topo.iterrows()}

main_lines = df_topo[df_topo['是否是干路'] == 'True']['编号'].apply(lambda x: f'L{x}').tolist()   # 提取干路线路编号（根据"是否是干路"列的值判断）
# 构建完整线路参数
line_parameters = {}
for _, row in df_topo.iterrows():
    line_id   = int(row['编号'])
    length_km = float(row['长度/km'])
    R_raw     = float(row['电阻/Ω'])      
    R_total = R_raw                    
    line_parameters[line_id] = {
        '起点'  : int(row['起点']),
        '终点'  : int(row['终点']),
        '开关'  : str(row['导线中间的开关']),
        '长度(km)' : length_km,
        '总电阻(Ω)' : R_total,                         
        '总电抗(Ω)' : float(row['电抗/Ω']),           
        'main'  : str(row['是否是干路']) == 'True'
    }
lines = [
    {
        '名称'      : f'L{i}',
        '长度(km)'  : line_parameters[i]['长度(km)'],
        '总电阻(Ω)' : line_parameters[i]['总电阻(Ω)'],
        '总电抗(Ω)' : line_parameters[i]['总电抗(Ω)'],
        '总阻抗(Ω)' : complex(
            line_parameters[i]['总电阻(Ω)'],
            line_parameters[i]['总电抗(Ω)']
        ),
        '故障状态'  : False
    }
    for i in range(1, 60) # L1~L59
]

TIE_LINES = [
    ("LT_13_43", 220, 0.01),   # 对应 S13-1
    ("LT_19_29", 220, 0.01),   # 对应 S29-2
    ("LT_23_62", 220, 0.01),   # 对应 S62-3
]
for name, irate, R in TIE_LINES:
    lines.append({
        "名称"      : name,
        "长度(km)"  : 0.01,
        "总电阻(Ω)" : R,
        "总电抗(Ω)" : 0.0,
        "总阻抗(Ω)" : complex(R, 0.0),
        "额定电流(A)": irate,         
        "故障状态"  : False
    })
# ==================== 分布式电源列表 ====================
dgs = [
    {'名称': f'DG{i}',
     '容量(kW)': DG_BASE,  
     '开关状态': '关',
     '故障状态': False}
    for i in range(1, 9)  # DG1~DG8
]

# ==================== 创建网络图 ====================
G = nx.Graph()

# 添加用户节点
for user in customers:
    G.add_node(user['名称'], type='用户', **user)
# 添加导线节点
for line in lines:
    G.add_node(line['名称'], type='导线', **line)
# 添加分段开关
for segment_switch in segment_switches:
    G.add_node(segment_switch['名称'], type='分段开关', **segment_switch)
# 添加联络开关
for tie_switch in tie_switches:
    G.add_node(tie_switch['名称'], type='联络开关', **tie_switch)
# 添加断路器
for breaker in circuit_breakers:
    G.add_node(breaker['名称'], type='断路器', **breaker)
# 添加分布式能源
for dg in dgs:
    G.add_node(dg['名称'], type='分布式能源', **dg)

# ==================== 构建简单边连接 ====================
connections = [
    # 断路器相关连接
    ("CB1", "U1", False),
    ("CB3", "U23", False),
    ("CB2", "U43", False),
    
    # 分布式电源连接
    ("U22", "DG2", False), ("U16", "DG1", False),
    ("U39", "DG5", False), ("U32", "DG3", False),
    ("U35", "DG4", False), ("U52", "DG7", False),
    ("U55", "DG8", False), ("U48", "DG6", False)
]
connections += [
    # S13-1
    ("U13" , "S13-1" , False),
    ("S13-1", "LT_13_43", False),
    ("LT_13_43", "U43" , False),

    # S29-2
    ("U19" , "S29-2" , False),
    ("S29-2", "LT_19_29", False),
    ("LT_19_29", "U29" , False),

    # S62-3
    ("U23" , "S62-3" , False),
    ("S62-3", "LT_23_62", False),
    ("LT_23_62", "U62" , False),
]

# 添加预定义的连接关系
for conn in connections:
    u, v, is_tie = conn
    G.add_edge(u, v, 联络边=is_tie)

# ==================== 处理带分段开关的线路连接 ====================
for line_id, params in line_parameters.items():
    start_user = f"U{params['起点']}"
    end_user = f"U{params['终点']}"
    switch_code = params['开关']  
    line_node = f"L{line_id}"

    # 处理带分段开关的线路
    if switch_code != '0':  
        switch_node = switch_code
        # 构建连接路径：用户-开关-导线-用户
        G.add_edge(start_user, switch_node, 联络边=False)
        G.add_edge(switch_node, line_node, 联络边=False)
        G.add_edge(line_node, end_user, 联络边=False)
    
    # 无分段开关的常规连接
    else: 
        # 构建连接路径：用户-导线-用户
        G.add_edge(start_user, line_node, 联络边=False)
        G.add_edge(line_node, end_user, 联络边=False)

# ==================== 潮流计算 ====================
# 计算干路电流
def feeder_currents(I_amp, net):
    """
    计算三个CB的总电流
    CB1 = L1
    CB2 = L41 + L53 + (L12 if S13-1开启)
    CB3 = L22 + (L59 if S62-3开启)
    """
    # CB1的电流 = L1
    I1 = I_amp.get('L1', 0.0)
    
    # CB2的电流 = L41 + L53
    I2 = I_amp.get('L41', 0.0) + I_amp.get('L53', 0.0)
    # 如果S13-1开启，加上L12
    if net.nodes.get('S13-1', {}).get('开关状态') == '开':
        I2 += I_amp.get('L12', 0.0)
    
    # CB3的电流 = L22
    I3 = I_amp.get('L22', 0.0)
    # 如果S62-3开启，加上L59
    if net.nodes.get('S62-3', {}).get('开关状态') == '开':
        I3 += I_amp.get('L59', 0.0)
    
    return {'CL1': I1, 'CL2': I2, 'CL3': I3}

# 从无向图 G 生成以 CB 为根的径向 DiGraph D：
def build_radial_digraph(G):
    """构建径向有向图用于潮流计算"""
    H = G.copy()

    # 0. 删除故障节点
    fault_nodes = [n for n, d in H.nodes(data=True)
                   if d.get('故障状态', False)
                   and d.get('type') in ('导线', '分段开关', '联络开关', '断路器', '分布式能源')]
    H.remove_nodes_from(fault_nodes)
    
    # 1. 删除"关"的开关
    closed_sw = [n for n, d in H.nodes(data=True)
                 if d.get('type') in ('分段开关', '联络开关')
                 and d.get('开关状态') == '关']
    H.remove_nodes_from(closed_sw)

    # 2. 收缩"开"的开关
    for sw, d in list(H.nodes(data=True)):
        if d.get('type') in ('分段开关', '联络开关') and d.get('开关状态') == '开':
            nbrs = list(H.neighbors(sw))
            if len(nbrs) == 2:
                u, v = nbrs
                H.add_edge(u, v)
                H.add_edge(v, u)
            H.remove_node(sw)

    # 3. 找断路器作为根
    roots = [cb for cb, d in H.nodes(data=True) if d.get('type') == '断路器']
    if not roots:
        raise ValueError("网络中没有断路器节点")

    # 4. 多源BFS
    dist = {}
    queue = deque()
    for cb in roots:
        dist[cb] = 0
        queue.append(cb)

    while queue:
        u = queue.popleft()
        for nbr in H.neighbors(u):
            if nbr not in dist:
                dist[nbr] = dist[u] + 1
                queue.append(nbr)
                
    # 5. 构造有向图
    D = nx.DiGraph()
    for line_id, d in H.nodes(data=True):
        if d.get('type') != '导线':
            continue
        
        ends = [nbr for nbr in H.neighbors(line_id)
                if H.nodes[nbr]['type'] in ('用户', '分布式能源')]
        if len(ends) != 2:
            continue
        
        u, v = ends
        du, dv = dist.get(u, math.inf), dist.get(v, math.inf)
        
        if du < dv:
            head, tail = u, v
        elif dv < du:
            head, tail = v, u
        else:
            head, tail = (u, v) if str(u) < str(v) else (v, u)
        
        # 添加边，包含电阻和电抗
        D.add_edge(head, tail,
                   id=line_id,
                   R=d.get('总电阻(Ω)', 0.0),
                   X=d.get('总电抗(Ω)', 0.0))
    
    return D, roots

def make_nodal_power_dict(G):
    P = {}
    for n, d in G.nodes(data=True):
        if d.get('故障状态', False):
            continue                 # 故障节点不计功率
        if d.get('type') == '用户':
            P[n] = d.get('用电功率(kW)', 0.0)
        elif d.get('type') == '分布式能源' and d.get('开关状态') == '开':
            P[n] = -d.get('容量(kW)', 0.0)
    return P

SQRT3 = math.sqrt(3)
def distflow_forward_backward(D, roots, P_nodal, debug=False):
    """
    改进的配电网前推回代潮流计算
    添加收敛判据和迭代控制
    """
    # 获取拓扑排序
    try:
        order = list(nx.topological_sort(D))
    except:
        return {}, {}, {}
    
    order_reverse = list(reversed(order))
    
    # 迭代控制参数
    max_iterations = 50
    tolerance = 1e-6
    
    # 初始化
    V_mag = {n: 1.0 for n in D.nodes()}
    V_mag_prev = {n: 1.0 for n in D.nodes()}  # 保存上一次迭代的电压
    P_flow = {}  # 线路有功功率流(kW)
    Q_flow = {}  # 线路无功功率流(kvar)
    
    # 迭代求解
    for iteration in range(max_iterations):
        # 保存上一次迭代的电压值
        V_mag_prev = V_mag.copy()
        
        # 后推：计算功率流
        P_flow.clear()
        Q_flow.clear()
        
        for node in order_reverse:
            # 节点注入功率
            P_node = P_nodal.get(node, 0.0)  # kW
            Q_node = P_node * math.tan(math.acos(POWER_FACTOR)) if P_node != 0 else 0  # kvar
            
            # 累加流出功率
            P_out = P_node
            Q_out = Q_node
            
            # 加上所有下游线路功率
            for child in D.successors(node):
                edge_data = D.edges[node, child]
                line_id = edge_data['id']
                if line_id in P_flow:
                    P_out += P_flow[line_id]
                    Q_out += Q_flow[line_id]
            
            # 计算流向上游的功率
            for parent in D.predecessors(node):
                edge_data = D.edges[parent, node]
                line_id = edge_data['id']
                R = edge_data['R']
                X = edge_data['X']
                
                # 计算线路损耗
                V_node = V_mag[node] * V_BASE / 1000  # kV
                if V_node > 0:
                    I_approx = math.sqrt(P_out**2 + Q_out**2) / (SQRT3 * V_node)  # A
                    P_loss = 3 * I_approx**2 * R / 1000  # kW
                    Q_loss = 3 * I_approx**2 * X / 1000  # kvar
                else:
                    P_loss = Q_loss = 0
                
                # 送端功率 = 受端功率 + 损耗
                P_flow[line_id] = P_out + P_loss
                Q_flow[line_id] = Q_out + Q_loss
        
        # 前推：更新电压
        for node in order:
            if node in roots:
                V_mag[node] = 1.0
            else:
                for parent in D.predecessors(node):
                    edge_data = D.edges[parent, node]
                    line_id = edge_data['id']
                    R = edge_data['R']
                    X = edge_data['X']
                    
                    P = P_flow.get(line_id, 0)
                    Q = Q_flow.get(line_id, 0)
                    V_parent = V_mag[parent]
                    
                    # 电压降落
                    if V_BASE > 0:
                        delta_V_pu = (P * R + Q * X) / (V_BASE**2 / 1000)
                        V_mag[node] = max(0.8, V_parent - delta_V_pu)  # 限制最低电压
        
        # 检查收敛性
        max_voltage_change = 0.0
        for node in D.nodes():
            voltage_change = abs(V_mag[node] - V_mag_prev[node])
            max_voltage_change = max(max_voltage_change, voltage_change)
        
        if debug and iteration == 0:
            print(f"潮流计算迭代 {iteration + 1}: 最大电压变化 = {max_voltage_change:.6f}")
        
        # 判断是否收敛
        if max_voltage_change < tolerance:
            if debug:
                print(f"潮流计算在第 {iteration + 1} 次迭代后收敛")
            break
    else:
        if debug:
            print(f"潮流计算达到最大迭代次数 {max_iterations}，最大电压变化 = {max_voltage_change:.6f}")
    
    # 计算线路电流
    I_amp = {}
    P_send = {}
    
    for (u, v), data in D.edges.items():
        line_id = data['id']
        if line_id in P_flow:
            P = P_flow[line_id]  # kW
            Q = Q_flow.get(line_id, P * math.tan(math.acos(POWER_FACTOR)))  # kvar
            V_u = V_mag[u] * V_BASE  # V
            
            if V_u > 0:
                # 视在功率
                S = math.sqrt(P**2 + Q**2) * 1000  # VA
                # 三相电流
                I = S / (SQRT3 * V_u)  # A
                
                I_amp[line_id] = I
                P_send[line_id] = P
    
    return P_send, I_amp, V_mag

# 潮流计算主函数入口
POWER_FACTOR = 0.95  # 功率因数
def run_distflow(G):
    """运行潮流计算并检查过载"""
    global fault_counter
    fault_counter = getattr(run_distflow, 'counter', 0) + 1
    run_distflow.counter = fault_counter
    debug = (fault_counter % 100 == 1)  # 每100个故障打印一次调试信息
    
    try:
        D, roots = build_radial_digraph(G)
        if D.number_of_edges() == 0:
            if debug:
                print("无可用导线，无法计算潮流")
            return []
        
        # 构建节点功率字典
        P_nodal = make_nodal_power_dict(G)
        
        # 运行潮流计算
        P_send, I_amp, V_mag = distflow_forward_backward(D, roots, P_nodal)
        
        # 计算CB电流
        cb_currents = feeder_currents(I_amp, G)
        
        # 过载检查
        overload = []
        
        # 线路过载检查
        for line_id, I in I_amp.items():
            if I > 1.1 * RATED_CURRENT:
                overload.append((line_id, I))
        
        # CB过载检查
        for cb_id, I in cb_currents.items():
            I_amp[cb_id] = I  # 保存CB电流到结果中
            if I > 1.1 * RATED_CURRENT:
                overload.append((cb_id, I))
        
        return overload
        
    except Exception as e:
        if debug:
            print(f"潮流计算错误: {e}")
            import traceback
            traceback.print_exc()
        return []
# ---- 在 worker 进程内的潮流缓存（按状态键）----
_RUN_CACHE = {}
# ==== 统一状态键 & 潮流缓存 ====
EVAL_CACHE = {}     # 候选评估全局缓存（每个worker进程内共享）
_RUN_CACHE = {}     # 潮流结果缓存（按状态键）

def statekey(net: nx.Graph):
    """网络状态键：按(故障集合, 关键开关状态)唯一标识一个潮流状态"""
    faults = tuple(sorted(
        n for n, d in net.nodes(data=True)
        if d.get('故障状态', False)
    ))
    sws = tuple(sorted(
        (n, net.nodes[n].get('开关状态'))
        for n, d in net.nodes(data=True)
        if d.get('type') in ('分段开关', '联络开关', '断路器')
    ))
    return faults, sws

def _state_key(net: nx.Graph):   # 兼容老名字（有人在其他地方调用）
    return statekey(net)

def run_distflow_cached(net: nx.Graph):
    """带缓存的潮流计算"""
    k = statekey(net)
    v = _RUN_CACHE.get(k)
    if v is None:
        v = run_distflow(net)
        _RUN_CACHE[k] = v
    return v

def is_reachable_from_cb(G, cb, target_node):
    powered_areas = []
    if G.nodes[cb].get('故障状态'):
        return False
    visited = []
    stack = [cb]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.append(node)
        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue
        # 检查边经过的元件是否故障和开关状态
            if not G.nodes[neighbor].get('故障状态', False):
                if G.nodes[neighbor].get('type') in ['分段开关', '联络开关']:
                    if G.nodes[neighbor].get('开关状态') == '开':
                        stack.append(neighbor)
                else:
                    stack.append(neighbor)
    powered_areas = [n for n in visited if n.startswith('U')]
    if target_node in powered_areas:
        return True
    else:
        return False
    
# 创建NetworkOperator类封装故障相关函数
class NetworkOperator:
    def __init__(self, G):
        self.G = G.copy()
        self.original_G = G
        self._init_network_status()

    # 初始化网络状态（重置并断开联络开关）
    def _init_network_status(self):
        self.G = copy.deepcopy(self.original_G)  
        for n in self.G.nodes:
            if self.G.nodes[n]['type'] == '联络开关':
                self.G.nodes[n]['开关状态'] = '关'
        self._configure_initial_state()

    # 从每个DG出发，搜索到最近的CB并保存路径到visited列表
    def _configure_initial_state(self):
        for dg in dgs:
            visited = []  # 保存所有路径中的节点
            dg_name = dg['名称']
            critical_node=None
            load=0
            total_power=self.G.nodes[dg_name]['容量(kW)']
            if dg_name not in self.G.nodes:
                continue

            # BFS搜索路径
            queue = deque([(dg_name, [dg_name])])
            visited_nodes = set()
            found = False
            while queue and not found:
                current, path = queue.popleft()
                if current in visited_nodes:
                    continue
                visited_nodes.add(current)

                # 检查当前节点是否是CB
                if current.startswith('CB'):
                    visited.extend(path)  # 保存路径节点
                    found = True
                    critical_node = current
                    break

                # 如果当前节点是开关，且开关关闭则阻断路径
                node_data = self.G.nodes[current]
                if node_data['type'] in ['分段开关', '联络开关']:
                    if node_data.get('开关状态', '关') != '开':
                        continue  

                # 遍历邻居节点
                for neighbor in self.G.neighbors(current):
                    if neighbor not in visited_nodes:
                        queue.append((neighbor, path + [neighbor]))

            for node in visited:
                if node.startswith('U'):
                    load += self.G.nodes[node]['用电功率(kW)']
                if load >= total_power and critical_node.startswith('CB'):
                    critical_node = node
                    break
            # 找到关键节点，向 DG 搜索，配置开关和 DG 状态
            if critical_node:
                critical_index = visited.index(critical_node)
                for idx in range(critical_index - 1, -1, -1):
                    node_name = visited[idx]
                    if node_name.startswith('S') and node_name not in main_switches:
                        self.G.nodes[node_name]['开关状态'] = '关'
                        self.G.nodes[dg_name]['开关状态'] = '开'
                        break
    
    # 计算失负荷
    def _risk(self, net, fault_loc=None):

        # ---------- 构造真实连通图 ----------
        H = net.copy()
        remove = []
        for n, d in H.nodes(data=True):
            if d.get('故障状态', False) and d['type'] in {'导线','分段开关','联络开关','断路器'}:
                remove.append(n)
            elif d['type'] in {'分段开关','联络开关'} and d.get('开关状态') == '关':
                remove.append(n)
        H.remove_nodes_from(remove)

        # ---------- 对每个连通分区判断是否"全供" ----------
        lost = 0.0
        for comp in nx.connected_components(H):
            load_sum = 0.0
            gen_cap  = 0.0
            has_cb   = False

            # 统计本分区负荷与发电
            for n in comp:
                nd = H.nodes[n]
                if nd['type'] == '用户':
                    load_sum += nd['用电功率(kW)']
                elif nd['type'] == '断路器' and not nd.get('故障状态', False):
                    has_cb = True           # 视为容量无限
                elif nd['type'] == '分布式能源' and \
                    nd.get('开关状态') == '开' and \
                    not nd.get('故障状态', False):
                    gen_cap += nd['容量(kW)']

            # 是否失电
            if not has_cb:                       
                if gen_cap < load_sum:           
                    # 按每个用户的权重乘以对应的用电功率累加
                    for u in comp:
                        if u.startswith('U'):
                            user_power = H.nodes[u]['用电功率(kW)']
                            user_weight = TYPE_W[H.nodes[u]['类型']]
                            lost += user_power * user_weight

        # ---------- 短路用户单独算 ----------
        if fault_loc and fault_loc.startswith('U'):
            nd = net.nodes[fault_loc]
            lost += nd['用电功率(kW)'] * TYPE_W[nd['类型']]
        return lost

    # 核心算法：根据故障位置 loc 判断故障类型 (1-5)
    def _determine_fault_type(self, loc):
        # 步骤1: 判断是否在干路（直接通过线路或所在线路判断）
        if self._is_main_line_fault(loc, main_lines):
            if loc.startswith('U'):
                return 0
            else:
                return 5  # 类型5，干路故障
        
        # 步骤2: 如果不在干路上，向两边搜索直到碰到干路或 DG
        is_found_dg = False
        found_dg = None
        for direction in ['upstream', 'downstream']:
            path = self._search_until_main_line_or_dg(loc, direction)
            
            # 如果路径中找到 DG，则标记并结束搜索
            for node in path:
                if self.G.nodes[node]['type'] == '分布式能源':
                    is_found_dg = True
                    found_dg = node
                    break
            
            if is_found_dg:
                break
        
        # 如果没找到 DG，则按情况2处理
        if not is_found_dg:
            return 2  # 类型2，未找到DG，视为支路故障

        # 步骤3: DG 是打开的（DG 自己的开关状态 == '开'）
        if self.G.nodes[found_dg]['开关状态'] == '开':
            if self._has_switch_in_path(loc, found_dg, '关'):
                return 2      # 类型 2：同一方向上有断开的开关
            else:
                return 1      # 类型 1：这条方向上一路畅通

        # 步骤4: DG 是关闭的（DG 自己的开关状态 == '关'）
        if self.G.nodes[found_dg]['开关状态'] == '关':
            if self._has_switch_in_path(loc, found_dg, '开'):
                return 4      # 类型 4：这条方向上有开启的开关
            else:
                return 3      # 类型 3：这条方向上完全没有开关
            
        return 5  # 其余默认归为干路故障

    def _has_switch_in_path(self, start, end, target_state):
        try:
            path = nx.shortest_path(self.G, start, end)
        except nx.NetworkXNoPath:
            return False

        for node in path:
            if self.G.nodes[node]['type'] in ('分段开关', '联络开关'):
                if self.G.nodes[node]['开关状态'] == target_state:
                    return True
        return False
        
    def _is_main_line_fault(self, loc, main_lines):
        # 情况1: 故障位置是干路线路
        if loc.startswith('L'):
            return loc in main_lines

        # 情况2: 故障位置是用户，检查其连接线路是否为干路
        if loc.startswith('U'):
            connected_lines = []
            for nbr in self.G.neighbors(loc):
                if self.G.nodes[nbr]['type'] == '导线':
                    connected_lines.append(nbr)

            # 检查是否所有连接的线路中是否有干路
            if len(connected_lines) > 0:
                return any(line in main_lines for line in connected_lines)
            return False
        if loc.startswith('CB'):
            return True
        if loc in ['S13-1', 'S29-2', 'S62-3']:
            return True
        else:
            return loc in main_switches
    
    def _search_until_main_line_or_dg(self, loc, direction):
        path = []
        current_node = loc
        visited = set()  # 记录已访问节点
        
        while True:
            if current_node in visited:
                break
            visited.add(current_node)
            
            neighbors = list(self.G.neighbors(current_node))
            if direction == 'upstream':
                neighbors = list(reversed(neighbors))
                
            found_target = False
            for nbr in neighbors:
                if nbr in visited:
                    continue
                    
                if nbr.startswith('L') and nbr in main_lines:
                    path.append(nbr)
                    found_target = True
                    break
                if self.G.nodes[nbr].get('type') == '分布式能源':
                    path.append(nbr)
                    found_target = True
                    break
            
            if found_target:
                break
            else:
                valid_neighbors = [n for n in neighbors if n not in visited]
                if not valid_neighbors:
                    break
                    
                path.append(current_node)
                current_node = valid_neighbors[0]  # 取第一个有效邻居
        
        return path

    def _has_open_switch_in_path(self, start, end):
        try:
            path = nx.shortest_path(self.G, start, end)
        except nx.NetworkXNoPath:
            return False

        for node in path:
            if self.G.nodes[node]['type'] in ['分段开关', '联络开关'] and self.G.nodes[node]['开关状态'] == '关':
                return True
        return False

    def _judge_main_fault_type(self, loc):
        tie_switch_constraints = {}
        if loc in ['CB1', 'L1', 'L2', 'L3', 'S1']:
            tie_switch_constraints = {
                "S62-3": ["S29", "S27"]                       # 打开S62-3时，S29/S27至少关一个
            }
        elif loc in ['L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12',
                        'S2', 'S3', 'S4', 'S5']:
            tie_switch_constraints = {
                "S29-2": ["S1", "S8", "S11", "S12"],          # 打开S29-2时，S1/S8/S11/S12至少关一个
                "S62-3": ["S29", "S27"]                       # 打开S62-3时，S29/S27至少关一个
            }
        elif loc in ['CB2']:
            tie_switch_constraints = {
                "S29-2": ["S1", "S8", "S11", "S12"],          # 打开S29-2时，S1/S8/S11/S12至少关一个
            }
        elif loc in ['L53', 'L54', 'L57', 'L58', 'L59', 'S27', 'S29']:
            tie_switch_constraints = {
                "S13-1": ["S1", "S2", "S3", "S4", "S5"],      # 打开S13-1时，S1/S2/S3/S4/S5至少关一个
                "S29-2": ["S1", "S8", "S11", "S12"],          # 打开S29-2时，S1/S8/S11/S12至少关一个
            }
        elif loc in ['CB3']:
            tie_switch_constraints = {
                "S13-1": ["S1", "S2", "S3", "S4", "S5"],      # 打开S13-1时，S1/S2/S3/S4/S5至少关一个
            }
        elif loc in ['L22', 'L23', 'L24', 'L25', 'L26', 'L27', 'S11', 'S12',
                        'L16', 'L17', 'L18', 'S8']:
            tie_switch_constraints = {
                "S13-1": ["S1", "S2", "S3", "S4", "S5"],      # 打开S13-1时，S1/S2/S3/S4/S5至少关一个
                "S62-3": ["S29", "S27"]                       # 打开S62-3时，S29/S27至少关一个
            }
        return tie_switch_constraints

    # 单故障核心处理函数，返回损失、策略等
    def simulate_fault(self, ftype, loc, reset=True):
        if reset:
            self._init_network_status()

        # 1) 获取网络状态与分支 DG
        branch_dg_list = self._branch_dgs(loc, self.G)

        # 2) 计算“短路用户”的额外损失（_risk 里已处理，这里保留以备扩展）
        node_data = self.G.nodes[loc]
        if node_data.get('type') == '用户':
            extra_lost = node_data['用电功率(kW)'] * TYPE_W[node_data['类型']]
        else:
            extra_lost = 0.0

        # 3) 构建基础故障网络
        net0 = copy.deepcopy(self.G)
        net0.nodes[loc]['故障状态'] = True
        if loc.startswith('L'):
            base_label = f"断路线路 {loc}"
        elif loc.startswith('S') or loc.startswith('CB'):
            net0.nodes[loc]['开关状态'] = '关'
            base_label = f"跳闸 {loc}"
        elif loc.startswith('DG'):
            net0.nodes[loc]['开关状态'] = '关'
            base_label = f"{loc}故障"
        else:
            # 用户短路：把两侧导线直接并上，等价删除该用户节点
            lines = [n for n in net0.neighbors(loc)]
            if len(lines) >= 2:
                u, v = lines[0], lines[1]
                net0.add_edge(u, v, 联络边=False)
            base_label = f"短路用户 {loc}"

        # ========== 候选生成：用 statekey 去重 ==========
        candidates: list[tuple[nx.Graph, str]] = []
        seen: set = set()

        def add_candidate(net: nx.Graph, label: str):
            """按 statekey 去重后加入候选"""
            try:
                k = statekey(net)  # 你已实现
            except Exception:
                # 兜底：即使 statekey 异常，也至少放进去一次
                k = None
            if (k is None) or (k not in seen):
                candidates.append((net, label))
                if k is not None:
                    seen.add(k)

        # 基础策略
        add_candidate(net0, base_label)

        # ---------- 故障类型1：打开上游最近的“关”开关 ----------
        if ftype == 1:
            # 找最近的分段/联络开关（且当前为“关”）
            visited, queue = set(), deque([loc])
            found_switch = None
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                if net0.nodes[curr]['type'] in ('分段开关', '联络开关') and net0.nodes[curr]['开关状态'] == '关':
                    found_switch = curr
                    break
                queue.extend([nbr for nbr in net0.neighbors(curr) if nbr not in visited])

            if found_switch is not None:
                # 基础：仅打开该开关
                base1 = copy.deepcopy(net0)
                base1.nodes[found_switch]['开关状态'] = '开'
                add_candidate(base1, f"打开开关 {found_switch}")

                # 若出现过载，再尝试联络组合
                overload = run_distflow_cached(base1)
                if overload:
                    switches = ['S8', 'S11', 'S12', 'S1', 'S2', 'S3', 'S4', 'S5', 'S29', 'S27', 'S62-3', 'S13-1', 'S29-2']
                    max_switch_num = 3
                    key_ties = {'S13-1', 'S29-2', 'S62-3'}
                    tie_switch_constraints = self._judge_main_fault_type(loc)

                    for i in range(1, min(max_switch_num, len(switches)) + 1):
                        for combo in combinations(switches, i):
                            # 至少包含一个关键联络
                            if not key_ties.intersection(combo):
                                continue
                            new_net = copy.deepcopy(base1)   # 注意：从 base1 出发，确保 found_switch 已打开
                            label_parts = [f"开{found_switch}"]

                            for sw in combo:
                                if sw in new_net.nodes:
                                    if new_net.nodes[sw]['开关状态'] == '关':
                                        new_net.nodes[sw]['开关状态'] = '开'
                                        label_parts.append(f"开{sw}")
                                    else:
                                        new_net.nodes[sw]['开关状态'] = '关'
                                        label_parts.append(f"关{sw}")

                            # 约束：若联络开，则约束集中至少一把为关
                            valid = True
                            for tie_sw, must_cut in tie_switch_constraints.items():
                                if tie_sw in new_net.nodes and new_net.nodes[tie_sw]['开关状态'] == '开':
                                    if not any(new_net.nodes[n]['开关状态'] == '关' for n in must_cut if n in new_net.nodes):
                                        valid = False
                                        break
                            if valid:
                                add_candidate(new_net, "操作：" + "+".join(label_parts))

        # ---------- 故障类型5：组合开关操作 ----------
        elif ftype == 5:
            switches = ['S8', 'S11', 'S12', 'S1', 'S2', 'S3', 'S4', 'S5', 'S29', 'S27', 'S62-3', 'S13-1', 'S29-2']
            max_switch_num = 3
            tie_switch_constraints = self._judge_main_fault_type(loc)

            for i in range(1, min(max_switch_num, len(switches)) + 1):
                for combo in combinations(switches, i):
                    new_net = net0.copy()

                    label_parts = []
                    for sw in combo:
                        if sw in new_net.nodes:
                            if new_net.nodes[sw]['开关状态'] == '关':
                                new_net.nodes[sw]['开关状态'] = '开'
                                label_parts.append(f"开{sw}")
                            else:
                                new_net.nodes[sw]['开关状态'] = '关'
                                label_parts.append(f"关{sw}")

                    valid = True
                    for tie_sw, must_cut in tie_switch_constraints.items():
                        if tie_sw in new_net.nodes and new_net.nodes[tie_sw]['开关状态'] == '开':
                            if not any(new_net.nodes[n]['开关状态'] == '关' for n in must_cut if n in new_net.nodes):
                                valid = False
                                break
                    if valid:
                        add_candidate(new_net, "操作：" + "+".join(label_parts))

        # ---------- 故障类型3：断开分支 DG ----------
        elif ftype == 3:
            for dg in branch_dg_list:
                if dg not in net0.nodes:
                    continue
                # 找 DG 上游最近的分段/联络开关（不限开关当前状态）
                visited, queue = set(), deque([dg])
                found_switch = None
                while queue:
                    curr = queue.popleft()
                    if curr in visited:
                        continue
                    visited.add(curr)
                    if net0.nodes[curr]['type'] in ('分段开关', '联络开关'):
                        found_switch = curr
                        break
                    queue.extend([nbr for nbr in net0.neighbors(curr) if nbr not in visited])

                if not found_switch:
                    continue

                tmp_net = copy.deepcopy(net0)
                tmp_net.nodes[dg]['开关状态'] = '开'
                if not loc.startswith('L'):
                    tmp_net.nodes[found_switch]['开关状态'] = '关'

                # 删除该开关节点，计算 DG 可到达区域
                H = tmp_net.copy()
                if found_switch in H:
                    H.remove_node(found_switch)
                reachable = nx.node_connected_component(H, dg)

                # 区域负荷（排除故障点）
                load_sum = 0.0
                for n in reachable:
                    n_data = tmp_net.nodes[n]
                    if n_data['type'] == '用户' and n != loc:
                        load_sum += n_data['用电功率(kW)']

                if load_sum <= tmp_net.nodes[dg]['容量(kW)']:
                    add_candidate(tmp_net, f"开{dg} + 断{found_switch}")

        # 其它类型：仅保留基础策略
        # ==============================================

        # 5) 评估候选：用 eval_cache（key=statekey）避免重复潮流计算
        #eval_cache: dict = {}
                # 5) 评估候选：用全局 EVAL_CACHE（key=statekey）避免重复潮流与重复打分
        best_risk = (float('inf'), float('inf'), float('inf'))  # (lost, over_penalty, total)
        best_strategy = base_label
        best_overload = []

        for net, label in candidates:
            k = statekey(net)

            if k in EVAL_CACHE:
                # 命中缓存：直接取
                lost, over_penalty, total, overload = EVAL_CACHE[k]
            else:
                # 潮流 + 过载罚分
                overload = run_distflow_cached(net)

                # === 过载罚分 ===
                w1, w2, threshold = 10.0, 10.0, 1.3
                over_penalty = 0.0
                for L, I in overload:
                    ratio = I / I_RATE

                    # 计算 w0（过载元件涉及的“权重×功率”）
                    w_0 = 0.0
                    if L in line_dict:
                        U1, U2 = line_dict[L]
                        U1_power = net.nodes[U1]['用电功率(kW)']
                        U2_power = net.nodes[U2]['用电功率(kW)']
                        U1_weight = TYPE_W[net.nodes[U1]['类型']]
                        U2_weight = TYPE_W[net.nodes[U2]['类型']]
                        w_0 = U1_weight * U1_power + U2_weight * U2_power
                    elif L.startswith('CL'):
                        if L == 'CL1':
                            U = 'U1'
                        elif L == 'CL2':
                            U = 'U43'
                        elif L == 'CL3':
                            U = 'U23'
                        else:
                            U = None
                        if U:
                            U_power = net.nodes[U]['用电功率(kW)']
                            U_weight = TYPE_W[net.nodes[U]['类型']]
                            w_0 = U_weight * U_power
                    elif L.startswith('LT'):
                        if L == 'LT_13_43':
                            U1, U2 = 'U13', 'U43'
                        elif L == 'LT_19_29':
                            U1, U2 = 'U19', 'U29'
                        elif L == 'LT_23_62':
                            U1, U2 = 'U23', 'U62'
                        else:
                            U1 = U2 = None
                        if U1 and U2:
                            U1_power = net.nodes[U1]['用电功率(kW)']
                            U2_power = net.nodes[U2]['用电功率(kW)']
                            U1_weight = TYPE_W[net.nodes[U1]['类型']]
                            U2_weight = TYPE_W[net.nodes[U2]['类型']]
                            w_0 = U1_weight * U1_power + U2_weight * U2_power

                    if ratio <= 1.1:
                        continue  # 未超 1.1，不罚
                    if ratio <= threshold:
                        over_penalty += w_0 * w1 * (ratio - 1.1)
                    else:
                        over_penalty += w_0 * w1 * (ratio - 1.1)
                        over_penalty += w_0 * w2 * (1 + (ratio - threshold)) ** 2

                # === 失负荷（_risk 内部已对用户短路做额外损失处理）===
                lost = self._risk(net, loc)

                total = (lost + over_penalty)

                # 与原先一致：乘 1e4
                total *= 1e4
                lost *= 1e4
                over_penalty *= 1e4

                # 写入全局缓存
                EVAL_CACHE[k] = (lost, over_penalty, total, overload)

            # 更新最优
            if total < best_risk[2] or (total == best_risk[2] and over_penalty < best_risk[1]):
                best_risk = (lost, over_penalty, total)
                best_strategy = label
                best_overload = overload

        return (*best_risk, best_strategy, best_overload)


    def _branch_dgs(self,node,net):
        comp = nx.node_connected_component(net,node)
        return [u for u in comp if net.nodes[u]['type']=='分布式能源']

    # 用于在 (1,1) 场景下判断哪一个故障更靠近主干
    def _distance_to_main(self, loc, net):
        
        cb = self._find_feeder_root(loc, net)
        if cb is None:
            return float('inf')
        try:
            return nx.shortest_path_length(net, source=loc, target=cb)
        except nx.NetworkXNoPath:
            return float('inf')

    # 单点故障 loc 的候选操作生成器
    def _generate_candidates_for_fault(self, loc):
        
        # 先打故障标记在一个临时网络上（但不改开关状态）
        net0 = copy.deepcopy(self.original_G)
        net0.nodes[loc]['故障状态'] = True

        ftype = self._determine_fault_type(loc)
        cands = []

        # 基础“不操作”策略
        cands.append(([], f"不操作（{loc} 基本策略）"))

        # 类型1：打开上游的断开开关
        if ftype == 1:
            # 向上游 BFS 找最近的分段/联络开关
            visited, queue = set(), deque([loc])
            found_switch = None
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)

                if net0.nodes[curr]['type'] in ('分段开关', '联络开关') and net0.nodes[curr]['开关状态'] == '关':
                    found_switch = curr
                    break
                queue.extend([nbr for nbr in net0.neighbors(curr) if nbr not in visited])
            new_net = net0.copy()

            new_net.nodes[found_switch]['开关状态'] = '开'

            # 考虑接入原本由DG供应的用户后可能会出现过负荷的情形
            overload = run_distflow_cached(new_net)
            if not overload:
                cands.append(( [("open", found_switch)], f"打开开关 {found_switch}" ))
            else:
                # 添加原始策略
                cands.append(( [("open", found_switch)], f"打开开关 {found_switch}" ))
                # 过负荷的情况，添加联络线策略
                switches = ['S8', 'S11', 'S12', 'S1', 'S2', 'S3', 'S4', 'S5', 'S29', 'S27', 'S62-3', 'S13-1', 'S29-2']
                max_switch_num = 3  # 最大同时操作开关数量
                key_ties = {'S13-1', 'S29-2', 'S62-3'}
                # 定义联络开关及其约束关系
                tie_switch_constraints = {
                }
                for i in range(1, min(max_switch_num, len(switches)) + 1):
                    for combo in combinations(switches, i):
                        # 判断是否包含至少一个关键联络开关
                        if not key_ties.intersection(combo):
                            continue
                        new_net = net0.copy()

                        label_parts = [f"打开开关{found_switch}"]
                        ops = [("open", found_switch)]
                        for sw in combo:
                            if sw in new_net.nodes:  
                                if new_net.nodes[sw]['开关状态'] == '关':
                                    new_net.nodes[sw]['开关状态'] = '开'
                                    label_parts.append(f"开{sw}")
                                    ops.append(("open", sw))
                                else:
                                    new_net.nodes[sw]['开关状态'] = '关'
                                    label_parts.append(f"关{sw}")
                                    ops.append(("close", sw))
                        # 检查是否符合约束关系
                        valid = True
                        for sw in combo:
                            if sw in tie_switch_constraints and new_net.nodes[sw]['开关状态'] == '开':
                                # 检查约束开关是否至少有一个关
                                if not any(new_net.nodes[const_sw]['开关状态'] == '关' for const_sw in tie_switch_constraints[sw]):
                                    valid = False
                                    break
                        if valid:
                            cands.append((ops, "操作：" + "+".join(label_parts)))

        # 类型5：组合开关操作
        elif ftype == 5:
            switches = ['S8', 'S11', 'S12', 'S1', 'S2', 'S3', 'S4', 'S5', 'S29', 'S27', 'S62-3', 'S13-1', 'S29-2']
            max_switch_num = 3  # 最大同时操作开关数量
            # 定义联络开关及其约束关系
            tie_switch_constraints = {
            }
            for i in range(1, min(max_switch_num, len(switches)) + 1):
                for combo in combinations(switches, i):
                    new_net = net0.copy()

                    label_parts = []
                    ops = []
                    for sw in combo:
                        if sw in new_net.nodes:  
                            if new_net.nodes[sw]['开关状态'] == '关':
                                new_net.nodes[sw]['开关状态'] = '开'
                                label_parts.append(f"开{sw}")
                                ops.append(("open", sw))
                            else:
                                new_net.nodes[sw]['开关状态'] = '关'
                                label_parts.append(f"关{sw}")
                                ops.append(("close", sw))
                    # 检查是否符合约束关系
                    valid = True
                    for sw in combo:
                        if sw in tie_switch_constraints and new_net.nodes[sw]['开关状态'] == '开':
                            # 检查约束开关是否至少有一个关
                            if not any(new_net.nodes[const_sw]['开关状态'] == '关' for const_sw in tie_switch_constraints[sw]):
                                valid = False
                                break
                    if valid:
                        cands.append((ops, "操作：" + "+".join(label_parts)))

        # 类型3：断开分支 DG
        elif ftype == 3:
            branch_dg_list = self._branch_dgs(loc, self.original_G)
            for dg in branch_dg_list:
                # 找到距离 dg 最近的分段/联络开关
                visited, queue = set(), deque([dg])
                found_sw = None
                while queue and found_sw is None:
                    u = queue.popleft()
                    visited.add(u)
                    for nbr in self.original_G.neighbors(u):
                        if nbr in visited: continue
                        if self.original_G.nodes[nbr]['type'] in ('分段开关','联络开关'):
                            found_sw = nbr
                            break
                        queue.append(nbr)
                if not found_sw: 
                    continue
                # 把 DG 打开、该开关关闭
                ops = [("open", dg), ("close", found_sw)]
                label = f"开{dg} + 断{found_sw}"
                cands.append((ops, label))

        # 其余类型仅保留“不操作”
        return cands
    
    # 寻找 loc 由哪个 CB 供电
    def _find_feeder_root(self, loc, net):
        for cb in ['CB1','CB2','CB3']:
            if not net.nodes[cb].get('故障状态', False) \
               and is_reachable_from_cb(net, cb, loc):
                return cb
        return None

    # 判断 loc1, loc2 是否在同一支路上
    def _is_same_branch(self, loc1, loc2, net):
        mains = main_lines
        def attached_main(loc):
            for nbr in net.neighbors(loc):
                if nbr in mains:
                    return nbr
            return None

        m1 = attached_main(loc1)
        m2 = attached_main(loc2)
        return (m1 is not None) and (m1 == m2)

    # 双故障处理核心函数，同样返回损失、策略等
    def simulate_two_faults(self, loc1, loc2):

        # ---------- 构造“基础双故障网络” -----------------
        net0 = copy.deepcopy(self.original_G)
        def _apply_single_fault(g, loc):
            if loc not in g.nodes:
                return
            g.nodes[loc]['故障状态'] = True
            if loc.startswith('L'):                     
                g.remove_node(loc)
            elif loc.startswith(('S', 'CB', 'DG')):           
                g.nodes[loc]['开关状态'] = '关'

        _apply_single_fault(net0, loc1)
        _apply_single_fault(net0, loc2)

        # ---------- 生成候选操作组合 -----------------------
        f1 = self._determine_fault_type(loc1)
        f2 = self._determine_fault_type(loc2)
        combined = []          

        # 1 判断馈线 / 支路关系
        feeder1 = self._find_feeder_root(loc1, self.original_G)
        feeder2 = self._find_feeder_root(loc2, self.original_G)
        same_feeder  = (feeder1 == feeder2)
        same_branch  = same_feeder and self._is_same_branch(loc1, loc2,
                                                            self.original_G)
        # 2 获取各自的单故障候选 
        c1 = self._generate_candidates_for_fault(loc1)
        c2 = self._generate_candidates_for_fault(loc2)

        if f1 == f2 == 5:   # 两个都是干路故障，只针对更靠近主干的做控制
            nearer = min((loc1, loc2),
                         key=lambda x: self._distance_to_main(x,
                                                              self.original_G))
            cs = self._generate_candidates_for_fault(nearer)
            for ops, lbl in cs:
                combined.append((ops, [], lbl+f"(源:{nearer})"))

        elif not same_feeder or (same_feeder and not same_branch):
            # 不同馈线或同馈线不同支路 → 做笛卡儿积
            for ops1, lbl1 in c1:
                for ops2, lbl2 in c2:
                    combined.append((ops1, ops2, lbl1 + " + " + lbl2))
        else:
            # 同支路 → 四种特殊规则
            if f1 == f2 == 1:
                nearer = min((loc1, loc2),
                             key=lambda x: self._distance_to_main(x,
                                                                  self.original_G))
                for ops, lbl in self._generate_candidates_for_fault(nearer):
                    combined.append((ops, [], lbl))
            elif {f1, f2} == {1, 2}:
                combined.append(([], [], "不做操作"))
            elif f1 == f2 == 3:
                for loc in (loc1, loc2):
                    for ops, lbl in self._generate_candidates_for_fault(loc):
                        combined.append((ops, [], lbl))
            elif (f1, f2).count(4) and (f1, f2).count(3):
                tgt = loc1 if f1 == 4 else loc2
                for ops, lbl in self._generate_candidates_for_fault(tgt):
                    combined.append((ops, [], lbl))
            else:
                combined.append(([], [], "不做操作"))
        
        tie_switch_constraints = {}
        tie_switch_constraints_1  = {
            "S13-1": ["S1", "S2", "S3", "S4", "S5"],      # 打开S13-1时，S1/S2/S3/S4/S5至少关一个
            "S29-2": ["S1", "S8", "S11", "S12"],          # 打开S29-2时，S1/S8/S11/S12至少关一个
            "S62-3": ["S29", "S27"]                       # 打开S62-3时，S29/S27至少关一个
        }
        tie_switch_constraints_2 = {
            "S13-1": ["S1", "S2", "S3", "S4", "S5"],      # 打开S13-1时，S1/S2/S3/S4/S5至少关一个
            "S29-2": ["S1", "S8", "S11", "S12"],          # 打开S29-2时，S1/S8/S11/S12至少关一个
            "S62-3": ["S29", "S27"]                       # 打开S62-3时，S29/S27至少关一个
        }
        # 故障类型为 1 或 5 时需要考虑断路
        if f1 == 1 or f1 == 5:
            tie_switch_constraints_1 = self._judge_main_fault_type(loc1)
        if f2 == 1 or f2 == 5:
            tie_switch_constraints_2 = self._judge_main_fault_type(loc2)

        # 求交集
        tie_switch_constraints = {k: set(tie_switch_constraints_1[k]) & set(tie_switch_constraints_2[k]) 
                                    for k in tie_switch_constraints_1.keys() & tie_switch_constraints_2.keys() 
                                    if set(tie_switch_constraints_1[k]) & set(tie_switch_constraints_2[k])}
        
        # 检查是否符合约束关系
        # 替换：按最终状态过滤候选（ops1/ops2 是 [("open"/"close", sw), ...]）
        def _final_switch_states(base_net, ops1, ops2):
            st = {n: base_net.nodes[n].get('开关状态')
                for n, d in base_net.nodes(data=True)
                if d.get('type') in ('分段开关', '联络开关', '断路器')}
            for act, sw in (ops1 + ops2):
                if sw in st:
                    st[sw] = '开' if act == 'open' else '关'
            return st

        filtered_combined = []
        for ops1, ops2, label in combined:
            st = _final_switch_states(net0, ops1, ops2)
            ok = True
            for tie, must_cut in tie_switch_constraints.items():
                if st.get(tie) == '开':  # 打开联络
                    if not any(st.get(c) == '关' for c in must_cut):  # 至少有一把约束开关为关
                        ok = False
                        break
            if ok:
                filtered_combined.append((ops1, ops2, label))

        combined = filtered_combined


        # ---------- 评估所有候选 ----------
        def _merge_ops(ops1, ops2):
            d = {}
            for act, sw in ops1: d[sw] = act
            for act, sw in ops2: d[sw] = act
            return list(d.items())

        best = (float('inf'), float('inf'), float('inf'), "无策略")
        for ops1, ops2, label in combined:
            net = copy.deepcopy(net0)           
            for sw, act in _merge_ops(ops1, ops2):
                if sw in net.nodes:
                    net.nodes[sw]['开关状态'] = '开' if act == 'open' else '关'

            # 计算潮流
            overload = run_distflow_cached(net)
            w1 = 10      # 1.1~1.3 倍区间线性权重
            w2 = 10     # 超过 1.3 倍后二次权重
            threshold = 1.3
            over_penalty = 0.0

            for L, I in overload:
                ratio = I / I_RATE

                # 计算不同的过负荷线路影响的用户权重
                w_0 = 0
                if L in line_dict:
                    U1, U2 = line_dict[L]
                    U1_power = net.nodes[U1]['用电功率(kW)']
                    U2_power = net.nodes[U2]['用电功率(kW)']
                    U1_weight = TYPE_W[net.nodes[U1]['类型']]
                    U2_weight = TYPE_W[net.nodes[U2]['类型']]
                    w_0 = U1_weight * U1_power + U2_weight * U2_power
                elif L.startswith('CL'):
                    if L == 'CL1':
                        U = 'U1'
                    elif L == 'CL2':
                        U = 'U43'
                    elif L == 'CL3':
                        U = 'U23'
                    U_power = net.nodes[U]['用电功率(kW)']
                    U_weight = TYPE_W[net.nodes[U]['类型']]
                    w_0 = U_weight * U_power
                elif L.startswith('LT'):
                    if L == 'LT_13_43':
                        U1 = 'U13'
                        U2 = 'U43'
                    elif L == 'LT_19_29':
                        U1 = 'U19'
                        U2 = 'U29'
                    elif L == 'LT_23_62':
                        U1 = 'U23'
                        U2 = 'U62'
                    U1_power = net.nodes[U1]['用电功率(kW)']
                    U2_power = net.nodes[U2]['用电功率(kW)']
                    U1_weight = TYPE_W[net.nodes[U1]['类型']]
                    U2_weight = TYPE_W[net.nodes[U2]['类型']]
                    w_0 = U1_weight * U1_power + U2_weight * U2_power
                
                if ratio <= threshold:
                    # 1.1→1.3 线性累加
                    over_penalty += w_0 * w1 * (ratio - 1.1) 
                else:
                    # 先把 >1.1 的线性罚分
                    over_penalty += w_0 * w1 * (ratio - 1.1) 
                    # 再把 >1.3 的部分二次放大
                    over_penalty += w_0 * w2 * (1 +(ratio - threshold) ) ** 2

            # 失负荷  
            lost = self._risk(net) + self._extra_user_loss((loc1, loc2))
            total = (lost + over_penalty)

            lost = lost * 1e4
            over_penalty = over_penalty * 1e4
            total = total * 1e4

            if total < best[2]:
                best = (lost, over_penalty, total, label)

        return best

    # 用户短路附加损失
    def _extra_user_loss(self, loc_pair):
        extra = 0.0
        for loc in loc_pair:
            if loc.startswith('U') and loc in self.original_G.nodes:
                nd = self.original_G.nodes[loc]
                extra += nd['用电功率(kW)'] * TYPE_W[nd['类型']]
        return extra

# 对故障节点分类
def device_category(loc):
    if loc.startswith('CB') or loc.startswith('S'):
        return 'switch'
    elif loc.startswith('DG'):
        return 'dg'
    elif loc.startswith('L'):
        return 'line'
    else:
        return 'other'

# ---- 种子工具（32位安全）----
U32_MASK = 0xFFFFFFFF
def _u32(x: int) -> int:
    return int(x) & U32_MASK

def _mix_seed(base: int, i: int, t: int) -> int:
    # 轻量 avalanche，全部落到 32bit
    s = (base ^ 0x9E3779B1) * 0x85EBCA77
    s ^= (i + 1) * 0xC2B2AE3D
    s ^= (t + 1) * 0x27D4EB2F
    return _u32(s)

# ============ 优化部分：全局进程池管理 ============
GLOBAL_POOL = None
GLOBAL_POOL_INITIALIZED = False

def cleanup_global_pool():
    """清理全局进程池（避免在 atexit/KeyboardInterrupt 阶段阻塞）"""
    global GLOBAL_POOL, GLOBAL_POOL_INITIALIZED
    try:
        if GLOBAL_POOL is not None:
            # 尽量优雅关闭；任何一步出错都不阻塞退出
            try:
                GLOBAL_POOL.close()
            except Exception:
                pass
            try:
                GLOBAL_POOL.terminate()
            except Exception:
                pass
            # 不强制 join，防止在解释器退出晚期卡住
            try:
                GLOBAL_POOL.join()
            except Exception:
                pass
    finally:
        GLOBAL_POOL = None
        GLOBAL_POOL_INITIALIZED = False

# ========= 进程池与抽样表的全局状态 =========
# 主进程缓存的抽样表
GLOBAL_EVENTS = None
GLOBAL_CDF = None
GLOBAL_TOTAL_PROB = None
GLOBAL_P_COND = None
GLOBAL_PROD_SAFE = None
# ========= 抽样表计算（主进程用） =========
def _compute_sampling_tables(base_graph):
    """给定基础图，计算抽样所需的 P_COND/PROD_SAFE/EVENTS/CDF/TOTAL_PROB"""
    # 元件失效概率（与你原来一致）
    p_fail = {}
    for n, d in base_graph.nodes(data=True):
        t = d['type']
        if t == '导线':
            p_fail[n] = d['长度(km)'] * 0.002
        elif t in ('分段开关', '联络开关', '断路器'):
            p_fail[n] = 0.002
        elif t in ('用户', '分布式能源'):
            p_fail[n] = 0.005
        else:
            p_fail[n] = 0.0
    p_safe = {n: 1.0 - p for n, p in p_fail.items()}
    # 候选故障事件
    all_faults = (
          [f"L{i}"  for i in range(1, 60)]
        + [f"U{i}"  for i in range(1, 63)]
        + [f"S{i}"  for i in range(1, 30)] + ["S13-1", "S29-2", "S62-3"]
        + ["CB1", "CB2", "CB3"]
        + [f"DG{i}" for i in range(1, 9)]
    )
    single_args = [x for x in all_faults if x in base_graph.nodes]
    double_args = [
        (a, b) for a, b in combinations(all_faults, 2)
        if device_category(a) != device_category(b)
        and a in base_graph.nodes and b in base_graph.nodes
    ]
    # 按设备类别做条件化（“同类最多一个故障”）
    devices_by_category = {}
    for j in all_faults:
        if j in base_graph.nodes:
            category = device_category(j)
            if category not in devices_by_category:
                devices_by_category[category] = []
            devices_by_category[category].append(j)
    category_probs = {}
    for category, devices in devices_by_category.items():
        prob_no_fault = np.prod([p_safe[d] for d in devices])
        prob_single_fault = sum(
            p_fail[d] * np.prod([p_safe[o] for o in devices if o != d])
            for d in devices
        )
        category_probs[category] = {
            'no_fault': prob_no_fault,
            'single_fault': prob_single_fault,
            'devices': devices
        }
    # 类内条件概率
    p_cond_category = {}
    for j in all_faults:
        if j in base_graph.nodes:
            category = device_category(j)
            cat_info = category_probs[category]
            p_j_fail = p_fail[j]
            others   = [d for d in cat_info['devices'] if d != j]
            p_others_safe = np.prod([p_safe[d] for d in others])
            denom = cat_info['no_fault'] + cat_info['single_fault']
            p_cond_category[j] = (p_j_fail * p_others_safe) / denom
    # 类间“其余类别无故障”的条件化
    category_no_fault_probs = {
        cat: info['no_fault'] / (info['no_fault'] + info['single_fault'])
        for cat, info in category_probs.items()
    }
    prod_safe_all = np.prod(list(category_no_fault_probs.values()))
    p_cond = {}
    for j in all_faults:
        if j in base_graph.nodes:
            category = device_category(j)
            other_prob = 1.0
            for oc in category_no_fault_probs:
                if oc != category:
                    other_prob *= category_no_fault_probs[oc]
            p_cond[j] = p_cond_category[j] * other_prob
    # 事件列表 + 权重
    event_single = single_args
    w_single = np.array([p_cond[x] for x in event_single])
    event_double = double_args
    w_double = np.array([p_cond[a] * p_cond[b] / prod_safe_all for a, b in event_double])
    events = event_single + event_double
    weights = np.concatenate([w_single, w_double])
    cdf = np.cumsum(weights)
    total_prob = float(cdf[-1]) if len(cdf) else 0.0

    return p_cond, prod_safe_all, events, cdf, total_prob

# ========= worker 初始化（广播抽样表 + 基础图） =========
def init_worker(base_graph, p_cond, prod_safe, events, cdf, total_prob, rng_seed=2025):
    """每个 worker 启动时拿到共享的抽样表和基础图"""
    global BASE_GRAPH, P_COND, PROD_SAFE, EVENTS, CDF, TOTAL_PROB, RNG
    BASE_GRAPH = base_graph
    P_COND = p_cond
    PROD_SAFE = prod_safe
    EVENTS = events
    CDF = cdf
    TOTAL_PROB = total_prob

    # 独立可复现的随机种子
    pid = os.getpid()
    seed = _u32(rng_seed + (pid % 1000))     # <== 32位
    RNG = np.random.default_rng(seed)

def create_operator_with_capacities(dg_capacities):
    """每次从 BASE_GRAPH 复制并覆盖 DG 容量，再构造 NetworkOperator"""
    G_cap = copy.deepcopy(BASE_GRAPH)
    for n, d in G_cap.nodes(data=True):
        if d.get('type') == '分布式能源':
            G_cap.nodes[n]['容量(kW)'] = float(dg_capacities[n])
    return NetworkOperator(G_cap)

# simulate_single：simulate_fault接口函数
# def simulate_single(args):
#     """修改为接收(loc, dg_capacities)元组"""
#     loc, dg_capacities = args
    
#     # 根据DG容量创建NetworkOperator
#     op = create_operator_with_capacities(dg_capacities)
    
#     ftype = op._determine_fault_type(loc)
#     lost, over, tot, strat, overload = op.simulate_fault(
#         ftype, loc, reset=False)
#     w = P_COND[loc]
#     return loc, lost, over, tot, strat, overload, w
def _state_key(net):
    faults = tuple(sorted(n for n, d in net.nodes(data=True)
                          if d.get('故障状态', False)))
    sws = tuple(sorted(
        (n, net.nodes[n].get('开关状态'))
        for n, d in net.nodes(data=True)
        if d.get('type') in ('分段开关', '联络开关', '断路器')
    ))
    statekey = _state_key
    return faults, sws

# simulate_double：simulate_two_faults接口函数
# def simulate_double(args):
    
#     """修改为接收(pair, dg_capacities)元组"""
#     pair, dg_capacities = args
#     loc1, loc2 = pair
   
   
#     # 根据DG容量创建NetworkOperator
#     op = create_operator_with_capacities(dg_capacities)
   
#     lost, over, tot, strat = op.simulate_two_faults(loc1, loc2)
#     w = P_COND[loc1] * P_COND[loc2] / PROD_SAFE
#     return loc1, loc2, lost, over, tot, strat, w

# ========= 抽样与场景评估（worker 用） =========
def _draw():
    r = RNG.random()
    idx = np.searchsorted(CDF, r * TOTAL_PROB, side='right')
    return EVENTS[min(idx, len(EVENTS)-1)]
# ---- 每个 worker 复用 OP（按容量指纹缓存）----
OPERATOR = None
OP_CAP_SIG = None

def _cap_sig(d):
    # 容量向量的稳定指纹
    return tuple((k, round(float(v), 6)) for k, v in sorted(d.items()))

def _get_operator(dg_capacities):
    global OPERATOR, OP_CAP_SIG
    sig = _cap_sig(dg_capacities)
    if OPERATOR is None or OP_CAP_SIG != sig:
        OPERATOR = create_operator_with_capacities(dg_capacities)
        OP_CAP_SIG = sig
    return OPERATOR
def simulate_single(args):
    loc, dg_capacities = args
    op = _get_operator(dg_capacities)  # ← 复用
    ftype = op._determine_fault_type(loc)
    lost, over, tot, strat, overload = op.simulate_fault(ftype, loc, reset=False)
    w = P_COND[loc]
    return loc, lost, over, tot, strat, overload, w

def simulate_double(args):
    pair, dg_capacities = args
    loc1, loc2 = pair
    op = _get_operator(dg_capacities)  # ← 复用
    lost, over, tot, strat = op.simulate_two_faults(loc1, loc2)
    w = P_COND[loc1] * P_COND[loc2] / PROD_SAFE
    return loc1, loc2, lost, over, tot, strat, w

def run_scenario(args):
    _dummy, dg_capacities = args
    s = _draw()
    if isinstance(s, tuple):
        loc1, loc2 = s
        _, _, lost, over, tot, *_ = simulate_double(((loc1, loc2), dg_capacities))
    else:
        loc = s
        _, lost, over, tot, *_ = simulate_single((loc, dg_capacities))
    return lost, over, tot

def run_batch(args):
    """
    每个worker一次性完成 n_batch 个样本，减少IPC和pickle开销。
    args: (base_seed, dg_capacities, n_batch, batch_id)
    返回: (loss_sum, over_sum, tot_sum, n_done)
    """
    base_seed, dg_capacities, n_batch, batch_id = args

    # 每批有独立RNG，保证可复现且与批数量无关
    rng = np.random.default_rng(_u32(base_seed + 0x9E3779B1 * (batch_id + 1)))

    loss_sum = 0.0
    over_sum = 0.0
    tot_sum  = 0.0

    # 直接用抽样表（EVENTS/CDF/TOTAL_PROB）+ 本地rng采样
    for _ in range(n_batch):
        r = rng.random()
        idx = np.searchsorted(CDF, r * TOTAL_PROB, side='right')
        s = EVENTS[min(idx, len(EVENTS) - 1)]

        if isinstance(s, tuple):
            loc1, loc2 = s
            _, _, lost, over, tot, *_ = simulate_double(((loc1, loc2), dg_capacities))
        else:
            loc = s
            _, lost, over, tot, *_ = simulate_single((loc, dg_capacities))

        loss_sum += lost
        over_sum += over
        tot_sum  += tot

    return loss_sum, over_sum, tot_sum, n_batch

# ========= 复用全局池的 Monte Carlo（主进程用） =========
# ========= 复用全局池，但用“完全遍历”替换 Monte Carlo =========
def monte_carlo_simulation(dg_capacities,
                           total_samples=4000,
                           rng_seed=2026,
                           batches=None):
    """
    穷举：单故障 + 跨类别双故障。
    机制保持原状：仍用 simulate_single / simulate_double 两段 imap_unordered。
    新增：进度打印（步数触发 + 每5秒触发）并 flush=True。
    """
    import time
    from itertools import combinations

    global GLOBAL_POOL, GLOBAL_POOL_INITIALIZED
    global GLOBAL_EVENTS, GLOBAL_CDF, GLOBAL_TOTAL_PROB, GLOBAL_P_COND, GLOBAL_PROD_SAFE

    t0_all = time.perf_counter()

    # 1) 首次计算概率表（含 P_COND、PROD_SAFE），后续复用
    if GLOBAL_P_COND is None or GLOBAL_PROD_SAFE is None or GLOBAL_EVENTS is None:
        GLOBAL_P_COND, GLOBAL_PROD_SAFE, GLOBAL_EVENTS, GLOBAL_CDF, GLOBAL_TOTAL_PROB = \
            _compute_sampling_tables(G)

    # 2) 如进程池未初始化则创建（只一次），通过 initializer 广播只读数据
    if not GLOBAL_POOL_INITIALIZED:
        num_workers = max(1, os.cpu_count() - 1)  # 按机器核数调整
        GLOBAL_POOL = mp.Pool(
            num_workers,
            initializer=init_worker,
            initargs=(G, GLOBAL_P_COND, GLOBAL_PROD_SAFE, GLOBAL_EVENTS,
                      GLOBAL_CDF, GLOBAL_TOTAL_PROB, _u32(rng_seed))
        )
        GLOBAL_POOL_INITIALIZED = True

    # Debug：当前这次评估的 DG 容量
    print(dg_capacities, flush=True)

    # 3) 事件清单（单/双故障；双故障只取跨类别）
    all_faults = (
          [f"L{i}"  for i in range(1, 60)]
        + [f"U{i}"  for i in range(1, 63)]
        + [f"S{i}"  for i in range(1, 30)] + ["S13-1", "S29-2", "S62-3"] + ["CB1", "CB2", "CB3"]
        + [f"DG{i}" for i in range(1, 9)]
    )

    single_args = [x for x in all_faults if x in G.nodes]
    double_args = [
        (a, b) for a, b in combinations(all_faults, 2)
        if device_category(a) != device_category(b)
        and a in G.nodes and b in G.nodes
    ]

    # 4) 并行遍历（单故障 → simulate_single；双故障 → simulate_double）
    exp_loss = 0.0
    exp_over = 0.0
    exp_risk = 0.0

    # ---------- 单故障 ----------
    total_single = len(single_args)
    if total_single > 0:
        t0 = time.perf_counter()
        last_print = t0
        # 约打印 ~20 次，至少每 100 个样本打印一次
        step_single = max(1, min(100, total_single // 20))

        cnt = 0
        for loc, lost, over, tot, strat, overload, w in GLOBAL_POOL.imap_unordered(
                simulate_single, [(loc, dg_capacities) for loc in single_args], chunksize=1):
            exp_loss += lost * w
            exp_over += over * w
            exp_risk += tot  * w
            cnt += 1

            now = time.perf_counter()
            if (cnt % step_single == 0) or (now - last_print > 5.0) or (cnt == total_single):
                elapsed = now - t0
                rate = cnt / max(elapsed, 1e-9)
                eta = (total_single - cnt) / max(rate, 1e-9)
                pct = 100.0 * cnt / total_single
                print(f"[单故障] {cnt}/{total_single} ({pct:5.1f}%) | "
                      f"E_over={exp_over:.4f} E(risk)={exp_risk:.4f} | "
                      f"elapsed {elapsed:6.1f}s ETA {eta:6.1f}s",
                      flush=True)
                last_print = now

    # ---------- 双故障（跨类别；仅遍历不含'L'，并按比例外推） ----------
    ESTIMATE_DOUBLE_RATIO = 0.469760170738439

    # 仅保留不含 'L' 的双故障组合（排除所有以 'L' 开头的元件，如 Lxx、LT_xx 等）
    double_args_nol = [(a, b) for (a, b) in double_args
                    if not (a.startswith('L') or b.startswith('L'))]
    total_double = len(double_args_nol)

    sum_lossw = 0.0
    sum_overw = 0.0
    sum_totw  = 0.0

    if total_double > 0:
        t0 = time.perf_counter()
        last_print = t0
        step_double = max(1, min(100, total_double // 20))

        cnt = 0
        for loc1, loc2, lost, over, tot, strat, w in GLOBAL_POOL.imap_unordered(
                simulate_double, [((a, b), dg_capacities) for (a, b) in double_args_nol], chunksize=8):
            # 仅累计“不含L”的双故障加权和
            sum_lossw += lost * w
            sum_overw += over * w
            sum_totw  += tot  * w
            cnt += 1

            # 进度打印：用“不含L”遍历的外推值进行预估显示
            now = time.perf_counter()
            if (cnt % step_double == 0) or (now - last_print > 5.0) or (cnt == total_double):
                elapsed = now - t0
                rate = cnt / max(elapsed, 1e-9)
                eta = (total_double - cnt) / max(rate, 1e-9)
                pct = 100.0 * cnt / total_double
                approx_double_risk = sum_totw / ESTIMATE_DOUBLE_RATIO
                approx_E_over = exp_over + sum_overw / ESTIMATE_DOUBLE_RATIO
                print(f"[双故障(无'L'外推)] {cnt}/{total_double} ({pct:5.1f}%) | "
                    f"E_over≈{approx_E_over:.4f} E(risk)≈{(exp_risk + approx_double_risk):.4f} | "
                    f"elapsed {elapsed:6.1f}s ETA {eta:6.1f}s",
                    flush=True)
                last_print = now

    # 外推：把不含'L'的加权和按比例放大，代表整体双故障，再与单故障累加
    exp_loss += sum_lossw / ESTIMATE_DOUBLE_RATIO
    exp_over += sum_overw / ESTIMATE_DOUBLE_RATIO
    exp_risk += sum_totw  / ESTIMATE_DOUBLE_RATIO

    elapsed_all = time.perf_counter() - t0_all
    print(f"遍历完成：单{len(single_args)}+双{len(double_args)}个场景；"
          f"E(loss)={exp_loss:.4f}, E(over)={exp_over:.4f}, E(risk)={exp_risk:.4f}，"
          f"总耗时{elapsed_all:.2f}s", flush=True)

    # 与原接口一致：返回期望风险
    return float(exp_risk)

# ========= 用户需提供的黑盒损失函数 =========
def loss_fn(t: int, x_t: np.ndarray, mc_seed: int | None = None) -> float:
    dg_capacities = {f'DG{k+1}': float(x_t[k]) for k in range(8)}
    return float(monte_carlo_simulation(
        dg_capacities,
        rng_seed = 0 if mc_seed is None else _u32(mc_seed)
    ))
def make_segmented_loss_evaluator(
    lo: np.ndarray,
    up: np.ndarray,
    breaks,                         # {(t,k): [..]} 或 {k:[..]} 或 [..]
    representative: str = "midpoint",   # "midpoint" | "left" | "right"
    base_seed: int = 0,                 # 固定每段的 MC 随机种子
    include_mc_seed_in_cache: bool = False,  # True 时把 mc_seed 也并入缓存键（命中率会下降）
    monte_carlo_func=None,              # 默认用全局 monte_carlo_simulation(dg_capacities, rng_seed=..)
):
    """
    返回:
      loss_eval(t:int, x_t:np.ndarray, mc_seed:int|None) -> float
      helper: 一个小工具对象，含：
        - edges_of(t,k) -> List[float]
        - segment_of(t,k,x) -> int
        - rep_of(t,k,seg_idx) -> float
        - cache_size() -> int
        - clear_cache() -> None
        - eval_key(t, seg_tuple[8]) -> float  # 强制计算某段并缓存
    约定:
      - 分段使用左闭右开 [L,R)，x==up 归入最后一段。
      - breaks 支持三种写法：
          {(t,k): [b...]}, {k:[b...]}, 或 [b...]
        实际有效断点为 (lo,up) 内部的断点 + 两端 lo/up，并自动去重。
      - 默认忽略外部 mc_seed，使得“段内常量”且强缓存命中最大化。若你确实需要让
        不同 mc_seed 触发不同评估，把 include_mc_seed_in_cache 设 True（不推荐）。
    """
    assert lo.shape == (12, 8) and up.shape == (12, 8)
    lo = np.asarray(lo, float)
    up = np.asarray(up, float)

    # ========== 小工具：32bit 安全混合 ==========
    U32 = 0xFFFFFFFF
    def _u32(x: int) -> int:
        return int(x) & U32
    def _mix(seed: int, *vals: int) -> int:
        s = _u32(seed)
        for v in vals:
            s = _u32((s ^ 0x9E3779B1) * 0x85EBCA77 + int(v) * 0xC2B2AE3D)
        return s

    # ========== 规范化断点 breaks 到 {(t,k): [...]} ==========
    def _norm_breaks(bks):
        if isinstance(bks, list):
            arr = sorted(float(x) for x in bks)
            return {(t, k): arr[:] for t in range(1, 13) for k in range(1, 9)}
        assert isinstance(bks, dict)
        # 可能是 {(t,k):[..]} 或 {k:[..]}
        keyed_by_tk = any(isinstance(k, tuple) and len(k) == 2 for k in bks)
        out = {}
        if keyed_by_tk:
            for (t, k), arr in bks.items():
                out[(int(t), int(k))] = sorted(float(x) for x in arr)
        else:
            for k, arr in bks.items():
                arr = sorted(float(x) for x in arr)
                for t in range(1, 13):
                    out[(t, int(k))] = arr[:]
        return out

    BK = _norm_breaks(breaks)

    # ========== 构造每个(t,k)的 edges（含 lo/up） ==========
    # edges[t-1][k-1] = [e0=lo, e1, ..., eM=up]
    edges: list[list[list[float]]] = [[[] for _ in range(8)] for __ in range(12)]
    for t in range(1, 13):
        for k in range(1, 9):
            lo_tk = float(lo[t-1, k-1])
            up_tk = float(up[t-1, k-1])
            if up_tk < lo_tk + 1e-12:  # 退化为单点
                edges[t-1][k-1] = [lo_tk, lo_tk]
                continue
            inner = [b for b in BK.get((t, k), []) if lo_tk < b < up_tk]
            arr = [lo_tk] + inner + [up_tk]
            # 去重
            dedup = [arr[0]]
            for v in arr[1:]:
                if v - dedup[-1] > 1e-9:
                    dedup.append(v)
            if len(dedup) == 1:  # 极端保护
                dedup = [lo_tk, up_tk]
            edges[t-1][k-1] = dedup

    # ========== 段索引/代表点 ==========
    def _seg_index(t: int, k: int, x: float) -> int:
        """左闭右开：找到 i 使 edges[i] <= x < edges[i+1]；x==up 归入最后一段"""
        ed = edges[t-1][k-1]
        pos = bisect_left(ed, x)   # 第一个 >= x 的位置
        i = pos - 1
        if i < 0:
            i = 0
        if i > len(ed) - 2:
            i = len(ed) - 2
        return i

    def _rep_point(t: int, k: int, i: int) -> float:
        ed = edges[t-1][k-1]
        a, b = ed[i], ed[i+1]
        if representative == "left":
            return a
        if representative == "right":
            return np.nextafter(b, a)  # 轻移到右端内侧
        return 0.5 * (a + b)          # midpoint

    # ========== 默认 Monte Carlo 函数 ==========
    if monte_carlo_func is None:
        # 使用全局已定义的 monte_carlo_simulation(dg_capacities, rng_seed)
        def monte_carlo_func(dg_capacities: dict, rng_seed: int) -> float:
            return float(monte_carlo_simulation(dg_capacities, rng_seed=rng_seed))

    cache: dict[tuple, float] = {}

    # ---------- 缓存指纹：确保不会把“别的题”的缓存装进来 ----------
    import hashlib, json, pickle, gzip, os, time

    def _problem_fingerprint() -> str:
        """
        把影响 loss_eval 结果的核心信息做成短哈希：
          - 代表点策略/是否含 mc_seed
          - 分段边界 edges（每个t,k的边列表）
          - lo/up（限制域）
          - base_seed（决定MC的可复现序列）
        注意：若你改了 Monte Carlo 的实现/抽样表构造，也建议换 base_seed 或在外层换文件名。
        """
        payload = {
            "rep": representative,
            "include_mc_seed": bool(include_mc_seed_in_cache),
            "base_seed": int(base_seed),
            "edges": [[[float(x) for x in edges[t][k]] for k in range(8)] for t in range(12)],
            "lo": lo.tolist(),
            "up": up.tolist(),
        }
        s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

    def _save_cache(path: str) -> dict:
        """把当前内存cache落盘（gzip+pickle）。返回meta，方便打印日志。"""
        meta = {
            "version": 1,
            "fingerprint": _problem_fingerprint(),
            "ts": time.time(),
            "size": len(cache),
        }
        obj = {"meta": meta, "data": cache}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return meta

    def _load_cache(path: str, strict: bool = True, merge: str = "union") -> dict:
        """
        从磁盘加载缓存：
          - strict=True：指纹不匹配则拒绝加载并返回空meta
          - merge:
              "replace" 覆盖当前内存cache；
              "update"  以磁盘为准覆盖当前相同key；
              "union"   只补充当前没有的key（推荐）
        返回已加载的meta；如果文件不存在或被拒绝，返回 {}。
        """
        if not os.path.exists(path):
            return {}
        try:
            with gzip.open(path, "rb") as f:
                obj = pickle.load(f)
            meta = obj.get("meta", {})
            data = obj.get("data", {})
            if strict and meta.get("fingerprint") != _problem_fingerprint():
                # 指纹不一致，拒绝加载
                return {}
            if merge == "replace":
                cache.clear()
                cache.update(data)
            elif merge == "update":
                cache.update(data)
            else:  # "union"
                for k, v in data.items():
                    cache.setdefault(k, v)
            return meta
        except Exception:
            return {}

    def _cache_stats() -> dict:
        return {
            "size": len(cache),
            "fingerprint": _problem_fingerprint(),
        }

    def _eval_key(t: int, seg_tuple: tuple[int, ...], mc_seed: int | None = None) -> float:
        if include_mc_seed_in_cache:
            key = (int(t),) + tuple(int(s) for s in seg_tuple) + (int(mc_seed or 0),)
        else:
            key = (int(t),) + tuple(int(s) for s in seg_tuple)

        v = cache.get(key, None)
        if v is not None:
            return v

        # 代表点 -> DG 容量向量
        x_rep = np.zeros(8, dtype=float)
        for k in range(1, 9):
            i = seg_tuple[k-1]
            x_rep[k-1] = _rep_point(t, k, i)
        dg_cap = {f"DG{k}": float(x_rep[k-1]) for k in range(1, 9)}

        # 统一的可复现 seed：默认不使用外部 mc_seed
        seed = _mix(base_seed, t, *seg_tuple)
        if include_mc_seed_in_cache and mc_seed is not None:
            seed = _mix(seed, mc_seed)

        v = monte_carlo_func(dg_cap, rng_seed=_u32(seed))
        cache[key] = float(v)
        return float(v)

    # ========== 对外的 loss_eval(t, x_t, mc_seed) ==========
    def loss_eval(t: int, x_t: np.ndarray, mc_seed: int | None = None) -> float:
        idxs = []
        xt = np.asarray(x_t, float).reshape(-1)
        assert xt.shape[0] == 8
        for k in range(1, 9):
            idxs.append(_seg_index(t, k, float(xt[k-1])))
        return _eval_key(t, tuple(idxs), mc_seed=mc_seed)

    # ========== helper 工具 ==========
        # === 工具 helper：供 PSO 预计算/缓存用（与 t 无关的场景分段） ===
    class _Helper:
        def edges_of(self, t: int, k: int) -> list[float]:
            # 本设计与 t 无关，按 k 返回“绝对出力 y”的分段边界
            return edges_y[k-1][:]

        def segment_of(self, t: int, k: int, x: float) -> int:
            # 把决策 x 映射为当期绝对出力 y=P+x，再定位段索引
            y = float(P_seq[t-1, k-1] + x)
            return _seg_idx_y(k, y)

        def eval_key(self, t: int, seg_tuple: tuple[int, ...], mc_seed: int | None = None) -> float:
            # 直接以“段键”评估一次，并写入缓存（与 loss_eval 使用同一把缓存）
            key = seg_tuple if not include_mc_seed_in_cache else seg_tuple + (int(mc_seed or 0),)
            v = cache.get(key)
            if v is not None:
                return v
            # 代表点（绝对出力域），并夹到当期可行区间
            y_rep = np.array([_rep_y(k+1, seg_tuple[k]) for k in range(8)], dtype=float)
            y_rep = _clamp_to_feasible_y_for_t(t, y_rep)
            dg_cap = {f"DG{k+1}": float(y_rep[k]) for k in range(8)}

            seed = _mix(base_seed, *seg_tuple)
            if include_mc_seed_in_cache and mc_seed is not None:
                seed = _mix(seed, mc_seed)

            v = monte_carlo_func(dg_cap, rng_seed=_u32(seed))
            cache[key] = float(v)
            return float(v)

        def cache_size(self) -> int:
            return len(cache)

        def clear_cache(self):
            cache.clear()

        def fingerprint(self) -> str:
            return _problem_fingerprint()

        def save_cache(self, path: str) -> dict:
            return _save_cache(path)

        def load_cache(self, path: str, strict: bool = True, merge: str = "union") -> dict:
            return _load_cache(path, strict=strict, merge=merge)

        def cache_stats(self) -> dict:
            return _cache_stats()

    # —— 关键：把 evaluator 和 helper 返回给调用方 —— 
    return loss_eval, _Helper()

# ======= 结束（一站式评估器） =======
# ========= 约束与可行化（repair/projection） =========
def build_bounds(P_seq: np.ndarray, Cpv) -> tuple[np.ndarray, np.ndarray]:
    """
    P_seq: shape (12, 8)  —— 对应 -x_t <= P(t)  ==>  x_t >= -P(t)
    Cpv:   shape (8,) 或标量 —— 盒约束：-0.15*Cpv <= x_t <= 0.15*Cpv
    返回：
      lo: shape (12, 8)
      up: shape (12, 8)
    """
    Cpv = np.asarray(Cpv, dtype=float).reshape(1, -1)  # (1,8)
    if Cpv.shape[1] != P_seq.shape[1]:
        raise ValueError("Cpv 维度与 P_seq 列数不一致")
    up_box = 0.15 * Cpv           # (1,8)
    lo_box = -0.15 * Cpv          # (1,8)
    lo_P   = -np.asarray(P_seq)   # (12,8)
    lo = np.maximum(lo_box, lo_P) # (12,8): 同时满足 x_t >= -P(t) 与 -0.15*Cpv
    up = np.broadcast_to(up_box, P_seq.shape)  # (12,8)
    return lo, up

def project_prefix_feasible(X: np.ndarray) -> np.ndarray:
    """
    将 X (12,8) 投影/修复到满足逐维前缀和约束：forall j, sum_{t=1..j} X[t,k] <= 0
    算法：逐维扫描，维护累计值 cum_k，强制第 t 步的上界 <= -cum_k
    注意：调用前应已做盒约束裁剪
    """
    T, D = X.shape
    X = X.copy()
    cum = np.zeros(D)
    for t in range(T):
        # 当前步必须满足：X[t] <= -cum（逐维）
        hi_allowed = -cum
        # 如果某维 hi_allowed < 当前值，则向下裁剪
        mask = X[t] > hi_allowed
        X[t, mask] = hi_allowed[mask]
        # 更新累计
        cum += X[t]
        # 保护性数值稳定：允许微小正偏差被拉回
        cum = np.minimum(cum, 0.0)
    return X

def random_feasible(lo: np.ndarray, up: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    采样一个可行解 X (12,8)，满足盒约束与前缀和约束
    策略：逐维构造。对每一维 k，按时间递推，hi_t = min(up[t,k], -cum_k)，
          再在 [lo[t,k], hi_t] 均匀采样；若区间空则贴边 hi_t。
    """
    T, D = lo.shape
    X = np.zeros((T, D), dtype=float)
    for k in range(D):
        cum = 0.0
        for t in range(T):
            hi_t = min(up[t, k], -cum)   # 前缀和≤0
            lo_t = float(lo[t, k])
            if hi_t < lo_t:
                # 区间空，则取 hi_t（最小破坏）
                x_tk = hi_t
            else:
                x_tk = rng.uniform(lo_t, hi_t)
            X[t, k] = x_tk
            cum += x_tk
    return X
def pso_optimize(P_seq: np.ndarray,
                 Cpv,
                 swarm_size: int = 80,
                 iters: int = 2000,
                 w_start: float = 0.85,
                 w_end: float   = 0.35,
                 c1: float = 1.6,
                 c2: float = 1.8,
                 seed: int = 2025,
                 checkpoint_path: str | None = None,
                 checkpoint_interval: int = 100,
                 resume: bool = False,
                 # ===== 分段/缓存控制 =====
                 segment_breaks: dict[int, list[float]] | dict[tuple[int,int], list[float]] | list[float] | None = None,
                 representative: str = "midpoint",      # "midpoint" | "left" | "right"
                 precompute: str = "auto",              # "none" | "auto" | "full"
                 precompute_limit_per_t: int = 4096,    # 每个时段允许的全预计算上限
                 include_mc_seed_in_cache: bool = False,
                 base_seed_for_segments: int = 12345,
                 # ===== 实体段缓存 =====
                 segment_cache_path: str | None = None,   # 不给就用指纹自动命名
                 segment_cache_autoload: bool = True,     # 若存在则自动加载
                 segment_cache_autosave: bool = False      # 预计算后/每次checkpoint后自动保存
                 ):
    """
    优化变量：X ∈ R^{12×8}（第13时段固定0，不参与优化）
    目标：sum_{t=1..12} loss_eval(t, X[t], ...)
    约束：
      1) 逐元素盒约束：lo <= X <= up，其中 lo = max(-P(t), -0.15*Cpv), up = 0.15*Cpv
      2) 逐维前缀和：forall j, sum_{t=1..j} X[t,k] <= 0

    段缓存：
      - `segment_cache_path` 指定路径；若为 None，将按 seg_helper 指纹自动生成：
      - 自动加载/保存由 `segment_cache_autoload` / `segment_cache_autosave` 控制。
    """
    import os, pickle, time
    import numpy as np
    from itertools import product

    # ---------- 32位安全的种子工具 ----------
    U32_MASK = 0xFFFFFFFF
    def _u32(x: int) -> int:
        return int(x) & U32_MASK

    def _mix_seed(base: int, i: int, t: int) -> int:
        s = (base ^ 0x9E3779B1) * 0x85EBCA77
        s ^= (i + 1) * 0xC2B2AE3D
        s ^= (t + 1) * 0x27D4EB2F
        return _u32(s)

    # ---------- checkpoint 工具 ----------
    def _save_ckpt(path: str, state: dict):
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_ckpt(path: str) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _rng_from_state(state_dict):
        rng = np.random.default_rng()
        rng.bit_generator.state = state_dict
        return rng

    # ---------- 基本设置 ----------
    rng = np.random.default_rng(seed)
    T, D = 12, 8
    if P_seq.shape != (T, D):
        raise ValueError("P_seq 需要形状 (12, 8)")

    # 边界
    lo, up = build_bounds(P_seq, Cpv)  # (12,8)

    # ====== 分段评估器：创建 loss_eval 与 helper ======
    # 默认使用你的“真实断点”（绝对断点，左闭右开，最后段右端点并入）
    if segment_breaks is None:
        segment_breaks = {
            1: [0, 420, 750, 1035],
            2: [0, 120, 180, 1035],
            3: [0, 320, 420, 1035],
            4: [0, 130, 280, 1035],
            5: [0, 300, 490, 1035],
            6: [0, 420, 770, 1035],
            7: [0, 620, 1035],
            8: [0, 350, 550, 1035],
        }

    loss_eval, seg_helper = make_scenario_loss_evaluator(
    lo, up, P_seq,
    breaks=segment_breaks,
    representative=representative,
    base_seed=base_seed_for_segments,
    include_mc_seed_in_cache=include_mc_seed_in_cache,
    monte_carlo_func=None,
    # === 新增：让缓存与 cap 无关 ===
    scope="global",          # 关键参数
    y_global_min=0.0,        # 你的全局 y 下界（可按需改）
    y_global_max=1035.0,      # 你的全局 y 上界（建议设为扫掠 cap 的上限）
)

    # 1) 优先尝试加载“新文件”的全局预计算缓存（兼容路径；不强校验指纹）
    #    请把下面 compat_cache_path 改成你新文件实际写出的文件路径
    compat_cache_path = os.path.join(SEG_CACHE_DIR, "mc_seg_global.pkl.gz")

    # 2) 再准备一个“指纹化”的本地缓存路径（可继续沿用）
    if segment_cache_path is None:
        segment_cache_path = os.path.join(SEG_CACHE_DIR, f"seg_{seg_helper.fingerprint()}.pkl.gz")

    if segment_cache_autoload:
        os.makedirs(os.path.dirname(segment_cache_path), exist_ok=True)

        # 先融入全局预计算缓存（不严格校验指纹，做并集）
        meta_compat = seg_helper.load_cache(compat_cache_path, strict=False, merge="union")
        if meta_compat:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta_compat.get("ts", time.time())))
            print(f"[段缓存] 已加载全局预计算缓存 {meta_compat.get('size',0)} 键（兼容模式），时间={ts}")

        # 再加载指纹化缓存（若存在则继续合并）
        meta = seg_helper.load_cache(segment_cache_path, strict=True, merge="union")
        if meta:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta.get("ts", time.time())))
            print(f"[段缓存] 已加载指纹缓存 {meta.get('size',0)} 键, 指纹={meta.get('fingerprint','')}, 时间={ts}")
        else:
            print(f"[段缓存] 未找到可用指纹缓存：{segment_cache_path}")


    # ====== 预计算/预热：把常用段提前算进缓存 ======
    def _make_bar(total: int, prefix: str):
        import time as _time
        start = _time.perf_counter()
        last_pct = -1
        def update(done: int):
            nonlocal last_pct
            done = min(done, total)
            pct = int(done * 100 / max(1, total))
            if pct == last_pct and done != total:
                return
            last_pct = pct
            bar_len = 30
            filled = int(bar_len * done / max(1, total))
            bar = "█" * filled + " " * (bar_len - filled)
            elapsed = _time.perf_counter() - start
            eta = (elapsed / done * (total - done)) if done > 0 else 0.0
            print(f"\r{prefix} |{bar}| {pct:3d}%  ({done}/{total})  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s", end="", flush=True)
            if done >= total:
                print("")
        return update

    def _full_precompute_for_t(t: int) -> int:
        seg_counts = [len(seg_helper.edges_of(t, k)) - 1 for k in range(1, 9)]
        total = 1
        for c in seg_counts:
            total *= max(1, c)
        if total <= 0:
            return 0
        ranges = [range(max(1, c)) for c in seg_counts]
        update = _make_bar(total, prefix=f"[预计算] t={t:2d}")
        cnt = 0
        step = max(1, total // 100)
        for seg_tuple in product(*ranges):
            seg_helper.eval_key(t, tuple(int(s) for s in seg_tuple))
            cnt += 1
            if (cnt % step == 0) or (cnt == total):
                update(cnt)
        return cnt

    def _shell_precompute_for_t(t: int, hamming_radius: int = 1) -> int:
        seg_counts = [len(seg_helper.edges_of(t, k)) - 1 for k in range(1, 9)]
        total = 1 + sum(max(0, c - 1) for c in seg_counts)
        if hamming_radius >= 2:
            s1 = [max(0, c - 1) for c in seg_counts]
            add_r2 = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    add_r2 += s1[i] * s1[j]
            total += add_r2
        if total <= 0:
            return 0

        update = _make_bar(total, prefix=f"[预热]   t={t:2d}")

        # 中心段（优先包含0）
        center = []
        for k in range(1, 9):
            ed = seg_helper.edges_of(t, k)
            lo_k, up_k = ed[0], ed[-1]
            x0 = 0.0 if (lo_k <= 0.0 <= up_k) else lo_k
            center.append(seg_helper.segment_of(t, k, x0))
        center = tuple(center)

        done_set = set()
        cnt = 0

        # center
        seg_helper.eval_key(t, center); done_set.add(center); cnt += 1; update(cnt)

        # r=1：单维扰动
        for k in range(1, 9):
            c_k = seg_counts[k - 1]
            for alt in range(max(1, c_k)):
                if alt == center[k - 1]:
                    continue
                st = list(center); st[k - 1] = alt; st = tuple(st)
                if st not in done_set:
                    seg_helper.eval_key(t, st); done_set.add(st); cnt += 1; update(cnt)

        # r=2（可选）
        if hamming_radius >= 2:
            for a in range(8):
                for b in range(a + 1, 8):
                    for alt_a in range(max(1, seg_counts[a])):
                        if alt_a == center[a]:
                            continue
                        for alt_b in range(max(1, seg_counts[b])):
                            if alt_b == center[b]:
                                continue
                            st = list(center); st[a], st[b] = alt_a, alt_b; st = tuple(st)
                            if st not in done_set:
                                seg_helper.eval_key(t, st); done_set.add(st); cnt += 1; update(cnt)
        return cnt

    precompute_counts = []
    if precompute.lower() in ("auto", "full"):
        for t in range(1, 13):
            seg_counts = [len(seg_helper.edges_of(t, k)) - 1 for k in range(1, 9)]
            total = 1
            for c in seg_counts:
                total *= max(1, c)
            if precompute.lower() == "full" or total <= precompute_limit_per_t:
                n = _full_precompute_for_t(t)
                print(f"[预计算] t={t}: 全枚举 {n} 段键（组合数={total}）")
                precompute_counts.append(n)
            else:
                n = _shell_precompute_for_t(t, hamming_radius=1)
                print(f"[预热] t={t}: 半径1壳层 {n} 段键（组合数={total} > 上限 {precompute_limit_per_t}）")
                precompute_counts.append(n)
        print(f"[预计算完成] 总缓存键数量≈{seg_helper.cache_size()}")

        if segment_cache_path and segment_cache_autosave:
            meta = seg_helper.save_cache(segment_cache_path)
            print(f"[段缓存] 预计算后已保存 {meta['size']} 键 → {segment_cache_path}，fp={meta['fingerprint']}")

    # 恢复或初始化
    it_start = 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = _load_ckpt(checkpoint_path)
            if ckpt["lo"].shape == lo.shape and ckpt["up"].shape == up.shape:
                X        = ckpt["X"]
                V        = ckpt["V"]
                pbest_X  = ckpt["pbest_X"]
                pbest_f  = ckpt["pbest_f"]
                gbest_X  = ckpt["gbest_X"]
                gbest_f  = ckpt["gbest_f"]
                it_start = int(ckpt["it"])
                rng      = _rng_from_state(ckpt["rng_state"])
                print(f"[恢复] 从 {checkpoint_path} 续跑：已完成 {it_start}/{iters} 轮，当前 gbest={gbest_f:.4f}")
            else:
                raise RuntimeError("checkpoint 与当前问题维度不匹配")
        except Exception as e:
            print(f"[恢复失败] {e}，重新初始化。")
            X = np.stack([random_feasible(lo, up, rng) for _ in range(swarm_size)], axis=0)
            V = rng.normal(scale=0.1, size=X.shape)
            pbest_X = X.copy()
            base = _u32(seed ^ 0xA5A5A5A5)
            pbest_f = np.array([
                sum(loss_eval(t+1, x[t], mc_seed=_mix_seed(base, i, t)) for t in range(T))
                for i, x in enumerate(pbest_X)
            ])
            g_idx   = int(np.argmin(pbest_f))
            gbest_X = pbest_X[g_idx].copy()
            gbest_f = float(pbest_f[g_idx])
    else:
        X = np.stack([random_feasible(lo, up, rng) for _ in range(swarm_size)], axis=0)
        V = rng.normal(scale=0.1, size=X.shape)
        pbest_X = X.copy()
        base = _u32(seed ^ 0xA5A5A5A5)
        pbest_f = np.array([
            sum(loss_eval(t+1, x[t], mc_seed=_mix_seed(base, i, t)) for t in range(T))
            for i, x in enumerate(pbest_X)
        ])
        g_idx   = int(np.argmin(pbest_f))
        gbest_X = pbest_X[g_idx].copy()
        gbest_f = float(pbest_f[g_idx])

    # 权重线性衰减
    def inertia(it):
        return w_start + (w_end - w_start) * (it / max(1, iters - 1))

    print(f"[{time.strftime('%H:%M:%S')}] 开始PSO优化，群体={swarm_size}，总迭代={iters}，从第 {it_start} 轮继续")

    # checkpoint 帮手
    def _maybe_save_ckpt(it_now: int):
        if not checkpoint_path:
            return
        state = {
            "it": it_now,
            "X": X, "V": V,
            "pbest_X": pbest_X, "pbest_f": pbest_f,
            "gbest_X": gbest_X, "gbest_f": gbest_f,
            "lo": lo, "up": up,
            "rng_state": rng.bit_generator.state,
        }
        _save_ckpt(checkpoint_path, state)

    try:
        for it in range(it_start, iters):
            w = inertia(it)
            r1 = rng.random(size=X.shape)
            r2 = rng.random(size=X.shape)

            # 速度更新
            V = (w * V
                 + c1 * r1 * (pbest_X - X)
                 + c2 * r2 * (gbest_X[np.newaxis, ...] - X))

            # 位置更新 + 盒裁剪
            X = np.minimum(np.maximum(X + V, lo[None, ...]), up[None, ...])

            # 前缀和可行化（逐粒子）
            for i in range(swarm_size):
                X[i] = project_prefix_feasible(X[i])

            # 评估（分段+缓存的 loss_eval）
            iter_seed = _u32((seed ^ 0xDEADBEEF) + (it + 1) * 0x9E3779B1)
            vals = np.empty(swarm_size, dtype=float)
            iter_t0 = time.perf_counter()
            np.set_printoptions(precision=3, suppress=True, linewidth=200)

            chunk_start = time.perf_counter()
            for i in range(swarm_size):
                vals[i] = sum(
                    loss_eval(t+1, X[i, t], mc_seed=_mix_seed(iter_seed, i, t))
                    for t in range(T)
                )
                if ((i + 1) % 5 == 0) or (i + 1 == swarm_size):
                    start_idx = i - (i % 5) + 1
                    end_idx   = i + 1
                    dt = time.perf_counter() - chunk_start
                    print(f"[{time.strftime('%H:%M:%S')}] 迭代 {it+1:>4}/{iters} | 粒子 {start_idx:>3}~{end_idx:<3} 用时 {dt:.2f}s")
                    chunk_start = time.perf_counter()

            # 更新个体/全局最优
            improved = vals < pbest_f
            pbest_X[improved] = X[improved]
            pbest_f[improved] = vals[improved]
            gi = int(np.argmin(pbest_f))
            if pbest_f[gi] < gbest_f:
                gbest_f = float(pbest_f[gi])
                gbest_X = pbest_X[gi].copy()

            iter_dur = time.perf_counter() - iter_t0
            print(f"[{time.strftime('%H:%M:%S')}] 迭代 {it+1:>4}/{iters} 完成，用时 {iter_dur:.2f}s | 当前最优损失 gbest={gbest_f:.6f}")
            print(f"[best X 12×8]\n{gbest_X}\n")

            if (it + 1) % 50 == 0 or it == 0:
                print(f"[{time.strftime('%H:%M:%S')}] PSO迭代 {it+1:>4}/{iters}, gbest={gbest_f:.4f}, w={w:.3f}")

            # 定期保存检查点 + 段缓存
            if checkpoint_path and ((it + 1) % int(checkpoint_interval) == 0):
                _maybe_save_ckpt(it + 1)
            if segment_cache_path and segment_cache_autosave and ((it + 1) % int(checkpoint_interval) == 0):
                meta = seg_helper.save_cache(segment_cache_path)
                print(f"[段缓存] checkpoint 同步保存，size={meta['size']}")

        print(f"[{time.strftime('%H:%M:%S')}] PSO优化完成，最终目标值: {gbest_f:.4f}")
        _maybe_save_ckpt(iters)
        if segment_cache_path and segment_cache_autosave:
            meta = seg_helper.save_cache(segment_cache_path)
            print(f"[段缓存] 最终保存，size={meta['size']} → {segment_cache_path}")

    except KeyboardInterrupt:
        print("\n[中断] 捕获 KeyboardInterrupt，保存检查点后退出。")
        _maybe_save_ckpt(it)  # 保存到当前迭代
        if segment_cache_path and segment_cache_autosave:
            meta = seg_helper.save_cache(segment_cache_path)
            print(f"[段缓存] 中断时保存，size={meta['size']} → {segment_cache_path}")
        raise

    return gbest_f, gbest_X

def make_scenario_loss_evaluator(
    lo: np.ndarray, up: np.ndarray, P_seq: np.ndarray, breaks,
    representative="midpoint",
    base_seed: int = 0,
    include_mc_seed_in_cache: bool = False,
    monte_carlo_func=None,
    scope: str = "global",           # 默认改为 global
    y_global_min=0.0,
    y_global_max=1035.0,
):
    import numpy as np
    from bisect import bisect_left
    import hashlib, json, gzip, pickle, os, time

    assert lo.shape == (12,8) and up.shape == (12,8) and P_seq.shape == (12,8)

    # ---- 断点归一化（保持原逻辑）----
    def _norm_breaks(bks):
        if isinstance(bks, list):
            arr = sorted(float(x) for x in bks)
            return {k: arr[:] for k in range(1,9)}
        assert isinstance(bks, dict)
        keyed_by_tk = any(isinstance(k, tuple) and len(k)==2 for k in bks)
        out = {}
        if keyed_by_tk:
            tmp = {k: [] for k in range(1,9)}
            for (t,k), arr in bks.items():
                tmp[int(k)].extend(arr)
            for k in range(1,9):
                out[k] = sorted(set(float(x) for x in tmp[k]))
        else:
            for k, arr in bks.items():
                out[int(k)] = sorted(float(x) for x in arr)
        return out
    BK = _norm_breaks(breaks)

    # ---- 选择“绝对出力”的包络范围：global（cap无关）或 per_cap（保持旧）----
    if scope == "global":
        y_lo_all = np.broadcast_to(np.asarray(y_global_min, float).reshape(-1), (8,))[:8]
        y_up_all = np.broadcast_to(np.asarray(y_global_max, float).reshape(-1), (8,))[:8]
    else:  # 旧行为：范围依赖 P_seq/lo/up → 会随 cap 变
        y_lo_all = (P_seq + lo).min(axis=0)
        y_up_all = (P_seq + up).max(axis=0)

    # ---- 构造每维的“绝对出力 y 分段边界” edges_y（与 t 无关）----
    edges_y = [[] for _ in range(8)]
    for k in range(8):
        lo_k, up_k = float(y_lo_all[k]), float(y_up_all[k])
        inner = [b for b in BK.get(k+1, []) if lo_k < b < up_k]
        arr = [lo_k] + inner + [up_k]
        dedup = [arr[0]]
        for v in arr[1:]:
            if v - dedup[-1] > 1e-9:
                dedup.append(v)
        if len(dedup) == 1:
            dedup = [lo_k, up_k]
        edges_y[k] = dedup

    # ---- 段索引与代表点 ----
    def _seg_idx_y(k: int, y: float) -> int:
        ed = edges_y[k-1]
        pos = bisect_left(ed, y)
        return max(0, min(pos-1, len(ed)-2))

    def _rep_y(k: int, i: int) -> float:
        ed = edges_y[k-1]; a, b = ed[i], ed[i+1]
        if representative == "left":  return a
        if representative == "right": return np.nextafter(b, a)
        return 0.5*(a+b)

    # ---- 指纹（global 模式下不含任何 cap 相关量）----
    def _problem_fingerprint():
        payload = {
            "rep": representative,
            "include_mc_seed": bool(include_mc_seed_in_cache),
            "base_seed": int(base_seed),
            "scope": scope,
            "edges_y": [[float(x) for x in edges_y[k]] for k in range(8)],
            "y_lo_all": [float(x) for x in y_lo_all],
            "y_up_all": [float(x) for x in y_up_all],
        }
        s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

    def _save_cache(path: str) -> dict:
        meta = {"version": 1, "fingerprint": _problem_fingerprint(), "ts": time.time(), "size": len(cache)}
        obj = {"meta": meta, "data": cache}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return meta

    def _load_cache(path: str, strict: bool = True, merge: str = "union") -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with gzip.open(path, "rb") as f:
                obj = pickle.load(f)
            meta = obj.get("meta", {}); data = obj.get("data", {})
            if strict and meta.get("fingerprint") != _problem_fingerprint():
                return {}
            if merge == "replace":
                cache.clear(); cache.update(data)
            elif merge == "update":
                cache.update(data)
            else:  # union
                for k, v in data.items():
                    cache.setdefault(k, v)
            return meta
        except Exception:
            return {}

    def _cache_stats() -> dict:
        return {"size": len(cache), "fingerprint": _problem_fingerprint()}

    # ---- 默认 MC 函数（保持原逻辑）----
    if monte_carlo_func is None:
        def monte_carlo_func(dg_capacities: dict, rng_seed: int) -> float:
            return float(monte_carlo_simulation(dg_capacities, rng_seed=rng_seed))

    # ---- 段键缓存（与 t 无关）----
    cache = {}
    U32 = 0xFFFFFFFF
    def _u32(x:int)->int: return int(x) & U32
    def _mix(seed:int, *vals:int)->int:
        s = _u32(seed)
        for v in vals:
            s = _u32((s ^ 0x9E3779B1) * 0x85EBCA77 + int(v) * 0xC2B2AE3D)
        return s

    # 按模式夹代表点（global：夹到全局包络；per_cap：夹到当期可行域）
    def _clamp_y_for_eval(t:int, y:np.ndarray)->np.ndarray:
        if scope == "global":
            return np.minimum(np.maximum(y, y_lo_all), y_up_all)
        else:
            y_lo_t = P_seq[t-1] + lo[t-1]
            y_up_t = P_seq[t-1] + up[t-1]
            return np.minimum(np.maximum(y, y_lo_t), y_up_t)

    # ---- 对外：loss_eval ----
    def loss_eval(t:int, x_t:np.ndarray, mc_seed:int|None=None) -> float:
        x_t = np.asarray(x_t, float).reshape(8,)
        y_t = P_seq[t-1] + x_t
        seg = tuple(_seg_idx_y(k+1, float(y_t[k])) for k in range(8))
        key = seg if not include_mc_seed_in_cache else seg + (int(mc_seed or 0),)

        v = cache.get(key)
        if v is not None:
            return v

        y_rep = np.array([_rep_y(k+1, seg[k]) for k in range(8)], float)
        y_rep = _clamp_y_for_eval(t, y_rep)
        dg_cap = {f"DG{k+1}": float(y_rep[k]) for k in range(8)}

        seed = _mix(base_seed, *seg)
        if include_mc_seed_in_cache and mc_seed is not None:
            seed = _mix(seed, mc_seed)

        v = monte_carlo_func(dg_cap, rng_seed=_u32(seed))
        cache[key] = float(v)
        return float(v)

    # ---- Helper（保持接口不变）----
    class _Helper:
        def edges_of(self, t:int, k:int) -> list[float]:
            return edges_y[k-1][:]

        def segment_of(self, t:int, k:int, x:float) -> int:
            y = float(P_seq[t-1, k-1] + x)
            return _seg_idx_y(k, y)

        def eval_key(self, t:int, seg_tuple:tuple[int,...], mc_seed:int|None=None) -> float:
            key = seg_tuple if not include_mc_seed_in_cache else seg_tuple + (int(mc_seed or 0),)
            v = cache.get(key)
            if v is not None:
                return v
            y_rep = np.array([_rep_y(k+1, seg_tuple[k]) for k in range(8)], float)
            y_rep = _clamp_y_for_eval(t, y_rep)
            dg_cap = {f"DG{k+1}": float(y_rep[k]) for k in range(8)}

            seed = _mix(base_seed, *seg_tuple)
            if include_mc_seed_in_cache and mc_seed is not None:
                seed = _mix(seed, mc_seed)

            v = monte_carlo_func(dg_cap, rng_seed=_u32(seed))
            cache[key] = float(v)
            return float(v)

        def cache_size(self) -> int: return len(cache)
        def clear_cache(self): cache.clear()
        def fingerprint(self) -> str: return _problem_fingerprint()
        def save_cache(self, path:str) -> dict: return _save_cache(path)
        def load_cache(self, path:str, strict:bool=True, merge:str="union") -> dict:
            return _load_cache(path, strict=strict, merge=merge)
        def cache_stats(self) -> dict: return _cache_stats()

    return loss_eval, _Helper()

if __name__ == "__main__":
    import os, time, atexit
    import numpy as np
    import pandas as pd
    import multiprocessing as mp

    # —— 多进程与善后 —— 
    mp.set_start_method('spawn', force=True)
    atexit.register(cleanup_global_pool)

    print(f"\n===== 扫掠启动 [{time.strftime('%Y-%m-%d %H:%M:%S')}] =====")
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "决策结果"), exist_ok=True)
    # 如你的 pso_optimize 支持实体段缓存（segment_cache_*），也可建：
    os.makedirs(SEG_CACHE_DIR, exist_ok=True)

    # —— 你的“真实断点”（左闭右开，最后一段右端点并入）——
    ABS_BREAKS = {
        1: [0, 420, 750, 1035],
        2: [0, 120, 180, 1035],
        3: [0, 320, 420, 1035],
        4: [0, 130, 280, 1035],
        5: [0, 300, 490, 1035],
        6: [0, 420, 770, 1035],
        7: [0, 620, 1035],
        8: [0, 350, 550, 1035],
    }
    
    # —— 以 300 kW 情景为基准出力曲线；其它容量按比例缩放 —— 
    power_values_300 = [6.21, 68.46, 158.25, 234.78, 282.39, 300.00,
                        293.04, 267.96, 229.32, 179.16, 119.07, 54.78]
    base_profile = np.array(power_values_300, dtype=float) / 300.0  # 每 kW 的小时出力

    def make_P_seq(cap_kw: float) -> np.ndarray:
        """给定装机 cap_kw -> (12×8) 的小时功率矩阵；8 台同曲线同容量。"""
        row = base_profile * cap_kw           # (12,)
        return np.repeat(row[:, None], 8, axis=1)

    summary = []  # 记录每个容量的最优目标
    best_mats = {}  # cap -> best_X (12x8)

    try:
        for cap in range(30, 901, 10):  # 0,10,...,900
            print(f"\n============= 情景 cap={cap} kW =============")
            P_seq = make_P_seq(cap)
            Cpv   = np.full(8, float(cap))   # 如需每台不同，改成你的 8 维向量

            ckpt = os.path.join(BASE_DIR, f"pso_cap{cap}.pkl")

            # —— 关键：把断点给到 pso，并启用分段预计算（自动/全量）——
            best_f, best_X = pso_optimize(
                P_seq, Cpv,
                swarm_size=80, iters=1200,
                w_start=0.95, w_end=0.25, c1=2.0, c2=2.0, seed=2025,
                checkpoint_path=ckpt, checkpoint_interval=50, resume=True,
                segment_breaks=ABS_BREAKS, representative="midpoint",
                precompute="auto", precompute_limit_per_t=4096,
                include_mc_seed_in_cache=False, base_seed_for_segments=12345,
            )

            # 保存本情景结果
            np.save(os.path.join(BASE_DIR, "决策结果", f"best_X_cap{cap}.npy"), best_X)
            summary.append({"cap_kW": cap, "best_f": float(best_f)})
            
            best_mats[cap] = best_X

    finally:
        # 汇总无论是否中断都尽量写出
        if summary:
            df = pd.DataFrame(summary).sort_values("cap_kW")
            df.to_csv(os.path.join(BASE_DIR, "决策结果", "summary_cap_100_600_step30.csv"), index=False)
            print("\n扫掠完成（或已输出部分进度）。汇总：决策结果\\summary_cap_100_600_step30.csv")
        cleanup_global_pool()
        print(f"程序结束 [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        if summary:
            df = pd.DataFrame(summary).sort_values("cap_kW")
            out_xlsx = os.path.join(BASE_DIR, "决策结果", "summary_and_solutions.xlsx")
            with pd.ExcelWriter(out_xlsx) as writer:  # 让 pandas 自动选用 openpyxl/xlsxwriter
                # 汇总表（每个cap的最优目标值）
                df.to_excel(writer, sheet_name="summary", index=False)

                # 每个cap一个工作表，写12x8的best_X
                for cap in sorted(best_mats):
                    X = best_mats[cap]
                    dfX = pd.DataFrame(
                        X,
                        columns=[f"DG{k}" for k in range(1, 9)],
                        index=[f"t{t}" for t in range(1, 13)]
                    )
                    dfX.to_excel(writer, sheet_name=f"cap_{cap}")

            print(f"\nExcel 写出完成：{out_xlsx}")