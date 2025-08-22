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
HERE = Path(__file__).parent

# ============ 常量 ============
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
    ("LT_13_43", I_RATE, 0.01),   # 对应 S13-1
    ("LT_19_29", I_RATE, 0.01),   # 对应 S29-2
    ("LT_23_62", I_RATE, 0.01),   # 对应 S62-3
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
@njit
def _compute_currents(R_arr, P_arr, U_line):
    n = R_arr.shape[0]
    I = np.zeros(n)
    Psend = np.zeros(n)
    for i in range(n):
        Rk = R_arr[i]
        Pk = P_arr[i]
        if Rk > 0.0 and Pk != 0.0:
            disc = U_line*U_line + 4*Rk*Pk*1e3
            I_val = (SQRT3*(math.sqrt(disc)-U_line)) / (6.0*Rk)
            I[i] = I_val if I_val>0.0 else 0.0
        # 计算送端功率
        Psend[i] = SQRT3*U_line*I[i]/1e3
    return Psend, I

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

        # 1 获取网络状态
        branch_dg_list = self._branch_dgs(loc, self.G)

        # 2 计算用户短路额外损失
        node_data = self.G.nodes[loc]
        if node_data.get('type') == '用户':
            extra_lost = node_data['用电功率(kW)'] * TYPE_W[node_data['类型']]
        else:
            extra_lost = 0

        # 3 构建基础故障网络
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
            lines = [n for n in net0.neighbors(loc)]
            if len(lines) >= 2:
                u, v = lines[0], lines[1]
                net0.add_edge(u, v, 联络边=False)
            base_label = f"短路用户 {loc}"

        # 4 候选策略生成
        candidates = []
        # 公共基础策略
        candidates.append((net0, base_label)) 

        # 故障类型1：打开上游开关
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
            new_net = copy.deepcopy(net0)
            new_net.nodes[found_switch]['开关状态'] = '开'

            # 考虑接入原本由DG供应的用户后可能会出现过负荷的情形
            overload = run_distflow(new_net)
            if not overload:
                candidates.append((new_net, f"打开开关 {found_switch}"))
            else:
                # 添加原始策略
                candidates.append((new_net, f"打开开关 {found_switch}"))
                # 过负荷的情况，添加联络线策略
                switches = ['S8', 'S11', 'S12', 'S1', 'S2', 'S3', 'S4', 'S5', 'S29', 'S27', 'S62-3', 'S13-1', 'S29-2']
                max_switch_num = 6  # 最大同时操作开关数量
                key_ties = {'S13-1', 'S29-2', 'S62-3'}

                # 定义联络开关及其约束关系
                tie_switch_constraints = {}
                tie_switch_constraints = self._judge_main_fault_type(loc)

                for i in range(1, min(max_switch_num, len(switches)) + 1):
                    for combo in combinations(switches, i):
                        # 判断是否包含至少一个关键联络开关
                        if not key_ties.intersection(combo):
                            continue
                        new_net = copy.deepcopy(net0)
                        label_parts = [f"打开开关{found_switch}"]
                        for sw in combo:
                            if sw in new_net.nodes:  
                                if new_net.nodes[sw]['开关状态'] == '关':
                                    new_net.nodes[sw]['开关状态'] = '开'
                                    label_parts.append(f"开{sw}")
                                else:
                                    new_net.nodes[sw]['开关状态'] = '关'
                                    label_parts.append(f"关{sw}")
                        valid = True
                        for sw in tie_switch_constraints:
                            if sw in new_net.nodes and new_net.nodes[sw]['开关状态'] == '开':
                                if not any(new_net.nodes[n]['开关状态'] == '关' for n in tie_switch_constraints[sw]):
                                    valid = False
                                    break
                        if valid:
                            candidates.append((new_net, "操作：" + "+".join(label_parts)))
                
        # 故障类型5：组合开关操作
        elif ftype == 5:
            switches = ['S8', 'S11', 'S12', 'S1', 'S2', 'S3', 'S4', 'S5', 'S29', 'S27', 'S62-3', 'S13-1', 'S29-2']
            max_switch_num = 6  # 最大同时操作开关数量

            # 定义联络开关及其约束关系
            tie_switch_constraints = {}
            tie_switch_constraints = self._judge_main_fault_type(loc)

            for i in range(1, min(max_switch_num, len(switches)) + 1):
                for combo in combinations(switches, i):
                    new_net = copy.deepcopy(net0)
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
                    for sw in tie_switch_constraints:
                        if sw in new_net.nodes and new_net.nodes[sw]['开关状态'] == '开':
                            if not any(new_net.nodes[n]['开关状态'] == '关' for n in tie_switch_constraints[sw]):
                                valid = False
                                break
                    if valid:
                        candidates.append((new_net, "操作：" + "+".join(label_parts)))

        # 故障类型3：断开分支DG
        elif ftype == 3:
            for dg in branch_dg_list:
                if dg not in net0.nodes:
                    continue

                # (1) 向上游 BFS 找最近的分段/联络开关
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

                # (2) 生成临时网络：开 DG，关最近开关
                tmp_net = copy.deepcopy(net0)
                tmp_net.nodes[dg]['开关状态'] = '开'
                if not loc.startswith('L'):
                    tmp_net.nodes[found_switch]['开关状态'] = '关'

                # (3) 删除该开关节点，计算 DG 可到达区域
                H = tmp_net.copy()
                H.remove_node(found_switch)
                reachable = nx.node_connected_component(H, dg)

                # (4) 统计区域内用户负荷（不含故障节点）
                load_sum = 0.0
                for n in reachable:
                    n_data = tmp_net.nodes[n]
                    if n_data['type'] == '用户' and n != loc:
                        load_sum += n_data['用电功率(kW)']

                # (5) 若 DG 能覆盖该负荷，才加入候选策略
                if load_sum <= tmp_net.nodes[dg]['容量(kW)']:
                    candidates.append((tmp_net, f"开{dg} + 断{found_switch}"))

        # 其他类型仅保留基础策略
        else:
            pass

        # 5 评估所有候选策略
        best_risk      = (float('inf'), float('inf'), float('inf'))
        best_strategy  = base_label
        best_overload  = []         

        for net, label in candidates:
            # 检查各 CB 送电区域是否超额
            for cb in ['CB1', 'CB2', 'CB3']:
                if cb not in net.nodes:
                    continue
                if net.nodes[cb].get('故障状态', False):
                    continue

                # 找 CB 可达区域
                visited = set()
                queue = deque([cb])
                total_load = 0.0

                while queue:
                    node = queue.popleft()
                    if node in visited:
                        continue
                    visited.add(node)
                    for nbr in net.neighbors(node):
                        if nbr in visited:
                            continue
                        if net.nodes[nbr].get('故障状态', False):
                            continue
                        if net.nodes[nbr]['type'] in ['分段开关', '联络开关']:
                            if net.nodes[nbr]['开关状态'] == '开':
                                queue.append(nbr)
                        else:
                            queue.append(nbr)

                # 汇总负荷
                for n in visited:
                    n_data = net.nodes[n]
                    if n_data['type'] == '用户':
                        total_load += n_data.get('用电功率(kW)', 0.0)

            # 计算潮流
            overload = run_distflow(net)
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
                    
            net2 = copy.deepcopy(net)
            lost = self._risk(net2, loc)
            total =lost + over_penalty 
            # 结果乘以1e4，保持数据精度
            total = total * 1e4
            lost = lost * 1e4
            over_penalty = over_penalty * 1e4

            # 更新最优策略
            if total < best_risk[2] or (total == best_risk[2] and over_penalty < best_risk[1]):
                best_risk     = (lost, over_penalty, total)
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
            new_net = copy.deepcopy(net0)
            new_net.nodes[found_switch]['开关状态'] = '开'

            # 考虑接入原本由DG供应的用户后可能会出现过负荷的情形
            overload = run_distflow(new_net)
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
                        new_net = copy.deepcopy(net0)
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
                    new_net = copy.deepcopy(net0)
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
        filtered_combined = []
        for combo, ops, label_parts in combined:
            valid = True
            for key, constraint_set in tie_switch_constraints.items():
                # 如果key在combo中，则其约束开关至少有一个也在combo中
                if key in combo:
                    if not any(const_sw in combo for const_sw in constraint_set):
                        valid = False
                        break
            if valid:
                filtered_combined.append((combo, ops, label_parts))
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
            overload = run_distflow(net)
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

# 在子进程内刷新额定电流，全局生效
def update_current(cur):
    global I_RATE, RATED_CURRENT
    I_RATE = RATED_CURRENT = cur
    return os.getpid()

# ============ 主程序 ===============
if __name__ == '__main__':
    # 用于计时
    from time import perf_counter

    # 固定DG容量为300通过改变电流进行灵敏度分析
    CAP_SEQ       = [300.0]
    CURRENT_SEQ  = list(range(190, 251, 10))    # 190 A → 250 A，步长为 10
    PROGRESS_INT = 100                          # 打印频率
    OUT_DIR      = str((HERE / "灵敏度分析_电流_结果").resolve())
    os.makedirs(OUT_DIR, exist_ok=True)

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

        # 计算故障概率
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
                
        # 并行计算风险
        all_rows, sum_p, sum_risk = [], 0.0, 0.0
        sum_loss, sum_over = 0.0, 0.0
        single_args  = [loc for loc in all_faults if loc in op0.G.nodes]
        double_args  = [
            (a, b) for a, b in combinations(all_faults, 2)
            if device_category(a) != device_category(b)
            and a in op0.G.nodes and b in op0.G.nodes
        ]

        cpu_num = max(1, mp.cpu_count() - 1)
        all_rows, sum_p, sum_risk = [], 0.0, 0.0

    with mp.Pool(processes=cpu_num,
                     initializer=init_worker,
                     initargs=(op0.G, p_cond, prod_safe_all)) as pool:

        # 循环：不同额定电流
        for I_cur in CURRENT_SEQ:
            t0 = perf_counter()
            print(f"\n### 灵敏度分析：额定电流 {I_cur} A ###")

            # 广播新载流量到所有 worker
            pids = pool.map(update_current, [I_cur]*cpu_num)
            print("  ⚙ Worker 同步完毕 (pids:", ",".join(map(str, pids)), ")")

            # 累加器归零
            sum_p = sum_loss = sum_over = sum_risk = 0.0
            all_rows = []

            # 单故障计算
            cnt = 0
            total_single = len(single_args)
            for loc, lost, over, tot, strat, overload, w in \
                    pool.imap_unordered(simulate_single, single_args, chunksize=50):

                cnt       += 1
                sum_p     += w
                sum_loss  += lost * w
                sum_over  += over * w
                sum_risk  += tot  * w

                if cnt % PROGRESS_INT == 0 or cnt == total_single:
                    pct = cnt / total_single * 100
                    print(f"    单故障 {cnt}/{total_single}  ({pct:5.1f} %)")
                    print(f"[{I_cur:>3} A]  "f"E(loss)={sum_loss:,.2f}  "f"E(over)={sum_over:,.2f}  "f"E(risk)={sum_risk:,.2f}")


                all_rows.append({
                    "rated_A"       : I_cur,
                    "loc"           : loc,
                    "type"          : "single",
                    "best_strategy" : strat,
                    "lost"          : lost,
                    "over_penalty"  : over,
                    "tot"           : tot,
                    "overload_lines": ";".join(f"{bid}:{I:.1f}A"
                                            for bid, I in overload),
                    "w"             : w
                })

            # 双故障
            cnt = 0
            total_double = len(double_args)
            for loc1, loc2, lost, over, tot, strat, w in \
                    pool.imap_unordered(simulate_double, double_args, chunksize=30):

                cnt       += 1
                sum_p     += w
                sum_loss  += lost * w
                sum_over  += over * w
                sum_risk  += tot  * w

                if cnt % PROGRESS_INT == 0 or cnt == total_double:
                    pct = cnt / total_double * 100
                    print(f"    双故障 {cnt}/{total_double}  ({pct:5.1f} %)")
                    print(f"[{I_cur:>3} A]  "f"E(loss)={sum_loss:,.2f}  "f"E(over)={sum_over:,.2f}  "f"E(risk)={sum_risk:,.2f}")

                all_rows.append({
                    "rated_A"       : I_cur,
                    "loc"           : f"{loc1}+{loc2}",
                    "type"          : "double",
                    "best_strategy" : strat,
                    "lost"          : lost,
                    "over_penalty"  : over,
                    "tot"           : tot,
                    "overload_lines": "-",
                    "w"             : w
                })

            # 结果汇总和输出
            E_loss = sum_loss
            E_over = sum_over
            E_risk = sum_risk

            print(f"[{I_cur:>3} A]  "
                f"E(loss)={E_loss:,.2f}  "
                f"E(over)={E_over:,.2f}  "
                f"E(risk)={E_risk:,.2f}  "
                f"(耗时 {perf_counter()-t0:,.1f}s)")

            df = pd.DataFrame(all_rows)
            df["expected_loss"] = E_loss
            df["expected_over"] = E_over
            df["expected_risk"] = E_risk

            fname = f"电流灵敏度结果_{int(cap)}kW_{I_cur}A.xlsx"
            path  = os.path.join(OUT_DIR, fname)
            os.makedirs(OUT_DIR, exist_ok=True)
            df.to_excel(path, index=False)
            print(f" ✓ 结果已保存至 {path}")

        pool.close(); pool.join()
        print("\n 灵敏度分析全部完成")