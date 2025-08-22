import pandas as pd
import numpy as np
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent

TYPE_W = {'居民':0.067,'商业':0.226,'政府和机构':0.354,'办公和建筑':0.354}   # 不同用户权重
type_dict = {1: '居民', 2: '商业', 3: '政府和机构', 4: '办公和建筑'}
user_type_mapping = [
    1,1,1,1,1,1,4,1,3,1,
    2,4,1,4,1,2,1,4,1,1,
    3,1,1,1,1,1,2,1,3,1,
    2,4,2,2,1,3,1,2,1,4,
    1,2,1,1,3,1,4,1,2,1,
    1,1,2,1,1,2,1,3,1,1,
    3,1
]

# 提取用户编号函数
def extract_user_id(user_str):
    """
    从'U{i}'格式的字符串中提取数字编号
    例如: 'U1' -> 1, 'U25' -> 25
    """
    try:
        # 使用正则表达式提取数字
        match = re.search(r'U(\d+)', str(user_str))
        if match:
            return int(match.group(1))
        else:
            # 如果不是U{i}格式，尝试直接转换为整数
            return int(user_str)
    except:
        print(f"警告: 无法解析用户编号 '{user_str}'")
        return None

# ================= 读取Excel文件第三个sheet =================
def read_probability_data():
    """
    读取Excel文件第三个sheet的用户编号和总风险概率数据
    """
    excel_path = str((HERE / "失负荷和过负荷概率_300kW.xlsx").resolve())
    
    try:
        # 读取第三个sheet (索引为2)
        df = pd.read_excel(excel_path, sheet_name=2)
        print(f"成功读取Excel文件第三个sheet，数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
        return df
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return None

# ================= 概率加权计算 =================
def calculate_weighted_probability(df):
    """
    根据用户类型映射表和权重计算加权概率
    """
    if df is None:
        return None
    
    # 直接使用指定的列名
    user_col = "用户编号"
    prob_col = "总风险概率"
    
    # 检查列名是否存在
    if user_col not in df.columns:
        print(f"错误: 找不到列 '{user_col}'")
        print(f"可用列名: {list(df.columns)}")
        return None
    
    if prob_col not in df.columns:
        print(f"错误: 找不到列 '{prob_col}'")
        print(f"可用列名: {list(df.columns)}")
        return None
    
    print(f"使用用户编号列: {user_col}")
    print(f"使用概率列: {prob_col}")
    
    # 创建结果DataFrame
    result_data = []
    
    for index, row in df.iterrows():
        user_str = str(row[user_col])  # 保持原始格式
        user_id = extract_user_id(user_str)  # 提取数字编号
        probability = float(row[prob_col])
        
        if user_id is None:
            print(f"跳过无效的用户编号: {user_str}")
            continue
        
        # 获取用户类型 (用户编号从1开始，列表索引从0开始)
        if 1 <= user_id <= len(user_type_mapping):
            user_type_code = user_type_mapping[user_id - 1]
            user_type_name = type_dict[user_type_code]
            weight = TYPE_W[user_type_name]
            
            # 计算加权概率
            weighted_prob = probability * weight
            
            result_data.append({
                '用户编号': user_str,  # 保持原始'U{i}'格式
                '数字编号': user_id,   # 添加数字编号用于参考
                '原始总风险概率': probability,
                '用户类型代码': user_type_code,
                '用户类型': user_type_name,
                '权重': weight,
                '加权风险概率': weighted_prob
            })
        else:
            print(f"警告: 用户编号 {user_id} (来自 {user_str}) 超出范围")
    
    result_df = pd.DataFrame(result_data)
    return result_df

# ================= 主函数 =================
def main():
    """
    主函数：执行概率加权计算
    """
    print("开始读取概率数据并进行加权计算...")
    
    # 读取数据
    df = read_probability_data()
    
    if df is not None:
        # 计算加权概率
        result_df = calculate_weighted_probability(df)
        
        if result_df is not None:
            print("\n=== 加权计算结果 ===")
            print(result_df)
            
            # 计算总加权概率
            total_weighted_prob = result_df['加权风险概率'].sum()
            print(f"\n总加权风险概率: {total_weighted_prob:.6f}")
            
            # 按用户类型汇总
            type_summary = result_df.groupby('用户类型').agg({
                '原始总风险概率': 'sum',
                '加权风险概率': 'sum',
                '数字编号': 'count'  # 使用数字编号列进行计数
            }).rename(columns={'数字编号': '用户数量'})
            
            print("\n=== 按用户类型汇总 ===")
            print(type_summary)
            
            # 保存结果到Excel文件
            output_path = str((HERE / "加权概率结果_300kW.xlsx").resolve())
            try:
                with pd.ExcelWriter(output_path) as writer:
                    result_df.to_excel(writer, sheet_name='详细结果', index=False)
                    type_summary.to_excel(writer, sheet_name='类型汇总')
                print(f"\n结果已保存到: {output_path}")
            except Exception as e:
                print(f"保存文件失败: {e}")
        
        else:
            print("加权计算失败")
    else:
        print("数据读取失败")

if __name__ == "__main__":
    main()