# -*- coding: utf-8 -*-
"""
merge_seg_cache_parts.py  (DG7×DG8 六分片版)
把分布式预计算得到的 6 份段缓存：
  seg_<fingerprint>_dg78p0.pkl.gz
  ...
  seg_<fingerprint>_dg78p5.pkl.gz
按并集（union）合并成：
  seg_<fingerprint>.pkl.gz

增强点：
- 自动发现分片文件：优先新命名 dg78p*，若无则回退兼容旧命名 dg8p*；
- 可选严格检查：--require-all 要求 6 片都在，否则报错退出；
- 其余参数、ABS_BREAKS、指纹一致性检查（strict=True）保持不变。
"""

import os
import sys
import argparse
import importlib.util
import time
import glob
import numpy as np

# === 默认路径：与你的工程保持一致（按需修改） ===
BASE_DIR = r"C:\Users\xueyixian\Desktop\深圳杯\第四问附加"
DEFAULT_SEG_CACHE_DIR = os.path.join(BASE_DIR, "segcache")
DEFAULT_MODULE_FILE = os.path.join(BASE_DIR, "遍历_粒子群算法_8.19.py")

# === 与预计算一致的分段（ABS_BREAKS） ===
ABS_BREAKS = {
    1: [0, 420, 750, 1035],
    2: [0, 120, 180, 1035],
    3: [0, 320, 420, 1035],
    4: [0, 130, 280, 1035],
    5: [0, 300, 490, 1035],
    6: [0, 420, 770, 1035],
    7: [0, 620, 1035],        # DG7 → 两段
    8: [0, 350, 550, 1035],   # DG8 → 三段
}

def load_mc_module(module_file: str, module_name: str = "monte_carlo_module"):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块文件：{module_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def find_part_files(segcache_dir: str, fp: str, parts=None):
    """
    自动发现分片文件。
    优先新命名：seg_<fp>_dg78p*.pkl.gz；若找不到，回退到旧命名 seg_<fp>_dg8p*.pkl.gz
    可用 parts 指定想要的分片编号（列表或集合），否则抓取目录下所有匹配的。
    返回（files, scheme, expected_count）
      scheme: "dg78p" 或 "dg8p"
      expected_count: 6（新）或 3（旧）
    """
    files_new = sorted(glob.glob(os.path.join(segcache_dir, f"seg_{fp}_dg78p*.pkl.gz")))
    if files_new:
        if parts is not None:
            files_new = [os.path.join(segcache_dir, f"seg_{fp}_dg78p{n}.pkl.gz") for n in sorted(parts)]
            files_new = [p for p in files_new if os.path.exists(p)]
        return files_new, "dg78p", 6

    files_old = sorted(glob.glob(os.path.join(segcache_dir, f"seg_{fp}_dg8p*.pkl.gz")))
    if files_old:
        if parts is not None:
            files_old = [os.path.join(segcache_dir, f"seg_{fp}_dg8p{n}.pkl.gz") for n in sorted(parts)]
            files_old = [p for p in files_old if os.path.exists(p)]
        return files_old, "dg8p", 3

    return [], "none", 0

def main():
    ap = argparse.ArgumentParser(description="合并段缓存分片（支持 DG7×DG8 六分片 dg78p*，兼容旧 dg8p*）。")
    ap.add_argument("--segcache", type=str, default=DEFAULT_SEG_CACHE_DIR,
                    help="段缓存目录（含分片与合并输出）")
    ap.add_argument("--module-file", type=str, default=DEFAULT_MODULE_FILE,
                    help="你的 MonteCarlo/PSO 侧模块的绝对路径（用于构造 helper 计算指纹）")
    # 下列参数需与预计算/PSO 保持一致
    ap.add_argument("--representative", type=str, default="midpoint",
                    choices=["midpoint", "left", "right"], help="段代表点策略")
    ap.add_argument("--base-seed", type=int, default=12345,
                    help="段缓存评估的基础随机种子（用于生成指纹）")
    ap.add_argument("--include-mc-seed-in-cache", action="store_true",
                    help="把 mc_seed 并入缓存键（通常不启用；务必与PSO一致）")
    ap.add_argument("--scope", type=str, default="global", choices=["global", "per_cap"],
                    help="global 键不含 t/cap；per_cap 键与当期可行域相关")
    ap.add_argument("--y-global-min", type=float, default=0.0,
                    help="scope=global 时有效")
    ap.add_argument("--y-global-max", type=float, default=1035.0,
                    help="scope=global 时有效")
    # 合并控制
    ap.add_argument("--require-all", action="store_true",
                    help="若启用：新命名要求 6 片、旧命名要求 3 片全在，否则报错退出")
    ap.add_argument("--parts", type=str, default=None,
                    help="仅合并指定分片编号，如 '0,2,5'；默认 None 表示发现目录中所有匹配分片")
    args = ap.parse_args()

    parts = None
    if args.parts:
        parts = {int(x) for x in args.parts.split(",") if x.strip() != ""}

    os.makedirs(args.segcache, exist_ok=True)

    # 1) 导入原模块，取 evaluator/helper 工厂
    mc = load_mc_module(args.module_file)
    make_scenario_loss_evaluator = mc.make_scenario_loss_evaluator

    # 2) 构造“空” evaluator & helper，仅用于计算指纹/承载缓存
    P_seq = np.zeros((12, 8), dtype=float)
    lo = np.zeros((12, 8), dtype=float)
    up = np.zeros((12, 8), dtype=float)

    loss_eval, helper = make_scenario_loss_evaluator(
        lo=lo,
        up=up,
        P_seq=P_seq,
        breaks=ABS_BREAKS,
        representative=args.representative,
        base_seed=args.base_seed,
        include_mc_seed_in_cache=args.include_mc_seed_in_cache,
        monte_carlo_func=None,          # 不需要跑仿真，只承载缓存
        scope=args.scope,
        y_global_min=args.y_global_min,
        y_global_max=args.y_global_max,
    )

    fp = helper.fingerprint()
    out_file = os.path.join(args.segcache, f"seg_{fp}.pkl.gz")

    # 3) 自动发现分片
    files, scheme, expected = find_part_files(args.segcache, fp, parts=parts)
    print(f"[merge] 目标指纹：{fp}")
    if scheme == "none":
        print("[error] 未找到任何匹配分片：既无 dg78p* 也无 dg8p*。请检查路径/指纹/文件是否存在。")
        sys.exit(2)

    print(f"[merge] 采用命名方案：{scheme} ；期望份数={expected}；实际发现={len(files)}")
    for p in (files if files else []):
        print("         ", p)

    if args.require_all and len(files) != expected:
        print(f"[error] --require-all 已开启，但仅发现 {len(files)}/{expected} 个分片。")
        sys.exit(3)

    # 4) 逐个加载分片（strict=True，fingerprint 不匹配会被拒绝）
    t0 = time.perf_counter()
    loaded_any = False
    total_loaded = 0
    for p in files:
        if not os.path.exists(p):
            print(f"[warn] 分片缺失：{p}")
            continue
        meta = helper.load_cache(p, strict=True, merge="union")
        if not meta:
            print(f"[warn] 跳过（指纹不匹配或文件损坏）：{p}")
            continue
        sz = meta.get('size', '?')
        print(f"[ok]   载入：{p} | 分片内记录数(meta.size)={sz}")
        total_loaded += (sz if isinstance(sz, int) else 0)
        loaded_any = True

    if not loaded_any:
        print("[error] 未载入任何分片，请检查路径/指纹/文件是否存在。")
        sys.exit(4)

    # 5) 保存合并后的完整缓存
    meta_out = helper.save_cache(out_file)
    dt = time.perf_counter() - t0
    print(f"[done] 已合并保存：{out_file} | 合并后 size={meta_out.get('size', 0)} | "
          f"合并输入总记录≈{total_loaded} | 用时 {dt:.1f}s")

if __name__ == "__main__":
    main()
