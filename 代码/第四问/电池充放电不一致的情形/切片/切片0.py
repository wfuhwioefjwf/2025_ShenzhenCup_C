# -*- coding: utf-8 -*-
"""
precompute_seg_cache.py  —— 带断点续跑与 checkpoint
"""

import os
import sys
import time
import argparse
import hashlib
import importlib.util
from typing import Tuple
import numpy as np
from pathlib import Path
HERE = Path(__file__).parent

# === 路径配置（保持你的原始设置）===
BASE_DIR = str((HERE.parent).resolve())
SEG_CACHE_DIR = os.path.join(BASE_DIR, "缓存")
MONTE_CARLO_FILE = os.path.join(BASE_DIR, "q4_3_各DG储能电池充放电不一致下的决策.py")

# 导入 MonteCarlo 模块（与你的原文件保持一致）
spec = importlib.util.spec_from_file_location("monte_carlo_module", MONTE_CARLO_FILE)
monte_carlo_module = importlib.util.module_from_spec(spec)
sys.modules["monte_carlo_module"] = monte_carlo_module  # 注册到 sys.modules 避免 pickle 问题
spec.loader.exec_module(monte_carlo_module)
make_scenario_loss_evaluator = monte_carlo_module.make_scenario_loss_evaluator

# === 统一分段（绝对出力，右端 1035） ===
ABS_BREAKS = {
    1: [0, 420, 750, 1035],
    2: [0, 120, 180, 1035],
    3: [0, 320, 420, 1035],
    4: [0, 130, 280, 1035],
    5: [0, 300, 490, 1035],
    6: [0, 420, 770, 1035],
    7: [0, 620, 1035],
    8: [0, 350, 550, 1035],  # ✅ DG8 完整，保证三机指纹一致
}

def shard_owner(seg_tuple: Tuple[int, ...], shards: int) -> int:
    """稳定分片：把段键均匀映射到 0..shards-1。"""
    if shards <= 1:
        return 0
    h = hashlib.sha1(str(seg_tuple).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % shards

def all_seg_tuples(helper):
    """
    生成整个 8 维分段空间（t 无关）。
    段计数取自 edges_of(1,k)，仅作为“分段定义”的统一参照。
    """
    from itertools import product
    seg_counts = [len(helper.edges_of(1, k)) - 1 for k in range(1, 9)]
    total = int(np.prod([max(1, n) for n in seg_counts]))
    print("[info] per-DG segment counts:", seg_counts, "→ total =", total)
    ranges = [range(max(1, n)) for n in seg_counts]
    for seg in product(*ranges):
        yield tuple(int(s) for s in seg)

def pair_id_dg7_dg8(seg7: int, seg8: int) -> int:
    """把 (DG7 段, DG8 段) 映射到 0..5：pair_id = seg7*3 + seg8。"""
    return seg7 * 3 + seg8

def _make_bar(total: int, prefix: str = ""):
    """增强进度条：返回 update(done) 回调，提供详细的时间估算信息。"""
    start = time.perf_counter()
    last_pct = -1
    bar_len = 30
    last_update_time = start
    speed_samples = []
    max_samples = 10

    def update(done: int):
        nonlocal last_pct, last_update_time, speed_samples
        done = min(done, total)
        pct = int(done * 100 / max(1, total))
        current_time = time.perf_counter()

        if done > 0 and current_time > last_update_time:
            current_speed = 1.0 / (current_time - last_update_time) if done > 1 else 0
            speed_samples.append(current_speed)
            if len(speed_samples) > max_samples:
                speed_samples.pop(0)

        if pct == last_pct and done != total:
            return
        last_pct = pct
        last_update_time = current_time

        filled = int(bar_len * done / max(1, total))
        bar = "█" * filled + " " * (bar_len - filled)
        elapsed = current_time - start

        if done > 0:
            avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 0
            if avg_speed > 0:
                eta = (total - done) / avg_speed
            else:
                eta = (elapsed / done * (total - done))
            avg_time_per_task = elapsed / done
            finish_time_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta))
            speed_str = f"{avg_speed:.2f}/s" if avg_speed > 0 else "计算中"

            print(f"\r{prefix} |{bar}| {pct:3d}% ({done}/{total}) "
                  f"已用时 {elapsed:6.1f}s | 剩余 {eta:6.1f}s | 速度 {speed_str} | "
                  f"平均 {avg_time_per_task:.2f}s/任务 | 预计完成 {finish_time_str}",
                  end="", flush=True)
        else:
            print(f"\r{prefix} |{bar}| {pct:3d}% ({done}/{total}) 准备中...",
                  end="", flush=True)

        if done >= total:
            final_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"\n[完成] 总用时 {elapsed:.1f}s，平均 {elapsed/total:.2f}s/任务，完成时间 {final_time}")
    return update

def main():
    ap = argparse.ArgumentParser(
        description="预计算 8 个 DG 的分段组合（不含 t）的单时段蒙特卡洛，并写入兼容缓存（含断点恢复）。"
    )
    ap.add_argument("--module-file", type=str, default=MONTE_CARLO_FILE)
    ap.add_argument("--module", type=str, default=None)

    ap.add_argument("--representative", type=str, default="midpoint",
                    choices=["midpoint", "left", "right"])
    ap.add_argument("--base-seed", type=int, default=12345)
    ap.add_argument("--include-mc-seed-in-cache", action="store_true")

    ap.add_argument("--scope", type=str, default="global", choices=["global", "per_cap"])
    ap.add_argument("--y-global-min", type=float, default=0.0)
    ap.add_argument("--y-global-max", type=float, default=1035.0)

    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--max-keys", type=int, default=0)

    ap.add_argument("--dg78-part", type=int, default=0, choices=list(range(6)))
    ap.add_argument("--outdir", type=str, default=SEG_CACHE_DIR)

    # checkpoint 频率（可按需调整）
    # 改成每处理1个键后写一次临时文件
    ap.add_argument("--ckpt-n", type=int, default=27, help="每处理N个键后写一次临时文件")
    ap.add_argument("--ckpt-seconds", type=int, default=9000, help="或每隔S秒写一次临时文件")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n==== 预计算（t 无关，cap 已移除） DG7×DG8 part={args.dg78_part} | rank {args.rank}/{args.shards} ====\n")
    print(f"[开始时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"[配置信息] 分段策略: {args.representative}, 随机种子: {args.base_seed}, 作用域: {args.scope}")
    print(f"[并行配置] DG7×DG8 分片: {args.dg78_part}/6, 总分片: {args.rank+1}/{args.shards}")

    # 1) 占位输入（scope=global 下不会影响键与代表点）
    P_seq = np.zeros((12, 8), dtype=float)
    lo = np.zeros((12, 8), dtype=float)
    up = np.zeros((12, 8), dtype=float)

    # 2) evaluator & helper
    loss_eval, helper = make_scenario_loss_evaluator(
        lo=lo,
        up=up,
        P_seq=P_seq,
        breaks=ABS_BREAKS,
        representative=args.representative,
        base_seed=args.base_seed,
        include_mc_seed_in_cache=args.include_mc_seed_in_cache,
        monte_carlo_func=None,
        scope=args.scope,
        y_global_min=args.y_global_min,
        y_global_max=args.y_global_max,
    )

    # 2.1 有了 helper 才能确定 fingerprint + 文件名
    fp = helper.fingerprint()
    cache_path_final = os.path.join(args.outdir, f"seg_{fp}_dg78p{args.dg78_part}.pkl.gz")
    cache_path_part  = cache_path_final + ".part"
    print(f"[断点恢复] 目标缓存: {os.path.basename(cache_path_final)}")

    # 2.2 载入已有缓存（final 或 .part），并把已完成 keys 记录下来
    done_keys = set()

    def _load_existing_cache_into_helper(path) -> bool:
        """若 helper 支持 load_cache，优先用；否则尽力从文件读 keys 以便跳过。"""
        nonlocal done_keys
        try:
            if not os.path.exists(path):
                return False
            if hasattr(helper, "load_cache"):
                meta = helper.load_cache(path)
                if hasattr(helper, "keys"):
                    try:
                        done_keys = set(helper.keys())
                    except Exception:
                        pass
                print(f"[恢复] 已加载 {os.path.basename(path)}；helper 当前已有 {len(done_keys)} 个键")
                return True
            else:
                # 兜底：直接读 gzip+pickle，尽力拉取 keys 用于跳过
                import gzip, pickle
                with gzip.open(path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                    for k in obj["data"].keys():
                        done_keys.add(tuple(k) if not isinstance(k, tuple) else k)
                    print(f"[恢复] {os.path.basename(path)} 解析出 {len(done_keys)} 个已完成键（仅用于跳过）")
                    return True
                print(f"[恢复] {os.path.basename(path)} 结构未知，跳过载入（仅依赖 helper 内部缓存）")
                return False
        except Exception as e:
            print(f"[恢复] 读取 {os.path.basename(path)} 失败：{e}")
            return False

    loaded = False
    if os.path.exists(cache_path_part):
        loaded = _load_existing_cache_into_helper(cache_path_part)
    if (not loaded) and os.path.exists(cache_path_final):
        loaded = _load_existing_cache_into_helper(cache_path_final)

    # 2.3 已完成判断函数（优先问 helper，其次看 done_keys）
    def already_done(seg_tuple):
        if hasattr(helper, "has_key"):
            try:
                # 兼容两种签名：has_key(1, seg) 或 has_key(seg)
                return helper.has_key(1, seg_tuple) or helper.has_key(seg_tuple)
            except Exception:
                pass
        return seg_tuple in done_keys

    # 3) 统计规模（考虑分片 + rank + 跳过已完成）
    print("\n[统计阶段] 正在分析任务规模...")
    total_combinations = 0
    assigned_all = 0
    assigned_remaining = 0

    for seg in all_seg_tuples(helper):
        total_combinations += 1
        seg7, seg8 = seg[6], seg[7]
        if pair_id_dg7_dg8(seg7, seg8) != args.dg78_part:
            continue
        if shard_owner(seg, args.shards) != args.rank:
            continue
        assigned_all += 1
        if not already_done(seg):
            assigned_remaining += 1

    will_process = assigned_remaining
    print(f"[任务规模] 总组合数: {total_combinations:,}")
    print(f"[任务分配] 本分片总计: {assigned_all:,} 个；已完成可跳过: {assigned_all - assigned_remaining:,} 个；需计算: {assigned_remaining:,} 个")

    if args.max_keys and will_process > args.max_keys:
        print(f"[限制设置] 受 --max-keys 限制，实际将评估: {args.max_keys:,} 个组合")
        will_process = args.max_keys

    estimated_time_per_task = 0.5
    estimated_total_time = will_process * estimated_time_per_task
    print(f"[时间预估] 预计每个任务耗时: {estimated_time_per_task}s")
    print(f"[时间预估] 预计总耗时: {estimated_total_time/60:.1f}分钟 ({estimated_total_time:.0f}秒)\n")

    update = _make_bar(max(1, will_process), prefix="[eval] 进度")

    # 4) 断点续跑计数器 & checkpoint 定时
    last_ckpt_time = time.perf_counter()
    processed_since_ckpt = 0
    CKPT_N = max(1, int(args.ckpt_n))
    CKPT_S = max(1, int(args.ckpt_seconds))

    # 5) 正式评估（不考虑 t），并实时打印进度（支持断点恢复 + 周期性 checkpoint）
    t0 = time.perf_counter()
    done_new = 0

    for seg in all_seg_tuples(helper):
        seg7, seg8 = seg[6], seg[7]
        if pair_id_dg7_dg8(seg7, seg8) != args.dg78_part:
            continue
        if shard_owner(seg, args.shards) != args.rank:
            continue
        if already_done(seg):
            continue

        helper.eval_key(1, seg)
        done_new += 1
        processed_since_ckpt += 1

        if done_new <= will_process:
            update(done_new)

        # checkpoint：条数达到或时间到就写 .part
        now = time.perf_counter()
        need_ckpt = (processed_since_ckpt >= CKPT_N) or ((now - last_ckpt_time) >= CKPT_S)
        if need_ckpt:
            try:
                tmp_path = cache_path_part + ".tmp"
                meta_part = helper.save_cache(tmp_path)  # 先写临时文件
                os.replace(tmp_path, cache_path_part)    # 原子替换到 .part
                print(f"\n[checkpoint] 已临时保存 {meta_part.get('size',0)} 键 → {os.path.basename(cache_path_part)}")
                if hasattr(helper, "keys"):
                    try:
                        done_keys = set(helper.keys())
                    except Exception:
                        pass
            except Exception as e:
                print(f"\n[checkpoint] 写入失败：{e}")
            last_ckpt_time = now
            processed_since_ckpt = 0

        if args.max_keys and done_new >= args.max_keys:
            break

    # 6) 保存最终文件，并清理 .part
    fp = helper.fingerprint()
    cache_path_final = os.path.join(args.outdir, f"seg_{fp}_dg78p{args.dg78_part}.pkl.gz")
    meta = helper.save_cache(cache_path_final)
    dt = time.perf_counter() - t0

    try:
        if os.path.exists(cache_path_part):
            os.remove(cache_path_part)
    except Exception:
        pass

    actual_finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    avg_time_per_task = dt / max(1, done_new)
    tasks_per_second = done_new / dt if dt > 0 else 0
    cache_size_mb = os.path.getsize(cache_path_final) / (1024 * 1024) if os.path.exists(cache_path_final) else 0

    print(f"\n{'='*80}")
    print(f"[任务完成] DG7×DG8 分片 {args.dg78_part} | 分片 {args.rank+1}/{args.shards}")
    print(f"[完成时间] {actual_finish_time}")
    print(f"[处理统计] 本次新增评估: {done_new:,} 个；理应评估(剩余): {will_process:,} 个")
    print(f"[缓存统计] 写入缓存: {meta.get('size',0):,} 个键")
    print(f"[时间统计] 总耗时: {dt:.1f}s ({dt/60:.1f}分钟)")
    print(f"[性能统计] 平均每任务: {avg_time_per_task:.3f}s | 处理速度: {tasks_per_second:.2f} 任务/秒")
    print(f"[文件信息] 缓存文件: {os.path.basename(cache_path_final)}")
    print(f"[文件大小] {cache_size_mb:.2f} MB")
    print(f"[文件路径] {cache_path_final}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()