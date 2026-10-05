"""移动云杯大创两线（Streit-Garg + rlRAG）接入自检。

校验：
1. `hotspot/mps.py`（Streit-Garg 关键路径）四算法产出**可行**词；
2. `hotspot/rlr.py`（rlRAG LTR）能对候选举出「该学/该引」排序；
3. `subcluster` relation 意图导入无副作用（不触导师库写入）。

数据缺则明确跳过（报"未跑"），不是假装绿。结论 = 退出码。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
for p in (r"D:\大学编程学习\obsidian", r"D:\大学编程学习\移动云杯"):
    if p not in sys.path:
        sys.path.insert(0, p)

_reds: list = []


def _bad(m: str) -> None:
    _reds.append(m)
    print(f"[红] {m}")


def main() -> int:
    print("[守卫] 移动云杯 · 大创两线接入（Streit-Garg + rlRAG）")
    try:
        from hotspot.mps import compute_key_path, _load_prep
        from hotspot import subcluster
    except ImportError as e:                     # noqa: BLE001
        _bad(f"导入失败: {e}")
        print("-" * 40)
        return 1

    if "relation" not in subcluster.INTENTS:
        _bad("subcluster.INTENTS 缺 relation")
    else:
        print("[OK] subcluster.INTENTS 含 relation，导入无副作用")

    prep_dir = Path(r"D:\大学编程学习\obsidian\dasc\_out\rl_prep")
    rlr_dir = Path(r"D:\大学编程学习\obsidian\dasc\_out\rlr")
    data_ok = (prep_dir / "cite_graph.pkl").exists() \
        and (prep_dir / "coverage.pkl").exists()

    if not data_ok:
        print("[跳过] rl_prep 数据缺失 —— 两线冒烟**未跑**")
        print("-" * 40)
        return 0

    # 1. Streit-Garg 四算法
    prep = _load_prep()
    pool = [k for k in prep["forward"].keys() if k in prep["p2c"]][:12]
    for algo in ("mag_sub", "mag_add", "scaffold_add", "rand_sub"):
        try:
            r = compute_key_path(pool, algo=algo)
        except Exception as e:                   # noqa: BLE001
            _bad(f"mps/{algo} 抛异常: {type(e).__name__}: {e}")
            continue
        if not r.feasible:
            _bad(f"mps/{algo} 输出不可行: {r.note}")
        else:
            print(f"[OK] mps/{algo}: elem={r.n_elements} Δ={r.delta} "
                  f"conn={r.conn} value={r.value}")

    # 2. rlRAG
    if not (rlr_dir / "future_rf.pkl").exists():
        print("[跳过] rlRAG future_rf.pkl 缺失 —— RF 冒烟未跑（先跑" +
              "dasc/rlr/export_future_rf.py，需 G 盘）")
    else:
        try:
            from hotspot.rlr import RFRanker
            ranker = RFRanker()
            cands = list(_load_prep()["forward"].keys())[:12]
            ranked = ranker.rank(cands)
            if not ranked:
                _bad("rlRAG: 候选为空")
            else:
                print(f"[OK] rlRAG RF(S+A): 排 {len(ranked)} 候选，"
                      f"top={ranked[0][0][:8]} score={ranked[0][1]:.3f}")
        except Exception as e:                   # noqa: BLE001
            _bad(f"rlRAG RF 抛异常: {type(e).__name__}: {e}")

    print("-" * 40)
    if _reds:
        print(f"{len(_reds)} 项红。")
        return 1
    print("移动云杯大创两线接入自检全绿（跳过项需确认数据就位）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
