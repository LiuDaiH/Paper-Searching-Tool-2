"""跨会留出验证（方案 §5.4 · 测试计划 §3）—— 数据切分 + baseline + 指标骨架。

Train = ICLR2025 + ICML2025，Holdout = NeurIPS2025（封存）。语料 venue 分布实测：
NeurIPS25=5823 / ICLR25=3703 / ICML25=3330 / CoRL off-axis 不用。

本脚本：流式读语料（不持全文、轻内存、0 token）——
- 切 train/holdout
- 算真实「主题热度」top-10（holdout 高频词，代理口径）
- 对照：随机 top-10 / 纯高频 top-10（train 高频词）
- 指标：Precision@10 + Spearman（hand-rolled）
「主题预测」的完整实现（subcluster 聚类产出主题 top-10）需向量+内存+LLM，
接口为 `predict_topic_top(mode='subcluster')` 占位 —— 待评测资源充足时接入。

用法：`python eval/holdout.py [--seed 42]`
产物：`workspace/ledger/holdout_baseline.json`
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotspot.config import load_config, paths_from  # noqa: E402

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_STOP = frozenset("""
the a an and or but if then of to in on for with by from as at this that these those
is are was were be been being have has had do does did will would can could should
may might must its it s t our their his her not no so such than into over under
about between which who whom when where why how what all any both each few more
most other some only own same too very just also new using use used via based
model method approach paper work study recent propose present learn training
learning reinforcement policy agent result experiment performance our proposed
""".split())
#: 会议时间轴：train 仓 = venue_order 去掉 holdout；holdout = venue_holdout。
TRAIN_VENUES = ("ICLR", "ICML")
HOLDOUT_VENUE = "NeurIPS"
TARGET_YEAR = 2025


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower())
            if len(w) > 1 and w not in _STOP]


def load_counts() -> tuple[Counter, Counter]:
    """流式读语料，按 venue/year 切 train/holdout，只累加词频（不持全文）。"""
    cfg = load_config()
    train: Counter = Counter()
    hold: Counter = Counter()
    srcs = (cfg.get("corpus") or {}).get("sources") or []
    seen = 0
    for s in srcs:
        p = Path(s)
        if not p.exists():
            print("  ⚠️ 缺语料源:", s)
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("year") != TARGET_YEAR:
                    continue
                v = r.get("venue") or ""
                tokens = tokenize((r.get("title") or "") + " " + (r.get("abstract") or ""))
                if v == HOLDOUT_VENUE:
                    hold.update(tokens)
                elif v in TRAIN_VENUES:
                    train.update(tokens)
                seen += 1
    print(f"  含 2025 论文(流式): {seen} 篇")
    return train, hold


def predict_topic_top(train: Counter, hold: Counter, *, n: int = 10,
                      mode: str = "freq", seed: int = 42) -> list[str]:
    """生成预测 top-10 主题（词代理口径）。

    - 'freq'：train 高频词 top-n（纯高频 baseline）
    - 'random'：hold 词表随机 top-n
    - 'subcluster'：⚠️ 占位 —— 未来用 train 聚类主题簇产出（需向量+内存）
    """
    if mode == "random":
        pool = list(hold.keys())
        rng = random.Random(seed)
        return rng.sample(pool, min(n, len(pool)))
    if mode == "subcluster":
        raise NotImplementedError("subcluster 主题预测未接入——需评测资源(向量/内存/LLM)")
    return [w for w, _ in train.most_common(n)]


def true_top(hold: Counter, n: int = 10) -> list[str]:
    return [w for w, _ in hold.most_common(n)]


def prec_at_10(pred: list[str], true: list[str]) -> float:
    if not true:
        return 0.0
    return len(set(pred) & set(true)) / len(true)


def spearman(pred: list[str], true: list[str]) -> float:
    """pred 的 10 词按其在 true 中的秩 vs 位置 1..10 的 pearson。不在 true 记 11+。"""
    rank = {t: i + 1 for i, t in enumerate(true)}
    xs = list(range(1, len(pred) + 1))
    ys = [rank.get(p, len(true) + 5) for p in pred]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config()
    ws = Path(paths_from(cfg)["workspace"])

    train, hold = load_counts()
    true = true_top(hold, a.n)
    print(f"真实 NeurIPS2025 top-{a.n}（高频词）: {true}")

    rows = []
    for mode in ("freq", "random"):
        pred = predict_topic_top(train, hold, n=a.n, mode=mode, seed=a.seed)
        rows.append({"mode": mode, "top": pred,
                     "precision@10": round(prec_at_10(pred, true), 4),
                     "spearman": round(spearman(pred, true), 4)})
        print(f"[{mode}] P@10={prec_at_10(pred, true):.3f} "
              f"Spearman={spearman(pred, true):.3f} top={pred[:5]}")
    out = ws / "ledger" / "holdout_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"train": "ICLR+ICML 2025", "holdout": f"{HOLDOUT_VENUE} {TARGET_YEAR}",
         "true_top": true, "n": a.n, "note": "词频代理口径；subcluster 主题预测待接入",
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"落盘：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
