"""L2 聚类 —— PCA 白化 + HDBSCAN（复用 `dasc.cluster.clusterers`）。

**0 token。** 把全库向量切成「哪些题是一簇」。产物是 :class:`ClusterMetric`、
带跨会三态（venue_matrix），落 `workspace/clusters/` 与 `workspace/metrics/`。

⚠️ 3 条沿用 `clusterers` 的纪律（都来自踩坑，见方案 §3.2）：
1. 噪声 `-1` 永不进 `clusters`；`n_noise` 与 `n_clusters` 并排报。
2. 选参用 `plateau()` **取平台不取冠军**（看簇大小分布，不只看噪声率）。
3. `n_components=64 / min_cluster_size=8` 是 **586 篇测的**；13,119 篇必须重扫（探针 3）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import hotspot
from ..config import load_config, paths_from
from ..schema import ClusterMetric
from .embed import load_all
from .venue_matrix import build_venue_map, build_year_map, venue_profile, ON_AXIS_NOTES

__all__ = ["run_cluster", "to_metrics", "sweep_params", "pick_params", "save_metrics"]


def run_cluster(vectors: np.ndarray, ids: list[str], *,
                min_cluster_size: int, min_samples: int,
                n_components: int) -> Any:
    """封装 `clusterers.run`（白化）。

    ⚠️ `metric="euclidean"`（2026-09-27）：原 dasc 用 `cosine` 会触发 sklearn HDBSCAN
    的 `brute`，需建全量距离矩阵（22,747² float64 = 3.86 GiB）→ 8GB 机器 OOM。PCA 白化后
    各向同性，euclidean 与 cosine 近似，且 euclidean 走 kdtree（省内存、秒级）。
    """
    from dasc.cluster import clusterers
    return clusterers.run(vectors, ids,
                          min_cluster_size=min_cluster_size,
                          min_samples=min_samples,
                          prep="whiten", n_components=n_components,
                          metric="euclidean")


def sweep_params(vectors: np.ndarray, ids: list[str],
                 components: Sequence[int | None] = (32, 64, 128),
                 sizes: Sequence[int] = (4, 6, 8, 12, 16, 24, 32)):
    """探针 3：13,119 篇上的两维扫参。返回 `SweepRow` 列表。"""
    from dasc.cluster import clusterers
    return clusterers.sweep_grid(vectors, ids, sizes=sizes,
                                 components=components, prep="whiten")


def pick_params(rows) -> dict:
    """从扫参结果里选参数：`plateau()` 取平台，**不取冠军**。"""
    from dasc.cluster import clusterers
    plat = clusterers.plateau(rows)
    if not plat:
        return {"on_plateau": False, "plateau": [], "note":
                "平台为空——这组网格上没有同时不碎片化、不吞成一团、分得出 ≥5 簇的参。"}
    # 取平台上噪声率最低的一组作代表（但要先确认它没在平台外被更差的顶掉）。
    best = min(plat, key=lambda r: r.n_noise)
    return {
        "on_plateau": True,
        "plateau": [{"n_components": r.n_components,
                     "min_cluster_size": r.min_cluster_size,
                     "n_clusters": r.n_clusters, "n_noise": r.n_noise}
                    for r in plat],
        "chosen": {"n_components": best.n_components,
                   "min_cluster_size": best.min_cluster_size},
    }


def _parse_ints(s: str | None) -> list[int] | None:
    """解析逗号分隔的整数参数，如 `--sizes 4,8,16`。"""
    if not s:
        return None
    out = [int(x) for x in s.split(",") if x.strip()]
    return out or None


def sweep_resumable(X, ids, *, out_json,
                    components=(32, 64, 128),
                    sizes=(4, 6, 8, 12, 16, 24, 32),
                    min_samples: int = 4, prep: str = "whiten") -> dict:
    """逐组串行扫参，**每组结果即时落盘**，断了可续跑（跳过已完成组）。

    相比 `dasc.sweep_grid` 一次全跑，这里逐个组合、单组失败（含 OOM）也
    记录进 `sweep.json` 并继续——内存不足时能明确看到是哪组挂、且不白跑。
    平台判定请基于落盘 `results`（`n_clusters`/`n_noise`）人工判，或后续写轻量选参。
    """
    from dasc.cluster import clusterers  # noqa: PLC0415
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    if out_json.exists():
        try:
            results = json.loads(out_json.read_text(encoding="utf-8")).get("results", {})
        except Exception:  # noqa: BLE001 —— 已损坏则重新开始
            results = {}

    for nc in components:
        for mcs in sizes:
            key = f"{nc}_{mcs}"
            if key in results:
                print(f"[sweep] {key} 已存在，跳过", flush=True)
                continue
            try:
                cl = clusterers.run(X, ids, min_cluster_size=mcs, min_samples=min_samples,
                                    prep=prep, n_components=nc)
                results[key] = {"n_components": nc, "min_cluster_size": mcs,
                                "n_clusters": cl.n_clusters, "n_noise": cl.n_noise}
            except Exception as e:  # noqa: BLE001 —— OOM/报错也落盘，不阻塞
                results[key] = {"n_components": nc, "min_cluster_size": mcs,
                                "error": str(e)}
            out_json.write_text(json.dumps(
                {"prep": prep, "components": list(components),
                 "sizes": list(sizes), "results": results},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[sweep] {key} 完成 -> 已落盘", flush=True)
    return results


def to_metrics(clustering, *, venue_of: dict[str, str],
               venue_order: tuple[str, ...], spread_min: int,
               off_axis: tuple[str, ...] = (),
               year_of: dict[str, int] | None = None) -> list[ClusterMetric]:
    """把聚类结果变成带三态的 :class:`ClusterMetric` 列表。

    只算能**客观**拿到的：规模、跨会三态、代表性论文（=质心最近，即
    `Cluster.centroid_nearest`——是"最典型"不是"最重要"）。
    `year_of`（`paper_id -> year`）用于生成 `VenueProfile.counts_by_year`
    （保留分年能力；三态仍由跨年混合的总计数判）。
    `top_terms` / `neighbor_ids` 属 D6 簇邻接与判别词，这里留空、由后续补。
    """
    metrics: list[ClusterMetric] = []
    for c in clustering.clusters:
        # 空 venue（`""`）不进计数：那是"没记到会"，不是"某会 0 篇"。
        counts: dict[str, int] = {}
        counts_by_year: dict[int, dict[str, int]] = {}
        for pid in c.members:
            v = venue_of.get(pid, "")
            if not v:
                continue
            counts[v] = counts.get(v, 0) + 1
            y = year_of.get(pid) if year_of else None
            if y is not None:
                yv = counts_by_year.setdefault(y, {})
                yv[v] = yv.get(v, 0) + 1
        vp = venue_profile(counts, counts_by_year=counts_by_year,
                           venue_order=venue_order,
                           spread_min=spread_min, off_axis=off_axis)
        metrics.append(ClusterMetric(
            cluster_id=c.label,
            size=c.size,
            venue=vp,
            representative_ids=(c.centroid_nearest,) if c.centroid_nearest else (),
        ))
    metrics.sort(key=lambda m: (-m.size, m.cluster_id))
    return metrics


def _to_metric_dicts(metrics: list[ClusterMetric], *, produced_by: str) -> list[dict]:
    out = []
    for m in metrics:
        d = m.to_dict()
        d["_produced_by"] = produced_by
        out.append(d)
    return out


def save_metrics(metrics: list[ClusterMetric], *,
                 cluster_members: dict[int, list[str]] | None = None,
                 out_clusters: Path | None = None,
                 out_metrics: Path | None = None,
                 out_members: Path | None = None,
                 extra: dict | None = None) -> tuple[Path, Path, Path]:
    """落盘：原始聚类摘要（`clusters/`）+ 逐簇 metric（`metrics/`）+ 簇成员（L3 取摘要用）。

    `cluster_members` 是 `cluster_id -> [paper_id]`，L3 生成 ClusterCard 时要
    按它现取每簇成员的 title+abstract（方案 §3.3），所以必须一起落盘。
    """
    ws = hotspot.WORKSPACE
    oc = out_clusters or (ws / "clusters" / "clusters.json")
    om = out_metrics or (ws / "metrics" / "metrics.json")
    omm = out_members or (ws / "metrics" / "cluster_members.json")
    oc.parent.mkdir(parents=True, exist_ok=True)
    om.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": ON_AXIS_NOTES,
        **(extra or {}),
        "metrics": _to_metric_dicts(metrics, produced_by="hotspot/structure/cluster.py"),
    }
    om.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if cluster_members is not None:
        omm.write_text(json.dumps(
            {str(k): v for k, v in sorted(cluster_members.items(), key=lambda kv: kv[0])},
            ensure_ascii=False, indent=2), encoding="utf-8")
    oc.write_text(json.dumps({
        "n": len(metrics),
        "sizes": [m.size for m in metrics],
        **(extra or {}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return oc, om, omm


def main() -> int:
    ap = argparse.ArgumentParser(description="全库向量 → 聚类 + 跨会三态 metric")
    ap.add_argument("--sweep", action="store_true", help="只扫参（探针 3），逐组落盘续跑")
    ap.add_argument("--components", type=str, default=None,
                    help="扫参的 n_components 候选，逗号分隔（默认 32,64,128；内存紧可缩）")
    ap.add_argument("--sizes", type=str, default=None,
                    help="扫参的 min_cluster_size 候选，逗号分隔（默认 4,6,8,12,16,24,32）")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--n-components", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config()
    paths = paths_from(cfg)
    cluster_cfg = cfg.get("cluster") or {}
    spread_cfg = cfg.get("spread") or {}

    X, ids = load_all(paths["workspace"] / "vectors")
    print(f"读入向量 {X.shape}（{len(ids)} 篇）")

    if a.sweep:
        comps = _parse_ints(a.components) or [32, 64, 128]
        sizes = _parse_ints(a.sizes) or [4, 6, 8, 12, 16, 24, 32]
        out = paths["workspace"] / "metrics" / "sweep.json"
        res = sweep_resumable(X, ids, out_json=out, components=comps, sizes=sizes)
        print(json.dumps({"components": comps, "sizes": sizes, "results": res},
                         ensure_ascii=False, indent=2))
        print("落盘:", out)
        return 0

    mcs = a.min_cluster_size or int(cluster_cfg.get("min_cluster_size") or 8)
    nc = a.n_components or int(cluster_cfg.get("n_components") or 64)
    clust = run_cluster(X, ids, min_cluster_size=mcs, min_samples=4, n_components=nc)
    print(clust.summary())

    from ..corpus import open_corpus  # noqa: PLC0415
    c = open_corpus(cfg)
    papers_all = c.list_papers(limit=None)
    venue_of = build_venue_map(papers_all)
    year_of = build_year_map(papers_all)

    metrics = to_metrics(
        clust, venue_of=venue_of, year_of=year_of,
        venue_order=tuple(spread_cfg.get("venue_order") or ()),
        spread_min=int(spread_cfg.get("spread_min") or 3),
        off_axis=tuple(spread_cfg.get("venue_off_axis") or ()),
    )
    oc, om, omm = save_metrics(
        metrics,
        cluster_members={c.label: list(c.members) for c in clust.clusters},
        extra={
            "n_papers": clust.n, "n_clusters": clust.n_clusters, "n_noise": clust.n_noise,
            "params": clust.params,
        },
    )
    print()
    for m in metrics[:15]:
        print(f"  #{m.cluster_id:<3d} {m.size:>4d} 篇  {m.venue.state.value}  "
              f"{m.venue.counts}")
    print(f"落盘：{oc}\n      {om}\n      {omm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
