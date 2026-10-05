"""P3 发布物生成 —— 把 arXiv 周报打成「已向量化」发布物（别人零模型下载即可检索）。

读 `config.corpus.arxiv_source`（arXivCorpus），复用 `structure.embed.embed_corpus`
分块向量化（落 `workspace/arxiv_vectors/vec_*.npz`，分片续跑 = 每周增量），再打包成
`weekly_arxiv/`：
  vectors.npy     稠密向量矩阵（float32，行序 = ids 序）
  ids.jsonl       与向量行对齐的 paper_id 序
  meta.jsonl      标题/摘要/arxiv_id/topics/first_seen
  topics.json     描述性热点统计（三态轴待拍板，见规划 §7；不输出退潮/下降/衰退）
  manifest.json   embedding 模型版本 / 覆盖周 / 篇数 / dim / 各 sha256 / 构建时间

默认产出 `workspace/exports/weekly_arxiv/`；`--publish-dir` 拷贝到穿透机/同步目录
（对象存储/云盘由外部同步，本脚本只保证发布物结构与内容）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import time
from collections import Counter

import numpy as np

from hotspot.config import load_config, paths_from
from hotspot.corpus import open_arxiv_corpus
from hotspot.structure.embed import embed_corpus, load_all

__all__ = ["build_weekly", "export_arxiv_weekly"]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _arxiv_vec_dir(cfg) -> Path:
    return paths_from(cfg)["workspace"] / "arxiv_vectors"


def _default_out(cfg) -> Path:
    return paths_from(cfg)["workspace"] / "exports" / "weekly_arxiv"


def _week_key(date_str: str) -> str | None:
    if not date_str:
        return None
    try:
        y, w, _ = _dt.date.fromisoformat(date_str).isocalendar()
        return f"{y}-W{w:02d}"
    except ValueError:
        return None


def _topics(papers) -> dict:
    today = _dt.date.today()
    week_str = _week_key(today.isoformat()) or ""
    new_week = [p for p in papers if _week_key(p.first_seen) == week_str]
    by_year = Counter(p.year for p in papers if p.year)
    by_topics = Counter(t for p in papers for t in p.topics if t)
    by_seen = Counter(_week_key(p.first_seen) for p in papers if p.first_seen)
    new_top = Counter(t for p in new_week for t in p.topics if t).most_common(10)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cover_week": week_str,
        "n": len(papers),
        "new_this_week": len(new_week),
        "by_year": {str(k): v for k, v in sorted(by_year.items())},
        "by_topics": dict(by_topics),
        "by_first_seen_week": {str(k or ""): v for k, v in
                               sorted(by_seen.items(), key=lambda kv: kv[0] or "")},
        "week_new_top_topics": [{"topic": t, "n": n} for t, n in new_top],
        "note": "热点三态轴（分类/周）口径待拍板（规划 §7）；此处仅给描述性统计，"
                "不输出退潮/下降/衰退（红线）。",
    }


def export_arxiv_weekly(cfg, *, limit=None, model=None,
                        out=None, publish_dir=None) -> Path:
    """生成 `weekly_arxiv/` 发布物。返回 out_dir。"""
    c = open_arxiv_corpus(cfg)
    em = cfg.get("embed") or {}
    model = model or em.get("model") or "BAAI/bge-m3"
    vec_dir = _arxiv_vec_dir(cfg)
    with c.lib():
        rep = embed_corpus(
            c, out_dir=vec_dir,
            chunk_size=int(em.get("chunk_size") or 500),
            limit=limit, model=model,
        )
    V, ids = load_all(vec_dir)
    n_vec = len(ids)

    papers: list = []
    for pid in ids:
        p = c.get(pid)
        if p is not None:
            papers.append(p)
    if len(papers) != n_vec:
        print(f"⚠️ 向量 {n_vec} 篇但语料命中 {len(papers)} —— 发布物将只含命中部分")

    out_dir = Path(out) if out else _default_out(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # meta.jsonl（顺序 = ids 序）
    meta_lines = []
    for pid in ids:
        p = c.get(pid)
        if p is None:
            continue
        meta_lines.append(json.dumps({
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": (p.abstract or "")[:2000],
            "arxiv_id": p.arxiv_id,
            "topics": list(p.topics),
            "first_seen": p.first_seen,
            "year": p.year,
            "is_rl": p.is_rl,
        }, ensure_ascii=False))
    meta_text = "\n".join(meta_lines) + ("\n" if meta_lines else "")
    (out_dir / "meta.jsonl").write_text(meta_text, encoding="utf-8")

    # vectors.npy + ids.jsonl
    np.save(out_dir / "vectors.npy", V)
    (out_dir / "ids.jsonl").write_text(
        "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    # topics.json
    (out_dir / "topics.json").write_text(
        json.dumps(_topics(papers), ensure_ascii=False, indent=2), encoding="utf-8")

    # manifest.json
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embed_model": model,
        "embed_dim": int(V.shape[1]) if V.size else 0,
        "n": n_vec,
        "cover_week": _week_key(_dt.date.today().isoformat()),
        "corpus_source": str(((cfg.get("corpus") or {}).get("arxiv_source") or "")),
        "vectors_sha256": hashlib.sha256(V.tobytes()).hexdigest(),
        "ids_sha256": _sha256_text("".join(ids)),
        "meta_sha256": _sha256_text(meta_text),
        "embed_report": rep,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if publish_dir:
        pd = Path(publish_dir)
        pd.mkdir(parents=True, exist_ok=True)
        for name in ("vectors.npy", "ids.jsonl", "meta.jsonl", "topics.json", "manifest.json"):
            shutil.copy2(out_dir / name, pd / name)
        print(f"已拷贝发布物 → {pd}")

    print(f"发布物已生成 → {out_dir}")
    print(f"  向量 {n_vec} 篇 × dim {manifest['embed_dim']}（模型 {model}）"
          f"｜新算 {rep['done']}，跳过 {rep['skipped']}")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 arXiv 周报「已向量化」发布物 weekly_arxiv/")
    ap.add_argument("--limit", type=int, default=None, help="只向量化前 n 篇（试跑/验证）")
    ap.add_argument("--model", default=None, help="embedding 模型（默认取 config.embed.model）")
    ap.add_argument("--out", default=None, help="发布物输出目录（默认 exports/weekly_arxiv）")
    ap.add_argument("--publish-dir", default=None, help="额外拷贝到的穿透/同步目录")
    a = ap.parse_args()
    export_arxiv_weekly(load_config(), limit=a.limit, model=a.model,
                        out=a.out, publish_dir=a.publish_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
