"""R1 周报引用图拉取 —— Semantic Scholar 单条 GET，增量断点续跑（2026-10-02-规划-周报接入大创算法.md）。

逐篇 GET `api.semanticscholar.org/graph/v1/paper/ARXIV:<id>?fields=references,externalIds`，
拉每篇的 `references[].arxivId`，落 `workspace/depgraph/`：
  paper_refs_arxiv.json  {arxiv_id: [ref_arxiv_id, ...]}（原始，含语料外引用）
  cite_graph_arxiv.pkl   {"forward": {arxiv_id: [ref,...]}, "reverse": {...}}（仅语料内边）
  fetch_state.json       已成功拉取的 arxiv_id 集合（断点续跑）

⚠️ 数据源实测结论（2026-10-02）：
  - S2 **batch** 端点不识别 `ARXIV:` 前缀（返回非 arXiv record、refs=0）⇒ 用**单条 GET**。
  - 无 key 时 S2 429 极频繁 ⇒ 本脚本**超慢节流**（`--delay`，默认 1.6s/篇）+ 429 指数退避；
    全量 4,369 篇无 key 需数小时。**强烈建议配置 S2 API key**（`config.llm_eval` 之外，
    见 `apis.api_key_env`）把 429 降到可忽略。
  - OpenAlex 对 arXiv 支持差（`filter=arxiv:` 400、不收录 arXiv DOI），不用它做引用边。
  `--limit` 试跑；断点续跑 = 每次成功落盘，杀进程只丢最后几篇。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import urllib.request

from hotspot.config import load_config, paths_from
from hotspot.corpus import open_arxiv_corpus

S2_GET = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{}"
# references 子对象默认不含 externalIds —— 必须显式 `references.externalIds` 才能拿 arXiv id
S2_FIELDS = "?fields=references.externalIds,references.title,title"

__all__ = ["fetch_arxiv_refs", "main"]


def _depgraph_dir(cfg) -> Path:
    return paths_from(cfg)["workspace"] / "depgraph"


def _api_key(cfg) -> str:
    env = (cfg.get("apis") or {}).get("api_key_env") or ""
    return os.environ.get(env, "") if env else ""


def _get_one(arxiv_id: str, *, api_key: str = "", timeout: int = 60,
             retries: int = 5) -> list[str] | None:
    """拉单篇引用。失败（含持续 429）返回 None（调用方跳过、断点续跑补）"""
    url = S2_GET.format(arxiv_id) + S2_FIELDS
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    delay = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            refs = []
            for x in (d.get("references") or []):
                ra = (x.get("externalIds") or {}).get("ArXiv") \
                    or x.get("arxivId") or ""
                if ra:
                    refs.append(ra)
            return sorted(set(refs))
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):
                return []   # 未收录 / 无权限，视为成功（空引用）
            if e.code == 429:
                time.sleep(delay * (1 + random.random()))
                delay = min(delay * 2, 90.0)
                continue
            raise
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 90.0)
            continue
    return None   # 多次失败 → 跳过，留给下一轮断点续跑补


def fetch_arxiv_refs(cfg, *, limit: int | None = None,
                     delay: float = 1.6, only=None) -> dict:
    """拉周报引用边，增量合并，返回 `{arxiv_id: [refs]}`。

    `only`：限定只处理这些 arxiv_id（供每周增量脚本只拉新增论文）；
    留空则处理全部未拉取的（断点续跑补齐历史）。
    """
    dg = _depgraph_dir(cfg)
    dg.mkdir(parents=True, exist_ok=True)
    state_path = dg / "fetch_state.json"
    raw_path = dg / "paper_refs_arxiv.json"
    graph_path = dg / "cite_graph_arxiv.pkl"
    api_key = _api_key(cfg)

    done: set = set()
    if state_path.exists():
        done = set(json.loads(state_path.read_text(encoding="utf-8")))
    raw: dict = {}
    if raw_path.exists():
        raw = {k: list(v) for k, v in
               json.loads(raw_path.read_text(encoding="utf-8")).items()}

    c = open_arxiv_corpus(cfg)
    arxiv_ids = sorted({p.arxiv_id for p in c.list_papers() if p.arxiv_id})
    todo = [a for a in arxiv_ids
            if (only is None or a in only) and a not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"语料 {len(arxiv_ids)} 篇；待拉 {len(todo)} 篇"
          f"（已拉 {len(done)}；{'带 key' if api_key else '无 key(慢)'}）")

    fetched = 0
    failed = 0
    try:
        for a in todo:
            refs = _get_one(a, api_key=api_key)
            if refs is None:
                failed += 1
                print(f"  跳过(429超时) {a} —— 本轮未完成，下轮续跑补")
                continue
            raw[a] = refs
            done.add(a)
            fetched += 1
            if fetched % 20 == 0 or fetched >= len(todo):
                raw_path.write_text(json.dumps(raw, ensure_ascii=False),
                                    encoding="utf-8")
                state_path.write_text(json.dumps(sorted(done), ensure_ascii=False),
                                      encoding="utf-8")
                print(f"  累计 {fetched} 篇")
            if fetched < len(todo):
                time.sleep(delay)
    finally:
        # 任意退出都保存进度（断点续跑）
        raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        state_path.write_text(json.dumps(sorted(done), ensure_ascii=False),
                              encoding="utf-8")

    # 语料内边 —— 统一标识符 = paper_id（与向量/R3 对齐；S2 给的是 arxiv_id，映射到 paper_id）
    pid_of = {p.arxiv_id: p.paper_id for p in c.list_papers() if p.arxiv_id}
    fwd: dict = {}
    for a, refs in raw.items():
        pa = pid_of.get(a)
        if pa is None:
            continue
        fwd[pa] = [pid_of[r] for r in refs if pid_of.get(r)]
    rev: dict = {}
    for pa, refs in fwd.items():
        for r in refs:
            rev.setdefault(r, []).append(pa)
    graph_path.write_bytes(pickle.dumps({"forward": fwd, "reverse": rev}, protocol=4))

    total_e = sum(len(v) for v in fwd.values())
    print(f"引用图已写 → {graph_path}")
    print(f"  语料内节点 {len(fwd)}，边 {total_e}（本轮新拉 {fetched} 篇；"
          f"原始引用累计 {sum(len(v) for v in raw.values())} 条）")
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description="R1 周报引用图拉取（S2 单条，增量断点续跑）")
    ap.add_argument("--limit", type=int, default=None, help="只拉前 n 篇（试跑）")
    ap.add_argument("--delay", type=float, default=1.6, help="每篇节流秒数（无 key 建议>1.5）")
    a = ap.parse_args()
    fetch_arxiv_refs(load_config(), limit=a.limit, delay=a.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
