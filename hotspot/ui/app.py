"""L6 交互层 —— Streamlit 单页问答（方案 §3.6）。

编排整条管线：方向选择 → 路由 → 装配上下文 → 答案生成 → 引用校验 → 展示。
数据/API 未就绪时给明确提示，而不是硬崩。

跑法：`streamlit run hotspot/ui/app.py`
"""

from __future__ import annotations

import json

import streamlit as st

import hotspot
from ..config import load_config, paths_from
from ..corpus import open_corpus
from ..direction import select
from ..llm import make_client
from ..retrieval.router import route, REFUSE_REASON
from ..retrieval.rewrite import rewrite
from ..retrieval.assemble import assemble
from ..reasoning.answer import generate_answer
from ..reasoning import verify
from ..schema import Answer, Route


def _card_body(cfg, max_cards: int = 20) -> str:
    ws = paths_from(cfg)["workspace"]
    lines: list[str] = []
    cards_dir = ws / "cards"
    if not cards_dir.exists():
        return "（尚未生成簇卡片——先跑 L3）"
    for p in sorted(cards_dir.glob("card_*.json"))[:max_cards]:
        d = json.loads(p.read_text(encoding="utf-8"))
        lines.append(
            f"[C:{d['cluster_id']}] {d['label']}（{d.get('scale', 0)} 篇，"
            f"{d.get('venue', {}).get('state', '')}）核心问题："
            f"{'；'.join(d.get('core_problems') or [])}"
        )
    return "\n".join(lines)


def _local_body(corpus, ids) -> tuple[str, dict]:
    body_lines, evidence = [], {}
    for pid in ids:
        p = corpus.get(pid)
        if p is None:
            continue
        body_lines.append(f"[P:{pid}] {p.title}\n{p.abstract or ''}")
        evidence[pid] = f"{p.title}\n{p.abstract or ''}"
    return "\n\n".join(body_lines), evidence


def run_qa(question: str, *, cfg=None, dry_run: bool = False) -> Answer:
    cfg = cfg or load_config()
    c = open_corpus(cfg)
    with c.lib():
        pool = select(c, question, cfg=cfg)
        ret = cfg.get("retrieval") or {}
        ret_client = (make_client(cfg, dry_run=False)
                      if not dry_run and ret.get("llm_router", True) else None)
        r = route(question, pool, llm=ret_client)
        if r is Route.REFUSE:
            return Answer(route=r, refused=True, refuse_reason=REFUSE_REASON)

        if r is Route.GLOBAL:
            body, evidence = _card_body(cfg), {}
        else:
            top = int(ret.get("top_k_local") or 8)
            ids = list(pool.paper_ids[:top])
            if ret_client is not None and ret.get("query_rewrite", True):
                q = rewrite(question, client=ret_client)
                if q:
                    ids = list(c.search(q, k=top)) or ids
            body, evidence = _local_body(c, ids)

        ctx = assemble(r, body=body, question=question,
                       allowed_ids=frozenset(evidence) | frozenset(
                           _ids_from_cards(cfg)))
        text = generate_answer(ctx, cfg=cfg, dry_run=dry_run)
        claims = verify.parse_claims(text)
        verdicts = verify.verify_claims(
            claims, allowed_ids=ctx.allowed_ids,
            evidence_lookup=evidence, client=None, cfg=cfg)
        return verify.build_answer(r, text, verdicts)


def _ids_from_cards(cfg):
    ws = paths_from(cfg)["workspace"]
    ids = set()
    if (ws / "cards").exists():
        for p in (ws / "cards").glob("card_*.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            ids.add(str(d["cluster_id"]))  # 裸 cluster_id，与 [C:id] 解析出的值一致
    return ids


def main() -> None:
    st.set_page_config(page_title="移动云杯 · 热点挖掘", layout="wide")
    st.title("移动云杯 · 本地论文库热点挖掘")
    q = st.text_input("问一个方向（如『2025 RLHF 方向最火的是什么？』）")
    dry = st.checkbox("dry-run（不真调 LLM）", value=True)
    if not q:
        st.info("输入问题开始。本库语料：2025 NeurIPS/ICLR/ICML/CoRL。")
        return
    ans = run_qa(q, dry_run=dry)
    if ans.refused:
        st.warning(ans.refuse_reason or "本库无相关内容")
        return
    st.markdown(f"**路由**：`{ans.route.value}`　·　**过滤无据陈述：{ans.n_filtered} 条**")
    st.markdown(ans.text)
    if ans.verdicts:
        with st.expander("引用校验明细"):
            for v in ans.verdicts:
                mark = {"supported": "✅", "unsupported": "⚠️",
                        "missing_id": "❌", "no_citation": "—"}[v.status]
                st.write(f"{mark} `{v.status}` — {v.claim.text[:120]}")
                if v.detail:
                    st.caption(v.detail)


if __name__ == "__main__":
    main()
