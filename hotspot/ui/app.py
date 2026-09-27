"""L6 交互层 —— Streamlit 单页问答（方案 §3.6）。

编排整条管线：方向选择 → 路由 → 装配上下文 → 答案生成 → 引用校验 → 展示。
数据/API 未就绪时给明确提示，而不是硬崩。

跑法：`streamlit run hotspot/ui/app.py`
"""

from __future__ import annotations

import json
import re

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

_WORD = re.compile(r"[A-Za-z0-9]+")


def _card_body(corpus, question, cfg, max_cards: int = 8) -> tuple[str, frozenset[str]]:
    """按问题词命中 label/core_problems 挑相关卡片，每张附 `evidence` 代表论文标题+摘要。

    解决"GLOBAL 只取前 20 张任意卡 + 无标题可引"的缺陷——LLM 才有 [P:] 可引。
    返回 `(body, allowed_ids)`（含 `[C:cluster_id]` 与 `[P:paper_id]` 的裸 id）。
    """
    ws = paths_from(cfg)["workspace"]
    cards_dir = ws / "cards"
    if not cards_dir.exists():
        return "（尚未生成簇卡片——先跑 L3）", frozenset()
    terms = {t for t in _WORD.findall(question.lower()) if t}
    scored = []
    for p in cards_dir.glob("card_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        # 命中面 = 中文 label/core_problems + **evidence 论文的英文标题**（query 多为英文/中文）
        ev_titles = []
        for eid in (d.get("evidence_ids") or [])[:2]:
            pap = corpus.get(eid)
            if pap is not None:
                ev_titles.append(pap.title or "")
        hay = " ".join([
            d.get("label") or "",
            " ".join(d.get("core_problems") or []),
            " ".join(ev_titles),
        ]).lower()
        s = (sum(1 for t in terms if t in hay) if terms else 0)
        scored.append((s, d))
    # 有词按相关度降序；无词按规模降序（保守）
    scored.sort(key=lambda x: (-x[0], -int(x[1].get("scale") or 0)))
    lines, used_ids = [], set()
    for s, d in scored[:max_cards]:
        cid = str(d["cluster_id"])
        used_ids.add(cid)
        cby = (d.get("venue") or {}).get("counts_by_year") or {}
        yr = "；".join(f"{y}:{int(sum(cb.values()))}篇"
                       for y, cb in sorted(cby.items()))
        head = (f"[C:{cid}] {d.get('label', '')}（"
                f"{d.get('scale', 0)} 篇，{d.get('venue', {}).get('state', '')}"
                + (f"，分年[{yr}]" if yr else "") + "）"
                f"核心问题：{'；'.join(d.get('core_problems') or [])}")
        ev = ""
        added = 0
        for eid in (d.get("evidence_ids") or []):
            pap = corpus.get(eid)
            if pap is None:
                continue
            used_ids.add(eid)
            ev += f"[P:{eid}] {pap.title}\n{pap.abstract[:100]}\n"
            added += 1
            if added >= 1:
                break
        lines.append(head + ("\n" + ev if ev else ""))
    body = "\n\n".join(lines) or "（无相关簇卡片）"
    return body, frozenset(used_ids)


def _local_body(corpus, ids) -> tuple[str, dict]:
    body_lines, evidence = [], {}
    for pid in ids:
        p = corpus.get(pid)
        if p is None:
            continue
        body_lines.append(f"[P:{pid}] {p.title}\n{(p.abstract or '')[:400]}")
        evidence[pid] = f"{p.title}\n{p.abstract or ''}"
    return "\n\n".join(body_lines), evidence


def run_qa(question: str, *, cfg=None, dry_run: bool = False,
           corpus=None) -> Answer:
    cfg = cfg or load_config()
    c = corpus or open_corpus(cfg)
    with c.lib():
        pool = select(c, question, cfg=cfg)
        ret = cfg.get("retrieval") or {}
        ret_client = (make_client(cfg, dry_run=False)
                      if not dry_run and ret.get("llm_router", True) else None)
        # 中文方向词（如「多智能体」）三路常常全空 → 若据此拒答会误拒（库里有英文 multi-agent）。
        # 含 CJK 时不因 pool 空直接 REFUSE，改走 LOCAL + LLM 翻译成英文再搜。
        has_cn = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in question)
        if pool.size == 0 and has_cn:
            r = Route.LOCAL
        else:
            r = route(question, pool, llm=ret_client)
        if r is Route.REFUSE:
            return Answer(route=r, refused=True, refuse_reason=REFUSE_REASON)

        if r is Route.GLOBAL:
            # 中文方向（如「多智能体」）需要先翻成英文再去匹配卡 label/英文标题
            qg = question
            if ret_client is not None and ret.get("query_rewrite", True):
                qr = rewrite(question, client=ret_client)
                if qr:
                    qg = f"{question} {qr}"
            body, g_ids = _card_body(c, qg, cfg)
            evidence = {}
            allowed = g_ids | frozenset(_ids_from_cards(cfg))
        else:
            top = int(ret.get("top_k_local") or 8)
            ids = list(pool.paper_ids[:top])
            if ret_client is not None and ret.get("query_rewrite", True):
                q = rewrite(question, client=ret_client)
                if q:
                    ids = list(c.search(q, k=top)) or ids
            body, evidence = _local_body(c, ids)
            allowed = frozenset(evidence) | frozenset(_ids_from_cards(cfg))

        ctx = assemble(r, body=body, question=question, allowed_ids=allowed)
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


def _corpus_cfg_key(cfg):
    return tuple((cfg.get("corpus") or {}).get("sources") or ())


@st.cache_resource
def _cached_corpus(_srcs):
    from ..corpus import open_corpus
    return open_corpus()


def main() -> None:
    st.set_page_config(page_title="移动云杯 · 热点挖掘", layout="wide")
    st.title("移动云杯 · 本地论文库热点挖掘")
    cfg = load_config()
    c = _cached_corpus(_corpus_cfg_key(cfg))   # 缓存，避免每问重读 22,747
    q = st.text_input("问一个方向（如『2025 RLHF 方向最火的是什么？』）")
    dry = st.checkbox("dry-run（不真调 LLM）", value=False)
    if not q:
        st.info("输入问题开始。语料：24+25 共 22,747 篇（NeurIPS/ICLR/ICML/CoRL）。"
                "可直接换问题连续问；取消 dry-run 才会真调 DeepSeek。")
        return
    ans = run_qa(q, dry_run=dry, corpus=c)
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
