"""L6 交互层 —— Streamlit 渲染适配（**只画，不算**）。

改造后这里**不再 import 任何算法模块**：问答、检索、聚类、校验、图谱、token 用量
全部由 `hotspot.api.Engine` 一次返回。前端（HTML/JS）走的是同一个引擎，只是
换成 HTTP 传输（`hotspot/api/http.py`）。

保留的三件事都属于"前端本分"：会话列表、API 配置、把响应画出来。

跑法（必须用这个入口，否则相对导入失败）：
    streamlit run scripts/run_ui.py --server.port 8502
"""

from __future__ import annotations

import threading
import time

import plotly.graph_objects as go
import streamlit as st

from ..api import STAGES, Engine
from ..api.contract import ApiError, AskRequest, LLMConfig
from ..config import load_config
from . import store

_VERDICT_MARK = {"supported": "✅", "unsupported": "⚠️",
                 "missing_id": "❌", "no_citation": "—"}
_ROUTE_LABEL = {"global": "全局热点", "local": "局部事实",
                "followup": "追问", "refuse": "拒答", "error": "出错"}


@st.cache_resource
def _engine() -> Engine:
    """进程内单例：语料与 LLM 连接只建一次（重跑不重新解析 2 万篇 jsonl）。"""
    return Engine()


def _meta(eng: Engine) -> dict:
    if "meta" not in st.session_state:
        st.session_state["meta"] = eng.meta().to_dict()
    return st.session_state["meta"]


# ══════════════════════════════════════════════════════════════════════
# 进度面板：worker 线程跑引擎，主线程按 stage 事件画真实进度
# ══════════════════════════════════════════════════════════════════════


def _run_with_progress(eng: Engine, req: AskRequest) -> dict:
    """把一次问答放进线程，主线程实时渲染进度。

    **不搞进度诈骗**：只有引擎给了 `cur/total` 才画进度条；否则只显示阶段名与
    已耗时。未进入的阶段保持灰色 ○，绝不提前点亮。
    """
    keys = [k for k, _ in STAGES]
    box: dict = {"idx": -1, "cur": None, "total": None, "done": False,
                 "result": None, "err": None, "t0": time.time(), "at": time.time()}
    MIN_SHOW = 0.25

    def on_event(ev: dict) -> None:
        key = ev.get("stage")
        if key in keys:
            i = keys.index(key)
            if i > box["idx"]:
                box["idx"] = i
        box["cur"], box["total"] = ev.get("cur"), ev.get("total")
        box["at"] = time.time()

    def worker() -> None:
        try:
            box["result"] = eng.ask(req, on_event=on_event)
        except ApiError as exc:
            box["err"] = f"{exc.code}：{exc.message}"
        except Exception as exc:  # noqa: BLE001
            box["err"] = f"{type(exc).__name__}: {exc}"
        finally:
            box["done"] = True

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    with st.status("处理中…", expanded=True) as status:
        total_line = st.empty()
        rows = [st.empty() for _ in keys]

        while True:
            now = time.time()
            total_line.caption(f"⏱ 总用时 {now - box['t0']:5.1f}s")
            for i, key in enumerate(keys):
                label = dict(STAGES)[key]
                if i < box["idx"]:
                    rows[i].progress(1.0, text=f"✅ {label}")
                elif i == box["idx"]:
                    cur, total = box["cur"], box["total"]
                    if total:
                        v = min(1.0, float(cur or 0) / float(total))
                        rows[i].progress(v, text=f"⏳ {label}  {cur}/{total}"
                                                f"  ·  {now - box['at']:.1f}s")
                    else:
                        rows[i].progress(0.0, text=f"⏳ {label}  ·  {now - box['at']:.1f}s")
                else:
                    rows[i].progress(0.0, text=f"○ {label}")
            if box["done"]:
                time.sleep(MIN_SHOW)
                break
            time.sleep(0.05)
        status.update(label="完成 ✓", state="complete")

    th.join(timeout=1)
    if box["err"]:
        return {"route": "error", "answer": {"text": f"出错：{box['err']}"},
                "warnings": [], "topics": [], "graph": None}
    return box["result"].to_dict() if box["result"] is not None else {}


# ══════════════════════════════════════════════════════════════════════
# 渲染
# ══════════════════════════════════════════════════════════════════════


def _render_topics(topics: list[dict]) -> None:
    if not topics:
        return
    with st.expander(f"主题簇（{len(topics)}）", expanded=True):
        for t in topics:
            vc = "、".join(f"{k} {v}" for k, v in (t.get("venue_counts") or {}).items())
            yc = "、".join(f"{k} {v}" for k, v in (t.get("year_counts") or {}).items())
            meta_bits = " · ".join(x for x in (f"{t.get('size', 0)} 篇", vc,
                                               (f"分年 {yc}" if yc else "")) if x)
            st.markdown(f"**[C:{t['cluster_id']}] {t.get('label') or '（未命名）'}**  \n"
                        f"{t.get('core') or ''}  \n"
                        f"<span style='color:#888;font-size:0.85em'>{meta_bits}</span>",
                        unsafe_allow_html=True)


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"引用来源（{len(citations)}）"):
        for c in citations:
            if c.get("kind") == "cluster":
                st.markdown(f"`[C:{c['id']}]` **{c.get('title', '')}**")
                continue
            bits = " · ".join(x for x in (c.get("venue") or "",
                                          str(c.get("year") or "") if c.get("year") else "") if x)
            title = c.get("title") or "（语料中无此 id）"
            if c.get("url"):
                st.markdown(f"`[P:{c['id']}]` [{title}]({c['url']})"
                            + (f" — {bits}" if bits else ""))
            else:
                st.markdown(f"`[P:{c['id']}]` {title}" + (f" — {bits}" if bits else ""))
            if c.get("abstract"):
                st.caption(c["abstract"][:220])


def _graph_figure(g: dict):
    """把 `graph{nodes,edges}` 画成可交互路线图（只做渲染，布局数据来自服务层）。"""
    nodes = g.get("nodes") or []
    if not nodes:
        return None
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    xs = [n.get("step") or i + 1 for i, n in enumerate(nodes)]
    ys = [float(n.get("cluster") if n.get("cluster") is not None else 0)
          + 0.12 * (i % 5) for i, n in enumerate(nodes)]
    labels = [f"{n.get('step')}. {(n.get('title') or n['id'])[:38]}"
              + ("★" if n.get("new_cover") else "") for n in nodes]
    hovers = [f"<b>{n.get('title') or n['id']}</b><br>"
              f"第 {n.get('step')} 步 · 簇 {n.get('cluster')} · "
              f"覆盖累积 {n.get('covered_so_far')}" for n in nodes]

    fig = go.Figure()
    for e in (g.get("edges") or []):
        a, b = idx.get(e.get("source")), idx.get(e.get("target"))
        if a is None or b is None or a == b:
            continue
        cite = e.get("kind") == "cite"
        fig.add_trace(go.Scatter(
            x=[xs[a], xs[b]], y=[ys[a], ys[b]], mode="lines",
            line=dict(color="crimson" if cite else "#999",
                      width=1.4 if cite else 2,
                      dash="dot" if cite else "solid"),
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=labels,
        textposition="top center", textfont=dict(size=10),
        hovertext=hovers, hoverinfo="text",
        marker=dict(size=26, color="#a5d6ff", line=dict(width=1, color="#444"))))
    fig.update_layout(
        title="Streit-Garg 关键路径（必备先序）", height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        dragmode="pan")
    return fig


def _render_assistant(m: dict) -> None:
    with st.chat_message("assistant"):
        usage = m.get("usage") or {}
        caption = (f"路由 `{_ROUTE_LABEL.get(m.get('route'), m.get('route'))}`"
                   f" · 过滤无据 {m.get('n_filtered', 0)} 条")
        if usage.get("llm_calls"):
            caption += (f" · {usage.get('elapsed_s', 0)}s"
                        f" · LLM {usage['llm_calls']} 次"
                        f" · token {usage.get('tokens_in', 0)}+{usage.get('tokens_out', 0)}")
            if usage.get("cost_yuan"):
                caption += f" · ¥{usage['cost_yuan']:.4f}"
        st.caption(caption)
        if m.get("error"):
            st.error(m["error"].get("message") or "出错了")
        if m.get("warnings"):
            with st.expander(f"降级提示（{len(m['warnings'])}）"):
                for w in m["warnings"]:
                    st.write("· " + w)
        _render_topics(m.get("topics") or [])
        g = m.get("graph")
        if g and g.get("nodes"):
            fig = _graph_figure(g)
            if fig is not None:
                with st.expander("Streit-Garg 关键路径路线图", expanded=True):
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"scrollZoom": True})
                    st.caption(f"算法 `{g.get('algo')}` · 种子 {g.get('seed_count')} · "
                               f"value {g.get('value')} · Δ {g.get('delta')} · "
                               f"conn {g.get('conn')} · "
                               f"覆盖 {len(g.get('covered_clusters') or [])} 簇")
        st.markdown(m.get("text") or "（空）")
        verdicts = m.get("verdicts") or []
        if verdicts:
            bad = [v for v in verdicts if v.get("status") != "supported"]
            with st.expander(f"引用校验明细（{len(verdicts)} 条，"
                             f"{len(bad)} 条有问题）"):
                for v in verdicts:
                    st.write(f"{_VERDICT_MARK.get(v.get('status'), '·')} "
                             f"`{v.get('status')}` — {(v.get('text') or '')[:120]}")
                    if v.get("detail"):
                        st.caption(v["detail"])
        _render_citations(m.get("citations") or [])


def _render_message(m: dict) -> None:
    if m.get("role") == "user":
        with st.chat_message("user"):
            st.markdown(m.get("text") or "")
    else:
        _render_assistant(m)


# ══════════════════════════════════════════════════════════════════════
# 主页面
# ══════════════════════════════════════════════════════════════════════


def _init_state(cfg) -> None:
    ss = st.session_state
    if "sessions" not in ss:
        ss["sessions"] = store.load_sessions(cfg)
    if "cur" not in ss or ss["cur"] not in ss["sessions"]:
        if ss["sessions"]:
            ss["cur"] = list(ss["sessions"].keys())[-1]
        else:
            sid = store.new_session_id()
            ss["sessions"][sid] = {"name": "新对话", "history": []}
            ss["cur"] = sid
    ss.setdefault("api", {})
    if not ss["api"]:
        ss["api"] = store.load_api_settings(cfg) or {}
    ss.setdefault("src", "meeting")


def _new_session(cfg) -> None:
    sid = store.new_session_id()
    n = len(st.session_state["sessions"]) + 1
    st.session_state["sessions"][sid] = {"name": f"会话 {n}", "history": []}
    st.session_state["cur"] = sid
    store.save_sessions(cfg, st.session_state["sessions"])


def main() -> None:
    st.set_page_config(page_title="移动云杯 · 热点挖掘", layout="wide")
    st.title("移动云杯 · 本地论文库热点挖掘")

    cfg = load_config()
    eng = _engine()
    meta = _meta(eng)
    _init_state(cfg)

    sources = {s["key"]: s for s in meta.get("sources") or []}
    src_keys = list(sources) or ["meeting"]

    # 检索参数：整个侧栏里生成，提交时随请求带走（无会话状态）
    params_now: dict = {}

    with st.sidebar:
        st.subheader("会话")
        cur_src = st.session_state.get("src", "meeting")
        idx = src_keys.index(cur_src) if cur_src in src_keys else 0
        src_key = st.selectbox("数据源", src_keys, index=idx,
                               format_func=lambda k: sources.get(k, {}).get("label", k),
                               key="src_sel")
        st.session_state["src"] = src_key

        st.button("＋ 新对话", use_container_width=True, on_click=_new_session, args=(cfg,))
        sess = st.session_state["sessions"]
        names = list(sess.keys())
        if names:
            cur = st.session_state["cur"] if st.session_state["cur"] in names else names[0]
            pick = st.selectbox("当前会话", names,
                                index=names.index(cur),
                                format_func=lambda sid: sess[sid].get("name", "会话"),
                                key="pick_session")
            if pick != st.session_state["cur"]:
                st.session_state["cur"] = pick

        with st.expander("API 配置"):
            api = st.session_state.get("api") or {}
            b = st.text_input("base_url",
                              value=api.get("base_url") or meta.get("llm", {}).get("base_url", ""))
            mo = st.text_input("model",
                               value=api.get("model") or meta.get("llm", {}).get("model", ""))
            kx = st.text_input("API key", value=api.get("api_key") or "", type="password")
            if st.button("✓ 保存到本地"):
                st.session_state["api"] = {"base_url": b.strip() or None,
                                           "model": mo.strip() or None,
                                           "api_key": kx.strip() or None}
                store.save_api_settings(cfg, st.session_state["api"])
                st.success("已保存")

        with st.expander("检索参数", expanded=False):
            # 表单**由 meta.params 自动生成**：加参数只改 contract.PARAM_SPEC，这里不用动
            saved = store.load_params(cfg)
            groups: dict[str, list] = {}
            for prm in meta.get("params") or []:
                groups.setdefault(prm.get("group") or "其它", []).append(prm)
            for gname, items in groups.items():
                st.caption(gname)
                for prm in items:
                    key = prm["key"]
                    label = prm.get("label") or key
                    lo, hi = prm.get("min"), prm.get("max")
                    if prm.get("type") == "bool":
                        params_now[key] = st.checkbox(
                            label, value=bool(saved.get(key, prm.get("value"))),
                            key=f"pp_{key}", help=prm.get("help") or "")
                    else:
                        val = saved.get(key, prm.get("value")) or 0
                        try:
                            val = int(val)
                        except (TypeError, ValueError):
                            val = int(prm.get("default") or 0)
                        if lo is not None:
                            val = max(val, int(lo))
                        if hi is not None:
                            val = min(val, int(hi))
                        params_now[key] = st.number_input(
                            label, value=val,
                            min_value=int(lo) if lo is not None else None,
                            max_value=int(hi) if hi is not None else None,
                            step=int(prm.get("step") or 1),
                            key=f"pp_{key}", help=prm.get("help") or "")

        st.markdown("---")
        info = sources.get(src_key) or {}
        st.caption(info.get("note") or "")
        if info.get("papers"):
            extra = f"；本周新增 {info['new_this_week']} 篇" if info.get("new_this_week") else ""
            st.caption(f"{info.get('label', src_key)}：{info['papers']:,} 篇{extra}")

    history = st.session_state["sessions"][st.session_state["cur"]]["history"]
    for m in history:
        _render_message(m)
    if not history:
        st.info("输入问题开始。可开新对话，或打开「追问」基于前文继续（不重新检索）。")

    follow = st.toggle("追问（基于前文，不重新检索）",
                       help="打开=接着上文聊；关闭=当作新问题重新检索")
    q = st.chat_input("问一个方向（如『2025 RLHF 方向最火的是什么？』）")
    if not q:
        return

    api_now = {"base_url": (b.strip() if b else "") or None,
               "model": (mo.strip() if mo else "") or None,
               "api_key": (kx.strip() if kx else "") or None}
    st.session_state["api"] = api_now
    store.save_api_settings(cfg, api_now)

    history.append({"role": "user", "text": q})

    # 追问：把历史（含上一轮引用的论文 id）交给服务层，由它装配前文语料
    hist_msgs = []
    if follow and len(history) > 1:
        for m in history[:-1]:
            hist_msgs.append({
                "role": m["role"], "text": m.get("text") or "",
                "citation_ids": m.get("citation_ids") or []})

    store.save_params(cfg, params_now)
    req = AskRequest.from_dict({
        "question": q,
        "source": src_key,
        "mode": "followup" if hist_msgs else "new",
        "history": hist_msgs,
        "llm": LLMConfig(base_url=api_now["base_url"], model=api_now["model"],
                         api_key=api_now["api_key"]).__dict__,
        "params": params_now,
    })
    try:
        resp = _run_with_progress(eng, req)
    except ApiError as exc:
        resp = {"route": "error", "answer": {"text": f"{exc.code}：{exc.message}"},
                "warnings": [], "topics": [], "graph": None}

    answer = resp.get("answer") or {}
    record = {
        "role": "assistant",
        "text": answer.get("text") or "",
        "route": resp.get("route"),
        "n_filtered": answer.get("n_filtered", 0),
        "verdicts": answer.get("verdicts") or [],
        "citations": answer.get("citations") or [],
        "topics": resp.get("topics") or [],
        "graph": resp.get("graph"),
        "usage": resp.get("usage") or {},
        "warnings": resp.get("warnings") or [],
        "error": resp.get("error"),
        # 追问要把这些 id 带回去（服务层据此把前文论文并进上下文）
        "citation_ids": [c["id"] for c in (answer.get("citations") or [])
                         if c.get("kind") == "paper"],
    }
    history.append(record)
    store.save_sessions(cfg, st.session_state["sessions"])
    st.rerun()


if __name__ == "__main__":
    main()
