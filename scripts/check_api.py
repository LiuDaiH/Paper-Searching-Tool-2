"""接口自检：契约往返 + HTTP 冒烟（**离线、秒级、确定性**）。

前端改页面、算法改内核之前先跑它：只要它还绿，接口就没被改坏。

    python scripts/check_api.py

查四件事：
  1. 契约往返：`AskRequest.from_dict` → `to_dict` → JSON 可序列化
  2. 参数白名单：未知键/越界/类型错都**必须报错**（不静默忽略）
  3. 响应形状：`AskResponse.to_dict()` 的键齐全且 JSON 可序列化
  4. 检索语义：长句 AND 必空，并集检索必须能召回（2026-10-18 真实故障回归）
  5. HTTP 骨架：/api/health、/api/meta、/api/ask（含 SSE 流式）、坏请求 400、未知路由 404

⚠️ 不碰语料、不碰 LLM：用 StubEngine 顶替，所以在任何机器上都能跑绿。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import threading
import urllib.error
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=SyntaxWarning)

import hotspot  # noqa: E402,F401 —— 装闸① + 接线 dasc
from hotspot.api.contract import (  # noqa: E402
    CONTRACT_VERSION,
    PARAM_SPEC,
    ApiError,
    AskRequest,
    AskResponse,
    Citation,
    MetaResponse,
    SourceInfo,
    Topic,
    Usage,
    stage_event,
)
from hotspot.api.engine import Engine  # noqa: E402
from hotspot.api.http import create_server  # noqa: E402

FAILS: list[str] = []
CHECKS = 0
QUIET = False


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        if not QUIET:
            print(f"  ok  {name}")
    else:
        FAILS.append(f"{name}: {detail}")
        if not QUIET:
            print(f"  FAIL {name}  {detail}")


# ══════════════════════════════════════════════════════════════════════
# 1) 契约往返
# ══════════════════════════════════════════════════════════════════════


def check_contract() -> None:
    print("[1] 契约往返")
    req = AskRequest.from_dict({
        "question": "2025 RLHF 方向最火的是什么？",
        "source": "meeting",
        "mode": "followup",
        "history": [{"role": "user", "text": "前一个问题"},
                    {"role": "assistant", "text": "回答", "citation_ids": ["p1"]}],
        "llm": {"base_url": "https://x/v1", "model": "m", "api_key": "k"},
        "params": {"max_topics": 4, "streit_garg": False},
        "scope": {"years": [2025], "years_explicit": True, "venues": ["ICLR"]},
    })
    d = req.to_dict()
    json.dumps(d, ensure_ascii=False)
    check("字段解析", req.mode == "followup" and req.history[1].citation_ids == ["p1"])
    check("参数归一", req.params == {"max_topics": 4, "streit_garg": False})
    check("scope 解析", req.scope["years"] == [2025]
          and req.scope["years_explicit"] is True
          and req.scope["venues"] == ["ICLR"])
    check("key 不回显", d["llm"]["api_key"] == "***")
    check("JSON 可序列化", isinstance(json.dumps(d, ensure_ascii=False), str))

    for bad, why in (
        ({"question": ""}, "空问题"),
        ({"question": "x", "source": "nope"}, "非法 source"),
        ({"question": "x", "mode": "nope"}, "非法 mode"),
        ({"question": "x", "params": {"nope": 1}}, "未知参数"),
        ({"question": "x", "params": {"top_k_local": 999}}, "参数越界"),
        ({"question": "x", "params": {"top_k_local": "abc"}}, "参数类型"),
        ({"question": "x", "history": [{"role": "bot", "text": "x"}]}, "非法 role"),
    ):
        try:
            AskRequest.from_dict(bad)
        except ApiError:
            continue
        FAILS.append(f"应报错却通过：{why}")
        print(f"  FAIL 应报错却通过：{why}")

    resp = AskResponse(
        request_id="t1", source="meeting", mode="new", route="global",
        text="答案 [C:3] 与 [P:p1]", n_filtered=1,
        verdicts=[], topics=[Topic(cluster_id=3, label="示例", size=2)],
        citations=[Citation(id="p1", kind="paper", title="论文", year=2025)],
        usage=Usage(llm_calls=2, tokens_in=10, tokens_out=5, cost_yuan=0.0001))
    rd = resp.to_dict()
    json.dumps(rd, ensure_ascii=False)
    for key in ("contract_version", "request_id", "route", "answer", "topics",
                "graph", "usage", "warnings", "error", "meta"):
        check(f"响应含 {key}", key in rd)
    check("响应版本", rd["contract_version"] == CONTRACT_VERSION)
    check("引用摊平", rd["answer"]["citations"][0]["title"] == "论文")
    check("参数表非空", len(PARAM_SPEC) > 0)


# ══════════════════════════════════════════════════════════════════════
# 2) HTTP 冒烟（StubEngine）
# ══════════════════════════════════════════════════════════════════════


class StubEngine:
    """假引擎：不读语料、不调 LLM，只验证传输层。"""

    def meta(self) -> MetaResponse:
        return MetaResponse(
            sources=[SourceInfo(key="meeting", label="会议 24+25", papers=1200)],
            params=[p.to_dict() for p in PARAM_SPEC],
            capabilities={"stream": True, "dry_run": True},
            llm={"base_url": "stub", "model": "stub", "has_server_key": False})

    def ask(self, req, on_event=None):
        for key in ("parse", "retrieve", "name"):
            if on_event:
                on_event(stage_event(key, elapsed=0.01))
        if on_event:
            on_event(stage_event("verify", cur=1, total=2, elapsed=0.02))
            on_event(stage_event("verify", cur=2, total=2, elapsed=0.03))
        return AskResponse(
            request_id="stub", source=req.source, mode=req.mode, route="global",
            text=f"关于「{req.question}」的答案 [C:1][P:p9]",
            topics=[Topic(cluster_id=1, label="示例主题", size=9)],
            citations=[Citation(id="p9", kind="paper", title="示例论文", year=2025)],
            usage=Usage(llm_calls=1, tokens_in=100, tokens_out=50,
                        elapsed_s=0.03, stages={"parse": 0.01}))


def _get(url: str):
    req = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read().decode("utf-8")


def _post(url: str, payload=None, raw: bytes | None = None):
    data = raw if raw is not None else json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={
                                     "Content-Type": "application/json",
                                     "Connection": "close",
                                 })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def check_http() -> None:
    print("[4] HTTP 冒烟")
    srv = create_server("127.0.0.1", 0, engine=StubEngine(), quiet=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        status, body = _get(base + "/api/health")
        check("GET /api/health", status == 200 and json.loads(body)["ok"] is True)

        status, body = _get(base + "/api/meta")
        meta = json.loads(body)
        check("GET /api/meta", status == 200 and meta["contract_version"] == CONTRACT_VERSION)
        check("meta.params 可驱动设置页", len(meta["params"]) >= 10
              and {"key", "label", "value", "min", "max"} <= set(meta["params"][0]))

        status, body = _post(base + "/api/ask", {"question": "测试问题"})
        resp = json.loads(body)
        check("POST /api/ask", status == 200 and resp["route"] == "global")
        check("引用已摊平", resp["answer"]["citations"][0]["title"] == "示例论文")
        check("用量字段", resp["usage"]["llm_calls"] == 1)

        status, body = _post(base + "/api/ask",
                             {"question": "x", "params": {"nope": 1}})
        check("未知参数 → 400", status == 400 and json.loads(body)["error"]["code"]
              == "bad_request")

        status, body = _post(base + "/api/ask", raw=b"{not json")
        check("坏 JSON → 400", status == 400)

        status, body = _post(base + "/api/nope", {})
        check("未知路由 → 404", status == 404)

        # SSE：要读到 stage 事件与最终 result 事件
        req = urllib.request.Request(
            base + "/api/ask",
            data=json.dumps({"question": "流式", "stream": True}).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            ctype = r.headers.get("Content-Type", "")
            sse = r.read().decode("utf-8")
        check("SSE Content-Type", "text/event-stream" in ctype, ctype)
        check("SSE 有进度事件", '"type": "stage"' in sse and '"stage": "retrieve"' in sse)
        check("SSE 有最终结果", "event: result" in sse and "示例主题" in sse)
    finally:
        srv.shutdown()
        srv.server_close()


# ══════════════════════════════════════════════════════════════════════
# 3) 检索语义：corpus.search 是全词 AND，长句必须走并集
# ══════════════════════════════════════════════════════════════════════


def check_search_semantics() -> None:
    """用合成语料验证：长句直接搜会 0 命中，`_search_union` 必须能召回。

    这是 2026-10-18 实测到的真实故障：问「多智能体」→ 改写成长英文句 →
    AND 命中 0 → 空上下文 → 模型答"本库无相关内容"。不修就会被当成"库里没有"。
    """
    print("[2] 检索语义")
    from hotspot.corpus import MergedCorpus, Paper

    c = MergedCorpus.__new__(MergedCorpus)
    papers = [
        Paper(paper_id="p1", title="Multi-Agent Reinforcement Learning for Coordination",
              abstract="We study multi-agent systems and cooperative agents."),
        Paper(paper_id="p2", title="Scaling MARL with Centralized Critics",
              abstract="A multi-agent reinforcement learning method."),
    ]
    c._papers = papers
    c._by_id = {p.paper_id: p for p in papers}

    long_q = "multi-agent reinforcement learning methods for cooperative robotics systems"
    check("长句 AND 命中为 0（旧行为）", c.search(long_q, k=10) == [])
    hits = Engine._search_union(c, long_q, 8)
    check("并集检索能召回", len(hits) >= 2 and "p1" in hits, str(hits))
    check("空查询不炸", Engine._search_union(c, "", 8) == [])


def check_hybrid() -> None:
    """混合检索的纯函数部分：RRF 必须把双通道共同命中排在前面。"""
    from hotspot.retrieval.hybrid import rrf_fuse

    print("[2b] hybrid RRF")
    fused = rrf_fuse({
        "bm25": [("a", 10.0), ("b", 9.0)],
        "vector": [("b", 0.8), ("c", 0.7)],
    })
    check("RRF 双通道优先", max(fused, key=fused.get) == "b", str(fused))
    check("RRF 保留单通道候选", {"a", "b", "c"} <= set(fused), str(fused))


def check_coverage() -> None:
    """全局粗簇归属与题型路径必须可独立测试，不依赖工作区产物。"""
    import numpy as np

    from hotspot.retrieval.coverage import (
        GlobalCoverageIndex,
        coverage_policy,
        select_candidates,
    )

    print("[2g] global coverage")
    coverage = GlobalCoverageIndex(
        assignment={"a": 1, "c": 1, "e": 1, "b": 2, "f": 2},
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        centroid_ids=(1, 2),
        cluster_sizes={1: 3, 2: 2},
        spread_state={1: "铺开", 2: "新进"},
        vector_ids=["a", "c", "e", "b", "f", "d", "low"],
        vector_rows={pid: i for i, pid in enumerate(
            ["a", "c", "e", "b", "f", "d", "low"]
        )},
        index_vectors=np.array([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            [0.7071, 0.7071],
        ], dtype=np.float32),
    )
    check("已归属论文使用稳定簇", coverage.assign("a") == (1, "cluster_membership"))
    check("未归属论文使用最近质心",
          coverage.assign("d") == (2, "nearest_centroid"))
    check("低 margin 归属不伪装成确定簇",
          coverage.assign("low") == (None, "low_confidence"))
    kept, selection = select_candidates(
        coverage, ["a", "c", "e", "b", "f"], target_share=0.60,
    )
    check("趋势题保留粗主题候选",
          set(kept) == {"a", "c", "e"}
          and selection["selected_cluster_ids"] == [1],
          str(selection))
    check("题型覆盖策略稳定",
          coverage_policy({"answer_mode": "global_trend"}) == "coarse_then_fine"
          and coverage_policy({"answer_mode": "paper_fact"}) == "paper_first"
          and coverage_policy({"answer_mode": "compare"}) == "per_side"
          and coverage_policy({"answer_mode": "relation"}) == "dual_concept")


def check_query_plan() -> None:
    """QueryPlan 的需求矩阵、子问题和回答策略必须稳定归一化。"""
    from types import SimpleNamespace
    from hotspot.subcluster import ANSWER_MODES, PARSE_SCHEMA, parse_query

    print("[2d] query plan")

    class FakeClient:
        def __init__(self, data):
            self.data = data

        def ask(self, question, *, system, schema, required):
            return SimpleNamespace(ok=True, data=self.data)

    base = {
        "topic_en": "DPO versus RLOO sample efficiency",
        "core_terms": ["DPO", "RLOO"],
        "core_concepts": [
            {"name": "DPO", "aliases": ["Direct Preference Optimization"], "required": True},
            {"name": "RLOO", "aliases": ["REINFORCE leave-one-out"], "required": True},
        ],
        "anchors": ["DPO", "RLOO"],
        "keywords": ["DPO", "RLOO", "sample efficiency"],
        "years": [],
        "intent": "compare",
        "input_class": "valid",
        "confidence": 0.9,
    }
    structured = dict(
        base,
        requirements=[
            {"kind": "comparison", "value": "DPO", "required": True},
            {"kind": "side", "value": "RLOO", "required": True},
            {"kind": "metric_name", "value": "sample efficiency", "required": True},
        ],
        sub_questions=[
            {"id": "model-id-ignored", "question": "How sample-efficient is DPO?", "required": True},
            {"question": "How sample-efficient is RLOO?", "required": True},
        ],
        answer_mode="compare",
    )
    plan = parse_query("Compare DPO and RLOO on sample efficiency", FakeClient(structured))
    check("规划含 requirements", len(plan["requirements"]) == 3, str(plan))
    check("需求 kind 已归一",
          [r["kind"] for r in plan["requirements"]]
          == ["comparison_side", "comparison_side", "metric"], str(plan))
    check("子问题保留并重编号",
          [q["id"] for q in plan["sub_questions"]] == ["q1", "q2"], str(plan))
    check("answer_mode 合法",
          plan["answer_mode"] in ANSWER_MODES and plan["answer_mode"] == "compare",
          str(plan))
    check("规划可 JSON 序列化", bool(json.dumps(plan, ensure_ascii=False)))
    check("schema 含新增字段",
          {"requirements", "sub_questions", "answer_mode"}
          <= set(PARSE_SCHEMA["properties"]))

    fallback = dict(
        base,
        requirements="shared source",
        sub_questions="Which source supports both topics?",
        answer_mode="not-a-mode",
        intent="relation",
    )
    plan2 = parse_query("What source do DPO and RLOO share?", FakeClient(fallback))
    check("字符串需求可归一",
          plan2["requirements"] == [{"kind": "other", "value": "shared source", "required": True}],
          str(plan2))
    check("非法 answer_mode 按 intent 回退",
          plan2["answer_mode"] == "relation", str(plan2))

    combined = dict(
        structured,
        requirements=[{"kind": "comparison_side", "value": "DPO vs RLOO", "required": True}],
        sub_questions=[{"question": "Compare DPO and RLOO", "required": True}],
    )
    plan3 = parse_query("Compare DPO and RLOO", FakeClient(combined))
    check("合并的对比对象会拆分",
          [r["value"] for r in plan3["requirements"]] == ["DPO", "RLOO"], str(plan3))

    instruction_only = dict(
        base,
        requirements=[{
            "kind": "other",
            "value": "Explain the core innovation and contrast it with PPO",
            "required": True,
        }, {
            "kind": "method",
            "value": "RLOO core innovation",
            "required": True,
        }],
        sub_questions=[{"question": base["topic_en"], "required": True}],
        answer_mode="paper_fact",
    )
    plan4 = parse_query("Explain DPO versus RLOO", FakeClient(instruction_only))
    check("动作/泛化需求不进入硬门控",
          all(r["required"] is False for r in plan4["requirements"]), str(plan4))


def check_plan_consumption() -> None:
    """answer_mode/sub_questions 必须真正改变 Engine 的分派路径。"""
    import hotspot.api.engine as engine_mod

    print("[2e] query plan consumption")

    class DispatchEngine(engine_mod.Engine):
        def _ask_local_intent(self, *args, **kwargs):
            return engine_mod._Turn(route="local", text="local", meta={"mode": "paper_fact"})

        def _ask_compare_intent(self, *args, **kwargs):
            return engine_mod._Turn(route="local", text="compare", meta={"mode": "compare"})

        def _ask_multi_intent(self, *args, **kwargs):
            return engine_mod._Turn(route="local", text="multi", meta={"mode": "multi"})

        def _ask_relation_intent(self, *args, **kwargs):
            return engine_mod._Turn(route="local", text="relation", meta={"mode": "relation"})

    engine = DispatchEngine(cfg={})
    request = AskRequest.from_dict({
        "question": "test", "source": "meeting", "dry_run": False,
    })
    original = engine_mod.parse_query
    plans = [
        (
            "paper_fact",
            {
                "input_class": "valid", "intent": "fact",
                "answer_mode": "paper_fact", "sub_questions": [],
            },
        ),
        (
            "compare",
            {
                "input_class": "valid", "intent": "compare",
                "answer_mode": "compare", "requirements": [],
                "sub_questions": [],
            },
        ),
        (
            "multi",
            {
                "input_class": "valid", "intent": "fact",
                "answer_mode": "compare",
                "sub_questions": [
                    {"id": "q1", "question": "first", "required": True},
                    {"id": "q2", "question": "second", "required": True},
                ],
            },
        ),
        (
            "relation",
            {
                "input_class": "valid", "intent": "relation",
                "answer_mode": "relation", "core_concepts": [],
                "sub_questions": [],
            },
        ),
    ]
    try:
        for expected, plan in plans:
            engine_mod.parse_query = lambda *args, _plan=plan, **kwargs: _plan
            turn = engine._ask_intent(
                request, cfg={}, c=None, client=object(),
                emit=lambda *args, **kwargs: None,
            )
            check(f"{expected} 分派已消费",
                  turn is not None and turn.meta.get("mode") == expected,
                  str(turn))
        engine_mod.parse_query = lambda *args, **kwargs: {
            "input_class": "out_of_scope", "intent": "trend",
            "answer_mode": "refuse", "sub_questions": [],
        }
        turn = engine._ask_intent(
            request, cfg={}, c=None, client=object(),
            emit=lambda *args, **kwargs: None,
        )
        check("域外问题在检索前拒答",
              turn is not None and turn.refused and turn.route == "refuse",
              str(turn))
    finally:
        engine_mod.parse_query = original


def check_claim_first() -> None:
    """结构化 claim 生成必须过滤非法 ID，并由程序统一渲染引用。"""
    from types import SimpleNamespace
    from hotspot.reasoning.answer import generate_claim_answer
    from hotspot.schema import Context, Route
    from hotspot.api.contract import Verdict

    print("[2f] claim-first answer")

    class ClaimClient:
        def __init__(self, claims):
            self.claims = claims

        def ask(self, user, *, system, schema, required, **kwargs):
            return SimpleNamespace(ok=True, data={"claims": self.claims})

    class FailedClaimClient:
        def ask(self, user, *, system, schema, required, **kwargs):
            return SimpleNamespace(ok=False, error="schema violation")

    context = Context(
        route=Route.LOCAL,
        body="[P:p3] DPO removes the reward model.\n[P:p4] RLOO reduces variance.",
        question="Compare DPO and RLOO",
        allowed_ids=frozenset({"p3", "p4"}),
    )
    result = generate_claim_answer(
        context,
        cfg={"llm": {"model_pro": "stub"}},
        client=ClaimClient([
            {"text": "DPO removes the reward model", "evidence_ids": ["p3"]},
            {"text": "RLOO reduces variance", "evidence_ids": ["p4", "bad"]},
        ]),
    )
    check("claim-first 生成成功", result.ok, result.reason)
    check("claim-first 自动渲染引用",
          result.text
          == "DPO removes the reward model [P:p3]\nRLOO reduces variance [P:p4]",
          result.text)
    check("claim-first 丢弃非法 ID",
          all("bad" not in item for item in result.claims), str(result.claims))

    failed = generate_claim_answer(
        context,
        cfg={"llm": {"model_pro": "stub"}},
        client=ClaimClient([
            {"text": "Uncited claim", "evidence_ids": ["bad"]},
        ]),
    )
    check("非法结构化 claim 可回退到合法抽取 claim",
          failed.ok and "[P:p3]" in failed.text, str(failed))

    extracted = generate_claim_answer(
        context,
        cfg={"llm": {"model_pro": "stub"}},
        client=ClaimClient([]),
    )
    check("空结构化结果可使用抽取式 claim",
          extracted.ok and "[P:p3]" in extracted.text, str(extracted))

    empty_context = Context(
        route=Route.LOCAL, body="", question="empty",
        allowed_ids=frozenset(),
    )
    no_evidence = generate_claim_answer(
        empty_context,
        cfg={"llm": {"model_pro": "stub"}},
        client=ClaimClient([]),
    )
    check("无证据时不冒充答案", not no_evidence.ok, str(no_evidence))
    failed_output = generate_claim_answer(
        context,
        cfg={"llm": {"model_pro": "stub"}},
        client=FailedClaimClient(),
    )
    check("结构化调用失败也使用抽取式 claim",
          failed_output.ok and "[P:p3]" in failed_output.text,
          str(failed_output))

    punctuated = generate_claim_answer(
        context,
        cfg={"llm": {"model_pro": "stub"}},
        client=ClaimClient([
            {"text": "DPO removes the reward model.", "evidence_ids": ["p3"]},
        ]),
    )
    check("引用标记位于句号之前",
          "[P:p3]." in punctuated.text,
          punctuated.text)

    cluster_context = Context(
        route=Route.GLOBAL,
        body="[C:43] RLHF 热点：偏好反馈与策略优化",
        question="RLHF 热点",
        allowed_ids=frozenset({"43"}),
    )
    cluster_answer = generate_claim_answer(
        cluster_context,
        cfg={"llm": {"model_pro": "stub"}},
        client=ClaimClient([]),
    )
    check("全局卡片可作为抽取 claim",
          cluster_answer.ok and "[C:43]" in cluster_answer.text,
          str(cluster_answer))
    filtered_text, filtered_verdicts = Engine(cfg={})._filter_claim_first_output(
        [Verdict(status="supported", text="kept [P:p3]")],
        context,
        [],
    )
    check("契约 verdict 可安全过滤",
          filtered_text == "kept [P:p3]" and len(filtered_verdicts) == 1,
          filtered_text)


def check_evidence() -> None:
    """core_concepts + anchors + evidence 的最小行为测试。"""
    from types import SimpleNamespace
    from hotspot.reasoning.evidence import evaluate

    print("[2c] evidence sufficiency")

    class Corpus:
        def get(self, pid):
            data = {
                "p1": "RLOO improves REINFORCE leave-one-out baseline variance reduction",
                "p2": "Quantum algorithms for computing equilibria",
                "p3": "Direct Preference Optimization removes an explicit reward model",
                "p4": "RLOO improves sample efficiency for policy optimization",
                "p5": "Dreamer world model uses latent dynamics and actor-critic optimization",
                "p6": "Visuomotor policy uses RGB image input and robot state",
                "p7": "Traffic signal control with reinforcement learning reduces congestion",
                "p8": "Urban traffic signal control is optimized by reinforcement learning",
                "p9": "Deep segmentation networks for natural-image benchmarks",
                "p10": "Clinical benchmarks for image analysis",
            }
            return SimpleNamespace(title=data.get(pid, ""), abstract="")

    hits = [SimpleNamespace(paper_id="p1"), SimpleNamespace(paper_id="p2")]
    compare_hits = [SimpleNamespace(paper_id="p3"), SimpleNamespace(paper_id="p4")]
    plan = {
        "intent": "fact",
        "core_concepts": [{
            "name": "RLOO", "aliases": ["RLOO"], "required": True,
        }],
        "anchors": ["RLOO"],
    }
    check("anchor 命中可回答",
          evaluate(Corpus(), hits, plan).decision == "answer")

    plan2 = {
        "intent": "fact",
        "core_concepts": [
            {"name": "quantum computing", "aliases": ["quantum"], "required": True},
            {"name": "Schrodinger cat", "aliases": ["cat"], "required": True},
        ],
        "anchors": ["cat"],
    }
    check("缺 anchor 不冒充回答",
          evaluate(Corpus(), hits, plan2).decision == "insufficient_evidence")
    paraphrased = dict(
        plan,
        requirements=[{
            "kind": "method",
            "value": "critic-free advantage estimation",
            "required": True,
        }],
    )
    paraphrased_result = evaluate(Corpus(), hits, paraphrased)
    check("方法需求改写不硬拒",
          paraphrased_result.decision == "answer"
          and paraphrased_result.requirement_results[0]["satisfied"] is False,
          paraphrased_result.to_dict())

    versioned_plan = {
        "answer_mode": "paper_fact",
        "intent": "fact",
        "core_concepts": [{
            "name": "Model architecture of the Dreamer series",
            "aliases": ["Model architecture of the Dreamer series"],
            "required": False,
        }],
        "anchors": ["Dreamer", "DreamerV2", "DreamerV3"],
        "requirements": [],
    }
    versioned = evaluate(
        Corpus(), [SimpleNamespace(paper_id="p5")], versioned_plan,
    )
    check("同系列多版本 anchor 允许任一命中",
          versioned.decision == "answer", versioned.to_dict())

    partial_plan = {
        "answer_mode": "paper_fact",
        "intent": "fact",
        "core_concepts": [
            {"name": "visuomotor policy", "aliases": ["visuomotor"], "required": True},
            {"name": "input representation", "aliases": ["observation encoding"], "required": True},
            {"name": "visual observation", "aliases": ["RGB image"], "required": True},
            {"name": "proprioceptive state", "aliases": ["robot state"], "required": True},
        ],
        "anchors": [],
        "requirements": [],
    }
    partial = evaluate(
        Corpus(), [SimpleNamespace(paper_id="p6")], partial_plan,
    )
    check("核心概念覆盖率达到阈值可回答",
          partial.decision == "answer" and partial.required_coverage >= 0.75,
          partial.to_dict())

    optional_only = {
        "answer_mode": "global_trend",
        "intent": "fact",
        "core_terms": ["Dreamer world model"],
        "core_concepts": [{
            "name": "optional topic label",
            "aliases": ["optional topic label"],
            "required": False,
        }],
        "anchors": [],
        "requirements": [],
    }
    check("无可选概念时不被覆盖率误拒",
          evaluate(
              Corpus(), [SimpleNamespace(paper_id="p5")], optional_only,
          ).decision == "answer")

    traffic_plan = {
        "answer_mode": "global_trend",
        "intent": "trend",
        "core_terms": ["traffic signal control", "reinforcement learning"],
        "core_concepts": [
            {"name": "traffic signal control", "aliases": ["TSC"], "required": False},
            {"name": "reinforcement learning", "aliases": ["RL"], "required": False},
        ],
        "anchors": [],
        "requirements": [],
    }
    traffic = evaluate(
        Corpus(),
        [SimpleNamespace(paper_id="p7"), SimpleNamespace(paper_id="p8")],
        traffic_plan,
    )
    check("真实跨域应用有证据时不被领域门误拒",
          traffic.decision == "answer" and len(traffic.domain_support_ids) >= 2,
          traffic.to_dict())

    medical_plan = {
        "answer_mode": "global_trend",
        "intent": "trend",
        "core_terms": ["CT segmentation", "medical imaging"],
        "core_concepts": [{
            "name": "medical imaging",
            "aliases": ["clinical imaging"],
            "required": False,
        }],
        "anchors": [],
        "requirements": [],
    }
    medical = evaluate(
        Corpus(),
        [SimpleNamespace(paper_id="p9"), SimpleNamespace(paper_id="p10")],
        medical_plan,
    )
    check("候选未支持核心领域时硬拒",
          medical.decision == "out_of_scope", medical.to_dict())

    generic_plan = {
        "answer_mode": "global_trend",
        "intent": "trend",
        "core_terms": ["optimization", "control"],
        "core_concepts": [],
        "anchors": [],
        "requirements": [],
    }
    check("泛词不能单独证明域内可答",
          evaluate(
              Corpus(), [SimpleNamespace(paper_id="p4")], generic_plan,
          ).decision == "clarify")

    compare_plan = {
        "answer_mode": "compare",
        "intent": "compare",
        "core_concepts": [
            {
                "name": "DPO",
                "aliases": ["DPO", "Direct Preference Optimization"],
                "required": True,
            },
            {
                "name": "RLOO",
                "aliases": ["RLOO", "REINFORCE leave-one-out"],
                "required": True,
            },
        ],
        "requirements": [
            {"kind": "comparison_side", "value": "DPO", "required": True},
            {"kind": "comparison_side", "value": "RLOO", "required": True},
        ],
        "anchors": [],
    }
    compare_result = evaluate(Corpus(), compare_hits, compare_plan)
    check("对比双方可来自不同论文",
          compare_result.decision == "answer", compare_result.to_dict())
    check("需求矩阵记录两侧证据",
          len(compare_result.requirement_results) == 2
          and all(r["satisfied"] and r["paper_ids"] for r in compare_result.requirement_results),
          compare_result.to_dict())
    metric_missing = dict(
        compare_plan,
        requirements=[
            {"kind": "comparison_side", "value": "DPO", "required": True},
            {"kind": "comparison_side", "value": "RLOO", "required": True},
            {"kind": "metric", "value": "wall-clock efficiency", "required": True},
        ],
    )
    metric_result = evaluate(Corpus(), compare_hits, metric_missing)
    check("对比指标缺失不否决两侧证据",
          metric_result.decision == "answer"
          and metric_result.requirement_results[-1]["satisfied"] is False,
          metric_result.to_dict())

    missing_side = dict(
        compare_plan,
        requirements=[
            {"kind": "comparison_side", "value": "DPO", "required": True},
            {"kind": "comparison_side", "value": "SimPO", "required": True},
        ],
    )
    check("缺一侧对比证据要求澄清",
          evaluate(Corpus(), compare_hits, missing_side).decision == "clarify")

    from hotspot.api.engine import Engine
    cfg = {"api": {"min_answer_chars": 20}}
    check("空/超短答案无效", not Engine._answer_is_valid("太短", cfg))
    check("无引用答案无效", not Engine._answer_is_valid("这是一段足够长但没有引用的答案文本", cfg))
    check("有引用答案有效",
          Engine._answer_is_valid("这是一段足够长并且引用了论文的答案文本 [P:p1]", cfg))
    from hotspot.reasoning import verify
    extractive = verify.parse_claims("证据直接来自论文摘要 [P:p1]")
    check("抽句式降级答案免远端校验",
          Engine._fallback_verdicts(extractive, frozenset({"p1"}))[0].status
          == "supported")
    lexical = verify.parse_claims(
        "RLOO improves sample efficiency [P:p4]"
    )
    check("校验超时可用本地词面兜底",
          Engine._lexical_verdicts(
              lexical, frozenset({"p4"}),
              {"p4": "RLOO improves sample efficiency"},
          )[0].status == "supported")


# ══════════════════════════════════════════════════════════════════════
# 4) 真引擎的 meta 也不该崩（数据缺失 → 降级 + warning）
# ══════════════════════════════════════════════════════════════════════


def check_engine_meta() -> None:
    print("[3] 真引擎 meta（数据缺也要能返回）")
    try:
        meta = Engine().meta().to_dict()
        json.dumps(meta, ensure_ascii=False)
        check("meta 返回", "sources" in meta and "params" in meta)
    except Exception as exc:  # noqa: BLE001
        FAILS.append(f"Engine.meta 抛异常：{type(exc).__name__}: {exc}")
        print(f"  FAIL Engine.meta 抛异常：{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    global QUIET
    parser = argparse.ArgumentParser(description="接口契约与离线冒烟自检")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出失败项与总数")
    args = parser.parse_args(argv)
    QUIET = args.quiet

    checks = (
        check_contract,
        check_search_semantics,
        check_hybrid,
        check_coverage,
        check_query_plan,
        check_plan_consumption,
        check_claim_first,
        check_evidence,
        check_engine_meta,
        check_http,
    )
    sink = io.StringIO() if QUIET else None
    if QUIET:
        with contextlib.redirect_stdout(sink):
            for fn in checks:
                fn()
    else:
        print(f"契约版本 {CONTRACT_VERSION} · 参数 {len(PARAM_SPEC)} 个\n")
        for fn in checks:
            fn()
            print()

    if FAILS:
        print(f"✗ {len(FAILS)} 处失败：")
        for f in FAILS:
            print("   -", f)
        print(f"checks={CHECKS} failures={len(FAILS)}")
        return 1
    print(f"✓ checks={CHECKS} failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
