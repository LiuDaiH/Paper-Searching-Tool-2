"""接口自检：契约往返 + HTTP 冒烟（**离线、秒级、确定性**）。

前端改页面、算法改内核之前先跑它：只要它还绿，接口就没被改坏。

    python scripts/check_api.py

查四件事：
  1. 契约往返：`AskRequest.from_dict` → `to_dict` → JSON 可序列化
  2. 参数白名单：未知键/越界/类型错都**必须报错**（不静默忽略）
  3. 响应形状：`AskResponse.to_dict()` 的键齐全且 JSON 可序列化
  4. HTTP 骨架：/api/health、/api/meta、/api/ask（含 SSE 流式）、坏请求 400、未知路由 404

⚠️ 不碰语料、不碰 LLM：用 StubEngine 顶替，所以在任何机器上都能跑绿。
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(f"{name}: {detail}")
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
    })
    d = req.to_dict()
    json.dumps(d, ensure_ascii=False)
    check("字段解析", req.mode == "followup" and req.history[1].citation_ids == ["p1"])
    check("参数归一", req.params == {"max_topics": 4, "streit_garg": False})
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
                "graph", "usage", "warnings", "error"):
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
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode("utf-8")


def _post(url: str, payload=None, raw: bytes | None = None):
    data = raw if raw is not None else json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def check_http() -> None:
    print("[2] HTTP 冒烟")
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
# 3) 真引擎的 meta 也不该崩（数据缺失 → 降级 + warning）
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


def main() -> int:
    print(f"契约版本 {CONTRACT_VERSION} · 参数 {len(PARAM_SPEC)} 个\n")
    check_contract()
    print()
    check_engine_meta()
    print()
    check_http()
    print()
    if FAILS:
        print(f"✗ {len(FAILS)} 处失败：")
        for f in FAILS:
            print("   -", f)
        return 1
    print("✓ 接口自检全绿（契约 / 参数 / HTTP / SSE）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
