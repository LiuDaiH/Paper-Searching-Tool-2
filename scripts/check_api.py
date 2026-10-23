"""Offline checks for the stable public API shell and HTTP transport.

    python scripts/check_api.py --quiet

Algorithm v2 is intentionally not tested here until its tools are introduced.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hotspot.api.contract import (  # noqa: E402
    CONTRACT_VERSION,
    PARAM_SPEC,
    AskRequest,
    AskResponse,
    MetaResponse,
    SourceInfo,
    new_request_id,
)
from hotspot.api.engine import Engine  # noqa: E402
from hotspot.api.http import create_server  # noqa: E402

FAILS: list[str] = []
CHECKS = 0
QUIET = False


def check(name: str, cond: bool, detail="") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        if not QUIET:
            print(f"  ok  {name}")
        return
    FAILS.append(f"{name}: {detail}")
    if not QUIET:
        print(f"  FAIL {name}  {detail}")


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _post(url: str, payload=None, *, raw: bytes | None = None,
          headers: dict | None = None) -> tuple[int, str, dict]:
    data = raw if raw is not None else json.dumps(
        payload or {}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8"), dict(exc.headers)


class StubEngine:
    def meta(self) -> MetaResponse:
        return MetaResponse(
            sources=[SourceInfo(key="meeting", label="会议", papers=1)],
            params=[p.to_dict({}) for p in PARAM_SPEC],
            capabilities={"stream": True, "algorithm_v2_rebuilding": True},
        )

    def ask(self, request, on_event=None) -> AskResponse:
        req = request if isinstance(request, AskRequest) else AskRequest.from_dict(request)
        return AskResponse(
            request_id=new_request_id(),
            source=req.source,
            mode=req.mode,
            route="local",
            text="stub",
            meta={"stub": True},
        )


def check_contract() -> None:
    print("[1] contract")
    req = AskRequest.from_dict({
        "question": "测试问题",
        "source": "meeting",
        "mode": "followup",
        "history": [
            {"role": "user", "text": "上一问"},
            {"role": "assistant", "text": "上一答", "citation_ids": ["p1"]},
        ],
        "llm": {"base_url": "https://x/v1", "model": "m", "api_key": "secret"},
        "params": {"search_limit": 16},
        "scope": {"years": [2025], "years_explicit": True},
    })
    dumped = req.to_dict()
    check("request roundtrip", dumped["mode"] == "followup")
    check("api key hidden", dumped["llm"]["api_key"] == "***")
    check("scope retained", dumped["scope"]["years"] == [2025])
    check("params normalized", dumped["params"] == {"search_limit": 16})

    resp = StubEngine().ask(req)
    body = resp.to_dict()
    expected = {
        "contract_version", "request_id", "source", "mode", "route",
        "refused", "refuse_reason", "answer", "topics", "graph",
        "usage", "warnings", "error", "meta",
    }
    check("response keys", expected <= set(body))
    check("response json", isinstance(json.dumps(body, ensure_ascii=False), str))
    check("contract version", body["contract_version"] == CONTRACT_VERSION)


def check_http() -> None:
    print("[2] HTTP")
    server = create_server("127.0.0.1", 0, engine=StubEngine(), quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, text = _get(base + "/api/health")
        check("health", status == 200 and json.loads(text)["ok"] is True)

        status, text = _get(base + "/api/meta")
        meta = json.loads(text)
        check("meta", status == 200 and meta["contract_version"] == CONTRACT_VERSION)
        check("params drive settings", len(meta["params"]) == len(PARAM_SPEC))

        status, text, _ = _post(base + "/api/ask", {"question": "测试问题"})
        body = json.loads(text)
        check("ask", status == 200 and body["answer"]["text"] == "stub")

        status, text, _ = _post(
            base + "/api/ask", {"question": "x", "params": {"nope": 1}},
        )
        check("unknown param rejected",
              status == 400 and json.loads(text)["error"]["code"] == "bad_request")

        status, _, _ = _post(base + "/api/ask", raw=b"{bad json")
        check("bad json rejected", status == 400)

        status, _, _ = _post(base + "/api/nope", {})
        check("unknown route", status == 404)

        status, text, headers = _post(
            base + "/api/ask",
            {"question": "测试", "stream": True},
        )
        check("sse content type",
              status == 200 and "text/event-stream" in headers.get("Content-Type", ""))
        check("sse final event", "event: done" in text or '"route": "local"' in text)
    finally:
        server.shutdown()
        server.server_close()


def check_real_meta() -> None:
    print("[3] real engine shell")
    meta = Engine().meta().to_dict()
    check("real meta", bool(meta.get("sources")))
    check("algorithm v2", meta["capabilities"].get("algorithm_version") == "2")
    check("v2 params", {
        "search_limit", "fetch_limit", "max_claims",
        "agent_request_limit", "agent_tool_limit",
    } == {p["key"] for p in meta.get("params", [])})
    arxiv = next(
        (item for item in meta.get("sources", []) if item.get("key") == "arxiv"),
        {},
    )
    check("arXiv index unavailable is explicit",
          "索引未构建" in str(arxiv.get("note") or ""))


def main(argv: list[str] | None = None) -> int:
    global QUIET
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    QUIET = args.quiet
    sink = io.StringIO() if QUIET else None
    if QUIET:
        with contextlib.redirect_stdout(sink):
            check_contract()
            check_http()
            check_real_meta()
    else:
        check_contract()
        check_http()
        check_real_meta()
    if FAILS:
        print(f"✗ checks={CHECKS} failures={len(FAILS)}")
        for item in FAILS:
            print(" -", item)
        return 1
    print(f"✓ checks={CHECKS} failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
