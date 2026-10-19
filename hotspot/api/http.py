"""HTTP 适配层 —— 让新的 HTML/CSS/JS 前端（或任何语言）直接调引擎。

**零第三方依赖**（只用标准库 `http.server`）：不用装 fastapi/uvicorn 就能跑，
评审现场和队友机器上都不会卡在装包。要换 FastAPI 也只改这个文件，契约不动。

路由只有两条业务接口
--------------------

=====================  ===============================================
`GET  /api/meta`       语料概况 + 参数表 + 能力开关（前端据此生成设置页）
`POST /api/ask`        问一次（新问题/追问/参数/API 配置/引用/图谱/用量）
`GET  /api/health`     探活
`GET  /`（及静态文件）  托管 `web/` 目录（前端页面放这里即可）
=====================  ===============================================

`POST /api/ask` 两种响应，**同一个路由**：

* 默认 `application/json` —— 一次性返回完整结果；
* 请求体带 `"stream": true` —— 返回 `text/event-stream`，先逐条推
  `{"type":"stage",...}` 进度，最后一条是 `event: result` + 完整结果。
  前端用 `fetch` + `ReadableStream` 读（`EventSource` 只支持 GET，不适合）。

跑法见 `scripts/run_server.py`。
"""

from __future__ import annotations

import json
import mimetypes
import posixpath
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from .contract import CONTRACT_VERSION, E_INTERNAL, ApiError, AskRequest
from .engine import Engine

#: 请求体上限（防止有人一次塞几十 MB）。
MAX_BODY = 1 << 20        # 1 MiB

#: 静态文件后缀白名单（不做目录列表，不给源码）。
STATIC_EXT = {".html", ".css", ".js", ".mjs", ".json", ".svg", ".png", ".jpg",
              ".jpeg", ".webp", ".ico", ".woff", ".woff2", ".ttf", ".map"}


class Handler(BaseHTTPRequestHandler):
    """一个 Handler 实例处理一个请求；`server.engine` 是共享的 Engine。"""

    protocol_version = "HTTP/1.1"
    server_version = "HotspotAPI/" + CONTRACT_VERSION

    # ---- 基础工具 ---------------------------------------------------- #

    @property
    def engine(self) -> Engine:
        return self.server.engine          # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:   # noqa: A003
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"contract_version": CONTRACT_VERSION,
                            "error": {"code": code, "message": message}})

    # ---- 路由 -------------------------------------------------------- #

    def do_OPTIONS(self) -> None:                      # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:                         # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:                          # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._json(200, {"ok": True, "contract_version": CONTRACT_VERSION})
            elif path == "/api/meta":
                self._json(200, self.engine.meta().to_dict())
            elif path.startswith("/api/"):
                self._error(404, "not_found", f"没有这个接口：{path}")
            else:
                self._static(path)
        except ApiError as exc:
            self._error(exc.status, exc.code, exc.message)
        except Exception as exc:                       # noqa: BLE001
            self._error(500, E_INTERNAL, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:                         # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/ask":
            self._error(404, "not_found", f"没有这个接口：{path}")
            return
        try:
            raw = self._read_body()
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ApiError("bad_request", "请求体必须是 JSON 对象")
            stream = bool(payload.pop("stream", False))
            req = AskRequest.from_dict(payload)
            if stream:
                self._ask_stream(req)
            else:
                self._json(200, self.engine.ask(req).to_dict())
        except json.JSONDecodeError as exc:
            self._error(400, "bad_request", f"请求体不是合法 JSON：{exc}")
        except ApiError as exc:
            self._error(exc.status, exc.code, exc.message)
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            self._error(500, E_INTERNAL, f"{type(exc).__name__}: {exc}")

    # ---- 内部 -------------------------------------------------------- #

    def _read_body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError("bad_request", "Content-Length 不合法") from None
        if n <= 0:
            return b""
        if n > MAX_BODY:
            raise ApiError("bad_request", f"请求体过大（>{MAX_BODY} 字节）", status=413)
        return self.rfile.read(n)

    def _ask_stream(self, req: AskRequest) -> None:
        """SSE：先推进度，最后推 `event: result`。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        # 没有 Content-Length 的流：明确告知「写完就关」，否则客户端会尝试复用
        # 这条已关闭的连接，Windows 上表现为 ConnectionAborted(10053)
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        self.close_connection = True
        closed = {"v": False}

        def write(chunk: str) -> None:
            if closed["v"]:
                return
            try:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # 前端主动断开（关页面/取消）——别再写，但让引擎自己跑完
                closed["v"] = True

        def on_event(ev: dict) -> None:
            write("data: " + json.dumps(ev, ensure_ascii=False, default=str) + "\n\n")

        write(": connected\n\n")
        try:
            resp = self.engine.ask(req, on_event=on_event)
            write("event: result\ndata: "
                  + json.dumps(resp.to_dict(), ensure_ascii=False, default=str) + "\n\n")
        except ApiError as exc:
            write("event: error\ndata: "
                  + json.dumps({"error": exc.to_dict()}, ensure_ascii=False) + "\n\n")
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            write("event: error\ndata: " + json.dumps(
                {"error": {"code": E_INTERNAL,
                           "message": f"{type(exc).__name__}: {exc}"}},
                ensure_ascii=False) + "\n\n")
        finally:
            # 显式结束（不 close 连接，让 keep-alive 正常回收）
            self.close_connection = True

    def _static(self, path: str) -> None:
        root: Optional[Path] = getattr(self.server, "static_dir", None)  # type: ignore[attr-defined]
        if root is None or not root.is_dir():
            self._json(200, {
                "service": "hotspot-api", "contract_version": CONTRACT_VERSION,
                "hint": "没有 web/ 目录可托管。前端静态文件放到 web/，或用 GET /api/meta 开始。",
                "endpoints": ["GET /api/meta", "POST /api/ask", "GET /api/health"],
            })
            return
        rel = unquote(path).lstrip("/") or "index.html"
        # 目录穿越防护：归一化后必须仍在 root 内
        target = (root / posixpath.normpath(rel)).resolve()
        if root.resolve() not in target.parents and target != root.resolve():
            self._error(403, "forbidden", "非法路径")
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            # SPA 回退：前端路由（/chat、/settings）也能打开
            index = root / "index.html"
            if index.is_file():
                target = index
            else:
                self._error(404, "not_found", f"找不到 {rel}")
                return
        if target.suffix.lower() not in STATIC_EXT:
            self._error(403, "forbidden", "不允许的文件类型")
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",
                                                  "application/json"):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


def create_server(host: str = "127.0.0.1", port: int = 8000, *,
                  engine: Optional[Engine] = None,
                  static_dir: str | Path | None = None,
                  quiet: bool = False) -> ThreadingHTTPServer:
    """建一个可直接 `serve_forever()` 的服务器（测试里也能用，端口给 0 即随机）。"""
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.engine = engine or Engine()                    # type: ignore[attr-defined]
    srv.static_dir = Path(static_dir) if static_dir else None  # type: ignore[attr-defined]
    if quiet:
        Handler.log_message = lambda *a, **k: None      # type: ignore[assignment]
    return srv


def serve(host: str = "127.0.0.1", port: int = 8000, *,
          engine: Optional[Engine] = None,
          static_dir: str | Path | None = None) -> None:
    srv = create_server(host, port, engine=engine, static_dir=static_dir)
    print(f"[api] http://{host}:{srv.server_port}  "
          f"(meta=/api/meta  ask=POST /api/ask  health=/api/health)")
    if static_dir:
        print(f"[api] 静态目录：{Path(static_dir).resolve()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] 收到 Ctrl-C，退出")
    finally:
        srv.server_close()
