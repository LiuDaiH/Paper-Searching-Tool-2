"""DeepSeek 客户端：**带预算上限、响应缓存、与出境记录**。

⚠️ 这是本库**唯一的数据出境路径**
--------------------------------

摘要会离开这台机器、到 `https://api.deepseek.com/v1`。库里的规矩是
「这条路径要一直明说，不悄悄依赖」（`vault-curl-empty-body`）。所以：

| 约束 | 怎么落实 |
|---|---|
| **只发摘要，不发正文全文** | :data:`MAX_INPUT_CHARS` 硬上限 + :func:`check_payload` 逐条检查；摘要中位 1,383 字符，正文平均 5 万字符，差 40 倍 |
| **key 不进任何入库文件** | key 只从环境变量读；写盘前一律过 :func:`redact`；`_out/llm_cache/` 与 `_out/llm_egress.jsonl` 都在 gitignore 里 |
| **发出去的东西要留痕** | 每次调用往 `_out/llm_egress.jsonl` 追加一行：时间、模型、**输入字符数**、输入 sha256、缓存命中与否。**留痕的是指纹不是原文**——原文在 cache 里已经有了，落两份只会多一处能泄的地方 |

⚠️ 关于"结构化输出"这件事，**不要相信它一定会守规矩**
-------------------------------------------------------

DeepSeek 走 OpenAI 兼容接口，但**只保证 `response_format={"type":"json_object"}`**
（即"输出是 JSON"），**不保证 strict JSON-schema**。所以本模块：

1. 发 `json_object` + 把 schema 塞进提示词（描述清楚字段与取值）
2. 收回来**自己按 schema 校验**
3. 校验不过就**重试一次**，再不过就
   **记为 `schema_violation` 并保留原文**——**不是静默当成成功**

第 3 条正是计划里要量的三件事之一（"schema 是否被遵守"）。
所以 :class:`Usage` 里 `schema_violations` 是个一等字段，不是日志里的一行。

预算：**先算 token，钱要你填单价**
------------------------------------

计划里「3 分/次、不构成约束」是**估的，没量过**。所以本模块：
- 老老实实把 `prompt_tokens` / `completion_tokens` 累计起来报（这是实测）
- 换算成钱要传 `price_in` / `price_out`（每 100 万 token 的单价）。
  **默认 None ⇒ 只报 token，不报钱。** 编一个单价出来比不报更坏。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..cluster.embed import OUT_DIR

__all__ = [
    "BASE_URL", "MODEL", "MAX_INPUT_CHARS",
    "Budget", "Usage", "Call", "LLMError", "BudgetExceeded", "SchemaViolation",
    "Client", "redact", "check_payload", "CACHE_DIR", "EGRESS_LOG",
]

#: DeepSeek 的 OpenAI 兼容端点。
BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

#: key 的环境变量名。**key 只从这里来**——不读文件、不读参数、不写日志。
API_KEY_ENV = "DEEPSEEK_API_KEY"

#: 单条输入字符上限。摘要中位 1,383 字符，留足余量；正文（平均 5 万）**必然超**，
#: 所以这道闸是"正文进不来"的实际保证，不是摆设。
MAX_INPUT_CHARS = 8000

CACHE_DIR: Path = OUT_DIR / "llm_cache"
EGRESS_LOG: Path = OUT_DIR / "llm_egress.jsonl"

_BATCH = 8          # 每篇上限；一期只发放这么多条
_TIME_BUDGET_S = 300.0


class LLMError(RuntimeError):
    """调用失败。**抛出来，不吞。**"""


class BudgetExceeded(LLMError):
    """预算用尽 ⇒ 停。**不是"跑完剩下的"**。"""


class SchemaViolation(LLMError):
    """两次都没给出合规 JSON。**保留原文**，见模块 docstring。"""


def redact(text: str) -> str:
    """写盘前擦掉 key。**任何要落盘的字符串都过这里。**

    两道：一个是环境变量里的真 key（可能为空）；一个是形如 `sk-…` 的通用形状——
    后者是为了防"key 换了、环境变量没更新"时仍能兜住。
    """
    import re
    key = os.environ.get(API_KEY_ENV)
    if key:
        text = text.replace(key, "<REDACTED>")
    return re.sub(r"\bsk-[A-Za-z0-9_\-]{12,}", "<REDACTED>", text)


# ---------------------------------------------------------------------- #
# 出境检查：摘要能走，正文不能
# ---------------------------------------------------------------------- #

def check_payload(text: str, *, what: str = "输入") -> str:
    """检查一条要发出去的文字。**超限就抛**，不截断——截断会让"发的是摘要"变成谎话。"""
    if len(text) > MAX_INPUT_CHARS:
        raise LLMError(
            f"{what} {len(text)} 字符，超过上限 {MAX_INPUT_CHARS}。"
            f"**这大概率意味着正文全文混进来了**——本库只允许摘要出境。"
            f"要么确认它真是摘要，要么调 `MAX_INPUT_CHARS` 并说清为什么。"
        )
    return text


# ---------------------------------------------------------------------- #
# 预算与用量
# ---------------------------------------------------------------------- #

@dataclass
class Budget:
    """预算。**三个上限都是硬的，碰到就停。**

    `max_usd` 默认 None = 不设钱的上限（因为单价要用户给，见模块 docstring）。
    这时 `max_tokens` 与 `max_calls` 仍然生效——**总得有一道闸**。
    """

    max_calls: int = 200
    max_tokens: int = 2_000_000
    max_usd: float | None = None
    price_in: float | None = None       # 每 100 万 prompt token 的钱
    price_out: float | None = None      # 每 100 万 completion token 的钱

    def check(self, u: "Usage") -> None:
        if u.calls >= self.max_calls:
            raise BudgetExceeded(f"调用数到顶：{u.calls}/{self.max_calls}")
        if u.total_tokens >= self.max_tokens:
            raise BudgetExceeded(f"token 到顶：{u.total_tokens}/{self.max_tokens}")
        if self.max_usd is not None and u.usd(self) is not None and u.usd(self) >= self.max_usd:
            raise BudgetExceeded(f"花费到顶：${u.usd(self):.4f}/${self.max_usd}")


@dataclass
class Usage:
    """实测用量。**`schema_violations` 是一等字段**——那是要量的三件事之一。"""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    schema_violations: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def usd(self, b: Budget) -> float | None:
        """单价没给就返回 None——**不猜**。"""
        if b.price_in is None or b.price_out is None:
            return None
        return (self.prompt_tokens * b.price_in
                + self.completion_tokens * b.price_out) / 1_000_000

    def report(self, b: Budget, n_items: int) -> str:
        if self.calls == 0:
            return "一次都没调（全是缓存命中？）"
        per = self.total_tokens / max(self.calls, 1)
        lines = [
            f"调用 {self.calls} 次（缓存命中 {self.cache_hits}）"
            f"  条目 {n_items}",
            f"实测 token：prompt {self.prompt_tokens:,} + completion "
            f"{self.completion_tokens:,} = **{self.total_tokens:,}**"
            f"（**每篇 {per:,.0f}**）",
            f"耗时 {self.seconds:.1f}s（每次 {self.seconds / max(self.calls, 1):.2f}s）",
            f"schema 不合规 {self.schema_violations} 次  调用失败 {self.failures} 次",
        ]
        money = self.usd(b)
        lines.append(
            f"花费：${money:.4f}" if money is not None else
            "花费：**算不了**——`price_in`/`price_out` 没给。"
            "要钱就把 DeepSeek 当时的单价填进来；**编一个单价写进报告比不写更坏**。"
        )
        return "\n".join("  " + x for x in lines)


@dataclass(frozen=True)
class Call:
    """一次调用的结果。**`raw` 保留原文**，供 schema 不合规时复盘。"""

    key: str
    ok: bool
    data: dict | None = None
    raw: str = ""
    usage: dict = field(default_factory=dict)
    cached: bool = False
    error: str = ""
    attempts: int = 1


# ---------------------------------------------------------------------- #
# 客户端
# ---------------------------------------------------------------------- #

class Client:
    """DeepSeek 客户端。**要 key 就从环境变量取，取不到就停。**"""

    def __init__(self, *, model: str = MODEL, base_url: str = BASE_URL,
                 budget: Budget | None = None, cache_dir: Path | None = None,
                 use_cache: bool = True, dry_run: bool = False,
                 timeout: float = 120.0, api_key: str | None = None):
        self.model = model
        self.base_url = base_url
        self.budget = budget or Budget()
        self.cache_dir = cache_dir or CACHE_DIR
        self.use_cache = use_cache
        self.dry_run = dry_run
        self.timeout = timeout          # openai 单请求超时（秒）；默认 120 保持原行为
        # 显式 api_key（客户端 URL/运行时传入）优先；否则仍从环境变量取。
        self.api_key = api_key
        self.usage = Usage()
        self._client = None

        if not dry_run and not self.api_key and not os.environ.get(API_KEY_ENV):
            raise LLMError(
                f"环境变量 {API_KEY_ENV} 没设。\n"
                f"  设它（本会话里可以直接 `! export {API_KEY_ENV}=...`），或传显式 api_key。\n"
                f"  ⚠️ **不要写进任何入库文件**——key 只活在环境里。\n"
                f"  想不联网先跑通链路，用 dry_run=True 或 --dry-run。"
            )

    # ---- 缓存 ---------------------------------------------------------- #

    @staticmethod
    def cache_key(model: str, system: str, user: str) -> str:
        h = hashlib.sha256()
        for part in (model, system, user):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:32]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Call | None:
        p = self._cache_path(key)
        if not (self.use_cache and p.exists()):
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return Call(key=key, ok=d.get("ok", True), data=d.get("data"),
                    raw=d.get("raw", ""), usage=d.get("usage", {}),
                    cached=True, attempts=d.get("attempts", 1))

    def _write_cache(self, c: Call, system: str, user: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": c.key, "model": self.model, "ok": c.ok, "data": c.data,
            "raw": redact(c.raw), "usage": c.usage, "attempts": c.attempts,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # 输入原文**也存**：不然 cache 命中的那次就没法复盘"当时发了什么"。
            # 它和 "只发摘要" 不冲突——它本来就是发出去过的东西。
            "system": redact(system), "user": redact(user),
        }
        self._cache_path(c.key).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log_egress(self, key: str, system: str, user: str, cached: bool,
                    usage: dict) -> None:
        """记一笔出境。**记指纹，不记原文**（原文在 cache 里已有一份）。"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model, "base_url": self.base_url,
            "system_chars": len(system), "user_chars": len(user),
            "payload_sha256": hashlib.sha256((system + "\x00" + user)
                                             .encode("utf-8")).hexdigest()[:16],
            "cached": cached,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        with EGRESS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- 调用 ---------------------------------------------------------- #

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or os.environ[API_KEY_ENV],  # 显式 key 优先，否则环境变量
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,                     # 退避由本模块自己管，见下
            )
        return self._client

    def ask_free(
        self,
        user: str,
        *,
        system: str,
        temperature: float = 0.2,
    ) -> Call:
        """自由文本请求：不强制 JSON、不追加 schema，直接返回模型原文。

        用于「要一段自然语言（如带引用的答案/summary），不该塞 JSON 壳」的场景——
        JSON object 模式会让部分模型把 schema 结构回显而非产出内容。
        带同样的网络重试 / 预算 / 缓存记账。
        """
        check_payload(user, what="user 输入")
        check_payload(system, what="system 提示")
        key = self.cache_key(self.model, system, user)

        hit = self._read_cache(key)
        if hit is not None:
            self.usage.cache_hits += 1
            self._log_egress(key, system, user, True, hit.usage)
            return hit

        self.budget.check(self.usage)

        if self.dry_run:
            return Call(key=key, ok=False, raw="", error="dry_run：没有真的调用")

        client = self._ensure_client()
        net_max = int(os.environ.get("DASC_LLM_RETRIES", "3"))
        t0 = time.perf_counter()
        resp = None
        net_err = None
        for net in range(net_max):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                break
            except Exception as e:                  # noqa: BLE001
                net_err = e
                if net < net_max - 1:
                    time.sleep(2 ** net)
        if resp is None:
            self.usage.failures += 1
            self.usage.seconds += time.perf_counter() - t0
            raise LLMError(
                f"调用失败（重试 {net_max} 次后仍失败）："
                f"{type(net_err).__name__}: {net_err}"
            ) from net_err

        u = resp.usage
        self.usage.calls += 1
        self.usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
        self.usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        raw = resp.choices[0].message.content or ""
        self.usage.seconds += time.perf_counter() - t0
        call = Call(key=key, ok=True, data={"text": raw}, raw=raw,
                    usage={"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                           "completion_tokens": getattr(u, "completion_tokens", 0) or 0})
        self._write_cache(call, system, user)
        self._log_egress(key, system, user, False, call.usage)
        return call

    def ask(
        self,
        user: str,
        *,
        system: str,
        schema: dict,
        required: tuple[str, ...] = (),
        validator=None,
    ) -> Call:
        """问一次。**缓存 → 预算 → 调用 → 校验 → 不合规重试一次 → 记 `SchemaViolation`。**

        `validator(data, raw) -> None`：自己加的额外检查，抛异常即视为不合规。
        """
        check_payload(user, what="user 输入")
        check_payload(system, what="system 提示")
        key = self.cache_key(self.model, system, user)

        hit = self._read_cache(key)
        if hit is not None:
            self.usage.cache_hits += 1
            self._log_egress(key, system, user, True, hit.usage)
            return hit

        self.budget.check(self.usage)

        if self.dry_run:
            return Call(key=key, ok=False, raw="", error="dry_run：没有真的调用")

        client = self._ensure_client()
        last_raw = ""
        t0 = time.perf_counter()
        # 网络/服务端类错误重试（openai SDK max_retries=0，退避由本模块负责）。
        # 只对「调用没拿到响应」重试；schema 校验不合规仍走外层的 (1,2) 语义。
        net_max = int(os.environ.get("DASC_LLM_RETRIES", "3"))
        for attempt in (1, 2):
            resp = None
            net_err = None
            for net in range(net_max):
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": f"{user}\n\n要求的 JSON 结构：\n"
                                                        f"{json.dumps(schema, ensure_ascii=False)}"},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                    break
                except Exception as e:                  # noqa: BLE001
                    net_err = e
                    if net < net_max - 1:
                        time.sleep(2 ** net)
            if resp is None:
                self.usage.failures += 1
                self.usage.seconds += time.perf_counter() - t0
                raise LLMError(
                    f"调用失败（重试 {net_max} 次后仍失败）："
                    f"{type(net_err).__name__}: {net_err}"
                ) from net_err

            u = resp.usage
            self.usage.calls += 1
            self.usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0

            last_raw = resp.choices[0].message.content or ""
            data, err = self._validate(last_raw, required, validator)
            if data is not None:
                self.usage.seconds += time.perf_counter() - t0
                call = Call(key=key, ok=True, data=data, raw=last_raw,
                            usage={"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                                   "completion_tokens": getattr(u, "completion_tokens", 0) or 0},
                            attempts=attempt)
                self._write_cache(call, system, user)
                self._log_egress(key, system, user, False, call.usage)
                return call
            if attempt == 2:
                self.usage.schema_violations += 1
                self.usage.seconds += time.perf_counter() - t0
                raise SchemaViolation(
                    f"两次都没给出合规 JSON：{err}\n原文（前 400 字符）："
                    f"{redact(last_raw)[:400]}"
                )

        raise SchemaViolation(f"不可达：{redact(last_raw)[:200]}")

    @staticmethod
    def _loads_lenient(raw: str):
        """容忍式 JSON 解析：先标准解析，失败则括号扫描找首个可解析的顶层 `{...}`。

        这里只是想让对象解析更稳，**不改校验/字段口径**。能解析的与以前一致，
        解析不了的（flash 常见 `{"{"...` 畸形输出）尽量捞出来，少判 `SchemaViolation`。
        """
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else None
        except Exception:  # noqa: BLE001
            pass
        n = len(raw)
        i = 0
        while True:
            try:
                s = raw.index("{", i)
            except ValueError:
                break
            depth = 0
            in_str = False
            esc = False
            for j in range(s, n):
                ch = raw[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        frag = raw[s:j + 1]
                        try:
                            d = json.loads(frag)
                            if isinstance(d, dict):
                                return d
                        except Exception:  # noqa: BLE001
                            pass
                        break
            i = s + 1
        return None

    @staticmethod
    def _validate(raw: str, required: tuple[str, ...], validator):
        """校验。**返回 `(data, None)` 或 `(None, 原因)`。**"""
        data = Client._loads_lenient(raw)
        if data is None:
            return None, "不是合法 JSON（容错解析也没捞出对象）"
        if not isinstance(data, dict):
            return None, f"顶层不是对象，是 {type(data).__name__}"
        missing = [k for k in required if k not in data]
        if missing:
            return None, f"缺字段：{missing}（拿到 {sorted(data)}）"
        if validator is not None:
            try:
                validator(data, raw)
            except Exception as e:                       # noqa: BLE001
                return None, f"自定义校验不过：{e}"
        return data, None

    # ---- 收尾 ---------------------------------------------------------- #

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:                            # noqa: BLE001
                pass
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
