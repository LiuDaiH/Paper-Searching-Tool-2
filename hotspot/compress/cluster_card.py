"""L3 压缩表示层 —— **每簇一张 ClusterCard（★ 唯一花钱处，方案 §3.3）**。

输入：簇成员的 **title + abstract**（不是正文——正文含未展 LaTeX 宏、84% 吞词）。
输出：一张结构化的 :class:`ClusterCard`。

分工（防幻觉第一道闸，方案 §3.3）：
- `scale` / `venue`（跨会三态）**从 L2 的 metric 直接抄**，不让 LLM 判断
- LLM 只填 `label` / `core_problems` / `dominant_methods` / `open_questions` / `evidence_ids`
- `evidence_ids` 必须**都在实际放进输入的成员里**（validator 强制）

断点续跑：已落盘的卡片不重算；LLM 两次非法 JSON ⇒ 降级（`degraded_reason`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import hotspot
from ..config import load_config, paths_from
from ..llm import make_client
from ..schema import ClusterCard, SpreadState, VenueProfile
from dasc.llm.client import MAX_INPUT_CHARS  # noqa: E402 —— 出境闸

__all__ = ["CARD_SCHEMA", "gen_card", "make_cards", "load_cluster_members"]


#: LLM 输出契约（JSON mode）。`evidence_ids` 由系统自选（`_pick_evidence`），不由 LLM 返回。
CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "core_problems": {"type": "array", "items": {"type": "string"}},
        "dominant_methods": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label", "core_problems", "dominant_methods", "open_questions"],
}

_SYSTEM = (
    "你是研究主题命名器。给一簇论文起名并概括。只依据下面给出的论文"
    "（id + 标题 + 摘要）作判断。输出必须用 JSON：label(一句话主题名)、"
    "core_problems(3-5 个核心问题)、dominant_methods(3-5 个主流方法)、"
    "open_questions(0-3 个未解问题)。只概括事实，不用形容词堆砌。"
)


def load_cluster_members(cfg=None) -> dict[str, dict]:
    """读 `workspace/metrics/cluster_members.json`：`{cluster_id: {"members": [...]}}`。"""
    cfg = cfg or load_config()
    ws = paths_from(cfg)["workspace"]
    p = ws / "metrics" / "cluster_members.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在——先跑 L2 聚类（cluster.py）")
    return json.loads(p.read_text(encoding="utf-8"))


def _member_prompt(members: list[str], corpus, *, max_chars: int) -> tuple[list[str], str]:
    """把成员摘要拼进提示词，返回 `(实际进入输入的成员 id 列表, prompt 正文)`。

    一旦累计超过 `max_chars` 就停在**这一个成员之前**——保证 `evidence_ids`
    只能来自真正发给 LLM 的成员（validator 据此校验）。
    """
    lines: list[str] = []
    used: list[str] = []
    total = 0
    for pid in members:
        p = corpus.get(pid)
        if p is None:
            continue
        block = f"[{pid}] {p.title}\n{p.abstract or ''}"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        used.append(pid)
        total += len(block)
    return used, "\n\n".join(lines)


def _validator(allowed: frozenset[str]):
    def _v(data: dict, raw: str) -> None:
        ev = data.get("evidence_ids") or []
        if not ev:
            raise ValueError("evidence_ids 为空")
        bad = [e for e in ev if e not in allowed]
        if bad:
            raise ValueError(f"evidence_ids 引用了不在输入里的 id：{bad[:5]}")
    return _v


def _pick_evidence(used: list[str], metric: dict | None, *,
                   min_n: int = 3, max_n: int = 6) -> tuple[str, ...]:
    """从实际输入 `used` 里选簇的 representative（质心最近）+ 按序补足，作为卡片证据。

    全部 ∈ `used`（实际发过给 LLM 的成员），保证溯源合法；**不由 LLM 选择**，杜绝编造。
    """
    picked = [i for i in ((metric or {}).get("representative_ids") or ()) if i in used]
    for i in used:
        if len(picked) >= max_n:
            break
        if i not in picked:
            picked.append(i)
    if len(picked) < min_n:
        picked = list(used[:max_n])
    return tuple(picked)


def _metric_of(metric: dict) -> tuple[int, int, VenueProfile]:
    """从 L2 的 metric dict 还原客观字段（scale / venue）。

    ⚠️ `ClusterMetric.to_dict()` 用 `size` 而不是 `scale`（2026-09-27 修）。
    """
    venue = VenueProfile.from_dict(metric.get("venue") or {})
    return int(metric.get("size") or metric.get("scale") or 0), venue


def gen_card(
    client,
    cluster_id: int,
    members: list[str],
    corpus,
    *,
    metric: dict | None = None,
    card_dir: Path | None = None,
    max_chars: int = MAX_INPUT_CHARS - 800,   # 给 system + 结构提示留余量
) -> tuple[ClusterCard, bool]:
    """生成一张卡。`metric`（dict，来自 L2）提供 scale/venue；断点续跑跳过。

    返回 `(card, from_cache)`。
    """
    if card_dir is not None and (card_dir / f"card_{cluster_id}.json").exists():
        d = json.loads((card_dir / f"card_{cluster_id}.json").read_text(encoding="utf-8"))
        return ClusterCard.from_dict(d), True

    scale, venue = _metric_of(metric) if metric else (0, VenueProfile())
    used, body = _member_prompt(members, corpus, max_chars=max_chars)
    if not used:
        return ClusterCard(cluster_id=cluster_id, scale=scale, venue=venue,
                           label="", degraded_reason="没有可入输入的成员摘要"), False

    try:
        call = client.ask(
            body,
            system=_SYSTEM,
            schema=CARD_SCHEMA,
            required=("label",),
        )
    except Exception as e:  # noqa: BLE001 —— 调用失败/非法 JSON：降级不崩
        return ClusterCard(cluster_id=cluster_id, scale=scale, venue=venue,
                           label="",
                           degraded_reason=f"LLM 未产出合规 JSON（{type(e).__name__}）"), False
    data = call.data if call.ok else {}
    card = ClusterCard(
        cluster_id=cluster_id,
        scale=scale,
        venue=venue,
        label=str(data.get("label") or ""),
        core_problems=tuple(data.get("core_problems") or ()),
        dominant_methods=tuple(data.get("dominant_methods") or ()),
        open_questions=tuple(data.get("open_questions") or ()),
        evidence_ids=_pick_evidence(used, metric),
        degraded_reason="" if call.ok else "两次非法 JSON",
    )
    return card, False


def save_card(card: ClusterCard, card_dir: Path) -> Path:
    card_dir.mkdir(parents=True, exist_ok=True)
    out = card_dir / f"card_{card.cluster_id}.json"
    out.write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def make_cards(corpus, *, cfg=None, dry_run=False, max_clusters: int | None = None) -> dict:
    """对全部簇生成卡片到 `workspace/cards/`。返回进度报告。"""
    cfg = cfg or load_config()
    ws = paths_from(cfg)["workspace"]
    members_map = load_cluster_members(cfg)
    card_dir = ws / "cards"
    card_dir.mkdir(parents=True, exist_ok=True)

    # 读 L2 的 metrics 以便抄客观字段
    metrics_p = ws / "metrics" / "metrics.json"
    metric_by_id: dict[int, dict] = {}
    if metrics_p.exists():
        for m in (json.loads(metrics_p.read_text(encoding="utf-8")) or {}).get("metrics", []):
            metric_by_id[int(m["cluster_id"])] = m

    client = make_client(cfg, dry_run=dry_run)
    n_new = n_skip = n_degraded = 0
    out: list[ClusterCard] = []
    cids = [int(c) for c in members_map]
    if max_clusters is not None:
        cids = sorted(cids)[:max_clusters]

    for cid in cids:
        raw = members_map.get(str(cid)) or []
        members = raw if isinstance(raw, list) else (raw.get("members") or [])
        if not members:
            continue
        card, cached = gen_card(client, cid, members, corpus,
                                metric=metric_by_id.get(cid), card_dir=card_dir)
        if cached:
            n_skip += 1
        else:
            save_card(card, card_dir)
            n_new += 1
            if card.degraded_reason:
                n_degraded += 1
        out.append(card)
    return {
        "clusters": len(cids), "new": n_new, "skipped": n_skip,
        "degraded": n_degraded, "out": str(card_dir),
    }
