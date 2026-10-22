"""Offline golden tests for internal tool contracts and maturity gates.

    python scripts/check_tools.py

不调用 LLM、不加载全库。完整检索质量由 ``eval/retrieval_eval.py`` 单独测试。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hotspot.harness import ToolObservation, ToolSpec  # noqa: E402
from hotspot.harness.tools import (  # noqa: E402
    candidate_filter,
    evidence_gate,
    fetch_papers,
    local_cluster,
)
from hotspot.reasoning.evidence import evaluate  # noqa: E402
from hotspot.reasoning.verify import verify_claims  # noqa: E402
from hotspot.schema import Claim  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail="") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(f"{name}: {detail}")


def check_contract() -> None:
    spec = ToolSpec(
        name="paper_search",
        version="0.1",
        level="macro",
        args_schema={"query": "str", "top_k": "int"},
        returns_schema={"status": "str", "ids": "list[str]"},
        preconditions=("corpus_ready",),
        postconditions=("ids_are_known",),
        cost_model="lexical+dense; no generation",
        deterministic=True,
        fallback="lexical_only",
        owner="algorithm",
    )
    spec.validate()
    check("ToolSpec 可校验", spec.to_dict()["name"] == "paper_search")
    check("工具 contract 不依赖 answer_mode", "answer_mode" not in spec.args_schema)

    obs = ToolObservation(
        status="ok",
        ids=("p1", "p2"),
        ranks=(1, 2),
        scores=(0.9, 0.8),
        counts={"returned": 2},
        flags={"answerable": True},
        data={"query_coverage": 1.0},
    )
    check("Observation 可归一化", obs.to_dict()["ids"] == ["p1", "p2"])
    failed = ToolObservation.failure("index_missing", "index unavailable",
                                     retryable=True)
    check("失败 observation 结构化",
          failed.status == "error" and failed.error.retryable)


class FixtureCorpus:
    def __init__(self, papers: dict[str, str]):
        self.papers = papers

    def get(self, pid):
        if pid not in self.papers:
            return None
        text = self.papers[pid]
        return SimpleNamespace(title=text, abstract="", year=2024)


def check_evidence_fixture() -> None:
    path = ROOT / "eval" / "tool_fixtures" / "evidence_gate.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus = FixtureCorpus(data["papers"])
    for case in data["cases"]:
        result = evaluate(
            corpus,
            [SimpleNamespace(paper_id=pid) for pid in case["hits"]],
            case["plan"],
        )
        check(
            f"evidence:{case['id']}",
            result.decision == case["expected_decision"],
            result.to_dict(),
        )
        observed = evidence_gate(corpus, case["hits"], case["plan"])
        check(
            f"evidence-adapter:{case['id']}",
            observed.flags.get("decision") == case["expected_decision"],
            observed.to_dict(),
        )

    fetched = fetch_papers(corpus, ["p_answer", "missing"])
    check("fetch_papers 保留缺失 ID", fetched.flags.get("partial") is True)
    check("fetch_papers 只返回已知论文", fetched.ids == ("p_answer",))
    filtered = candidate_filter(
        corpus, ["p_answer", "missing", "p_noise"], years=[2025],
    )
    check("candidate_filter 处理未知 ID 与年份",
          filtered.ids == () and filtered.counts["unknown"] == 1,
          filtered.to_dict())


class VerifyClient:
    def ask(self, user, *, system, schema, required, **kwargs):
        supported = (
            "Direct Preference Optimization removes an explicit reward model"
            in user
            and "zero hallucination" not in user
        )
        return SimpleNamespace(
            ok=True,
            data={
                "verdict": "supported" if supported else "unsupported",
                "reason": "fixture rule",
            },
        )


def check_verify_fixture() -> None:
    path = ROOT / "eval" / "tool_fixtures" / "verify_claims.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = [
        Claim(text=item["text"], cited_ids=tuple(item["cited_ids"]))
        for item in data["claims"]
    ]
    verdicts = verify_claims(
        claims,
        allowed_ids=frozenset(data["allowed_ids"]),
        evidence_lookup=data["evidence"],
        client=VerifyClient(),
        batch=False,
    )
    expected = [item["expected_status"] for item in data["claims"]]
    actual = [item.status for item in verdicts]
    check("verify 四态完整", actual == expected, f"{actual} != {expected}")


def check_local_cluster() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(loc=1.0, scale=0.03, size=(40, 16)).astype(np.float32)
    right = rng.normal(loc=-1.0, scale=0.03, size=(40, 16)).astype(np.float32)
    vectors = np.vstack([left, right])
    ids = [f"p{i}" for i in range(len(vectors))]
    observed = local_cluster(vectors, ids, intent="trend")
    repeated = local_cluster(vectors, ids, intent="trend")
    check("local_cluster 不依赖 answer_mode",
          observed.error is None, observed.to_dict())
    check("local_cluster 返回稳定 observation",
          observed.status == "ok"
          and observed.counts.get("clusters", 0) >= 2,
          observed.to_dict())
    check("local_cluster 同输入结果稳定",
          observed.data == repeated.data,
          "cluster assignment changed")


def main() -> int:
    check_contract()
    check_evidence_fixture()
    check_verify_fixture()
    check_local_cluster()
    if FAILS:
        print(f"✗ tool checks={CHECKS} failures={len(FAILS)}")
        for item in FAILS:
            print(" -", item)
        return 1
    print(f"✓ tool checks={CHECKS} failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
