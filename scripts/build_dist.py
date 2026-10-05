"""服务端源码打包 —— 打 `发布/服务端源码-<日期>.zip`（可重复）。

内容：核心源码（hotspot/、scripts/）+ config.yaml + requirements.txt +
入口/说明文档。排除：workspace（数据产物，可重建）、日志、.git、.zhanlu、
__pycache__、*.pyc、.playwright-mcp、debug.log、临时测试脚本。
"""

from __future__ import annotations

import datetime
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "发布"

# 顶层条目（目录走整树拷贝，但排除其中不良子目录/文件后缀）
ITEMS = [
    "hotspot", "scripts",
    "config.yaml", "requirements.txt",
    "README.md", "AGENTS.md",
    "热点挖掘系统-方案设计.md",
    "测试计划-大创接入评测.md",
    "规划-接入arXiv周报与发布.md",
    "规划-周报接入大创算法.md",
]
EXCLUDE_DIRS = {"__pycache__", ".git", ".zhanlu", ".playwright-mcp", "workspace", "日志", "发布", "eval"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}

RUN_GUIDE = """# 发布运行说明（服务端 + 远程客户端）

> 架构：**单体 Streamlit 服务端（你自己跑） + 浏览器客户端（别人远程用，零安装）**。
> 客户端 = 浏览器，已天然分离：你启动 8502 并做内网穿透，别人输入公网地址即用。
>
> 本包是**服务端（引擎 + UI 进程）源码**；**不**含客户端安装包（客户端就是浏览器）。

## 0. 部署形态
```
[你的机器]  streamlit run scripts/run_ui.py --server.port 8502
      │  （你持有：引擎 + 语料 + 向量 + API key + 模型权重）
      │  内网穿透：frp / ngrok / 云上转发（暴露 8502）
      ▼
[别人浏览器]  http://<你的公网地址>:8502   （零安装，直接使用）
```
- 不需要改任何代码；穿透后多数网络延迟来自公网转发。
- 生产建议给穿透域名配 HTTPS（简易 http 亦可）。

## 1. 启动
```bash
pip install -r requirements.txt
streamlit run scripts/run_ui.py --server.port 8502
```
浏览器打开 http://127.0.0.1:8502 ，左侧切换「会议 24+25 / arXiv 周报」数据源。
> 运行环境建议 Python 3.10+；needs `gymna_env` 或等价环境含 numpy/scikit-learn/sentence-transformers。

## 2. A 类依赖（必就位；不在源码包内，需你本地放置）
| 依赖 | 位置 | 用途 |
|---|---|---|
| 会议语料 24+25 | `G:\\rl_data24\\资源\\papers.jsonl`、`G:\\rl_data\\资源\\papers.jsonl` | 会议库（22,747 篇）|
| arXiv 周报语料 | `D:\\大学编程学习\\arXiv周报\\数据\\papers.jsonl` | arXiv 独立源（4,369 篇）|
| bge-m3 权重 | `D:\\hf_cache`（HF_HOME，~2.3GB）| embedding（离线）|
| 大创 dasc 库 | `D:\\大学编程学习\\obsidian\\dasc`（+ `工具\\md2vectors.py`）| 聚类/MPS/rlRAG/LLM client |
| API key | 环境变量 `Mobile_trash_API_KEY` 等（`config.yaml` 的 `*.api_key_env` 指定变量名）| LLM，**永不落盘** |

> 若在某机器没有 G 盘 24+25 或 arXiv 周报，可在 `config.yaml` 的 `corpus.sources` /
> `corpus.arxiv_source` 指到存在的语料；只有会议库缺也没关系——检索会用现有的源。

## 3. B 类数据（workspace/ 产物：可重建，不随包分发）
| 产物 | 路径 | 重建命令 |
|---|---|---|
| 会议向量 | `workspace/vectors/` | `python -m hotspot.structure.embed` |
| arXiv 向量+打包 | `workspace/arxiv_vectors/` + `exports/weekly_arxiv/` | `python scripts/export_arxiv_weekly.py` |
| arXiv rl 数据 | `workspace/arxiv_rl/` | `scripts/prep_arxiv_rl.py` + `export_future_rf_arxiv.py` |
| 会议 rlRAG/MPS | `dasc/_out/rl_prep/`、`dasc/_out/rlr/future_rf.pkl` | 大创侧导出（已有）|
| 引用图 | `workspace/depgraph/` | `scripts/fetch_arxiv_refs.py` |

> 全部断点/分片续跑，重建不丢进度。别人端**不碰**任何 A/B 数据。

## 4. 每周增量（arXiv 周报；别人检索到最新）
```bash
python scripts/update_arxiv_weekly.py --init   # 首次(跳过历史，建水位)
python scripts/update_arxiv_weekly.py          # 之后每周(只处理新增，幂等)
```
> 可用 `automation`/cron 定时跑；跑完自动更新 `weekly_arxiv/` 发布物。

## 5. 常见问题
- **sklearn 版本警告**（`unpickle RF ... 1.7.2 vs 1.6.1`）：`future_rf.pkl` 用 1.7.x 导出，
  请把运行环境 scikit-learn 升到 1.7.x 或重导模型，否则 rlRAG 评分微偏。
- **API key 未设**：设环境变量 `Mobile_trash_API_KEY`（或改 `config.yaml` 的 `llm.api_key_env`）。
  ⚠️ 不要写进任何入库文件。
- **端口被占**：换 `--server.port`。远程访问记得 `--server.headless true`。
- **内网穿透**：frp 示例 `frpc.ini` 里 `remote_port` 转发到本机 8502；ngrok 等同理。

## 6. 说明
- workspace/ 属数据产物，可全量重建，不入 git。
- 本包为服务端源码交付；客户端 = 浏览器，无需单独打包。
"""



def _iter_files(src: Path):
    """递归产出一致相对路径文件序列（排除 DIR/后缀/文件）。"""
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in EXCLUDE_SUFFIXES or p.name in {"debug.log"}:
            continue
        if p.is_dir():
            continue
        yield rel


def build() -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().strftime("%Y%m%d")
    out = DIST_DIR / f"服务端源码-{stamp}.zip"
    top = "服务端源码"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in _iter_files(ROOT):
            if rel.startswith("每日") or rel.startswith("日志"):
                continue
            # 只取白名单顶层项
            if rel.split("/", 1)[0] not in ITEMS and rel not in ITEMS:
                continue
            zf.write(ROOT / rel, f"{top}/{rel}")
            count += 1
        zf.writestr(f"{top}/运行说明.md", RUN_GUIDE)
        count += 1
    print(f"✅ 服务端源码包 → {out}（{count} 个文件，{out.stat().st_size/1024:.0f} KB）")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="服务端源码打包")
    ap.parse_args()
    build()
