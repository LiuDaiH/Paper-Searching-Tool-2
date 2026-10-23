"""B 档发布：组装三个**文件夹**到 `G:\\work!`（不压缩）。

  G:\\work!\\
   ├─ 服务端源码\\        完整引擎源码 + 通用 config + requirements + 运行说明
   ├─ 样例数据集\\        「智引热点」独立应用快照：源码副本 + 精简 workspace
   │                      （语料/向量/bm25）+ sample config + 启动 + 运行说明 —— 即用，不依赖G盘
   ├─ 前端源码\\          UI 层源码（hotspot/ui/ + scripts/run_ui.py）+ 前端说明
   └─ README.md          总览 + 客户端(浏览器)访问方式

样例数据集=独立快照：HOTSPOT_ROOT=其根 => paths.workspace 相对解析到本地精简 workspace。
前端是 Streamlit 页面，源码在 hotspot/ui/，与引擎同库；这里单列一份便于查看/交付。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"G:\work!")
STAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\Admin\AppData\Local\Temp\zhanlu\release_stage")

EXCLUDE_PARTS = {"__pycache__", ".git", ".playwright-mcp"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".log", ".npz"}

# 服务端源码含的顶层项
ITEMS = ["hotspot", "scripts", "config.yaml", "requirements.txt",
         "README.md", "AGENTS.md", "规划"]

SNAP = "智引热点"


def _clean(path: Path) -> None:
    # 允许删除失败（目录可能被占用/是某进程的 CWD），靠 dirs_exist_ok 覆盖填充
    shutil.rmtree(path, ignore_errors=True)


def _iter_project_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in EXCLUDE_PARTS for part in p.parts):
            continue
        if p.suffix.lower() in EXCLUDE_SUFFIX or p.name == "debug.log":
            continue
        if rel.split("/", 1)[0] not in ITEMS and rel not in ITEMS:
            continue
        yield rel


def _copy_items(src_root: Path, dst_root: Path) -> None:
    for rel in _iter_project_files(src_root):
        s = src_root / rel
        d = dst_root / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)


def _snapshot_workspace(sample_dir: Path, ws: Path) -> None:
    v = ws / "vectors"; v.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_dir / "vec_meeting.npz", v / "vec_0000.npz")
    shutil.copy(sample_dir / "bm25_meeting.pkl", ws / "bm25_cache.pkl")
    shutil.copy(sample_dir / "sample_meeting.jsonl", ws / "sample_meeting.jsonl")
    shutil.copy(sample_dir / "sample_arxiv.jsonl", ws / "sample_arxiv.jsonl")
    av = ws / "arxiv_vectors"; av.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_dir / "vec_arxiv.npz", av / "vec_0000.npz")


def _sample_config_text() -> str:
    return """# 样例数据集运行配置（B 档独立快照；即用，不依赖 G 盘）
corpus:
  sources:
    - workspace/sample_meeting.jsonl     # 从全量 24+25 切的 1200 篇子集
  arxiv_source: workspace/sample_arxiv.jsonl   # 从 arXiv 周报切的 600 篇子集

paths:
  workspace: workspace

embed:
  model: BAAI/bge-m3
  chunk_size: 500

llm:
  api_key_env: Mobile_trash_API_KEY   # 你自己的 key 变量名（运行态设置，永不落盘）
  # 主问答端点与评测一致——你的 key 是移动云 MaaS 的，打官方 api.deepseek.com 会 401。
  base_url: https://zhenze-huhehaote.cmecloud.cn/v1
  model_flash: deepseek-v4-flash-0731
  model_pro: deepseek-v4-flash-0731
  timeout: 600

llm_eval:
  base_url: https://zhenze-huhehaote.cmecloud.cn/v1
  model: deepseek-v4-flash-0731
  api_key_env: Mobile_trash_API_KEY
  timeout: 600

agent:
  search_limit: 12
  fetch_limit: 6
  max_claims: 6
  request_limit: 10
  tool_limit: 10
  scope_timeout_s: 30
  agent_timeout_s: 150
  verify_timeout_s: 30
"""

SNAP_README = """# 智引热点 · 样例应用（独立快照）

- 独立运行：源码 + 精简 workspace 已就位，**不依赖 G 盘 / 原始几十 GB 语料**；
  基于 1200 篇会议 + 600 篇 arXiv 子集的样例数据。
- 数据源左侧切换「会议 24+25 / arXiv 周报」。

## 快速开始（三步）
1. 装依赖：`pip install -r requirements.txt`（含 `sentence-transformers`，会自动带 torch）
2. **下载嵌入模型 bge-m3（先联网下载一次，之后离线）**：
   ```bash
   pip install huggingface_hub
   huggingface-cli download BAAI/bge-m3 --local-dir D:\\hf_cache\\hub\\models--BAAI--bge-m3
   ```
   项目经 **`hotspot/vendor/md2vectors.py`（已随包 vendor）** 加载：`_resolve_local_model` 把模型名解析成
   **本地 snapshot 绝对路径**（默认 `HF_HOME=D:\\hf_cache`）再 `SentenceTransformer(路径)` **离线加载、不联网**。
   权重别放 C 盘（C 盘满会 Python 段错误，见 `md2vectors.py`）。就位校验：
   `python -c "import md2vectors; print(md2vectors._resolve_local_model('BAAI/bge-m3'))"`
   → 应打印**本地 snapshot 路径**（而非模型名）。
3. 启动：`streamlit run run_ui.py --server.port 8502`，浏览器打开 http://127.0.0.1:8502。

> ⚠️ **bge-m3 是运行必备**：意图聚类 / 语义向量检索 / Streit-Garg 路线图都依赖在线 embedding，
> 缺它会退回纯词级 BM25、这些功能残缺（不报错但效果打折）。样例虽自带**预计算好的子集向量**
> （workspace/vectors/*.npz），但 query 向量化仍要模型现场加载。
> 切换 arXiv 源如需正确的 BM25，请对 workspace/sample_arxiv.jsonl 重建 bm25 缓存。
> （样例是**子集演示效果**；完整结论请在原始语料上跑服务端源码。）
"""

FRONT_README = """# 前端源码（Streamlit UI 层）

- 前端是 **Streamlit** 页面，源码：`ui/app.py`（多会话问答/数据源切换/追问）+ `scripts/run_ui.py`（启动入口）。
- 依赖服务端引擎（hotspot 的 corpus/检索/聚类/生成），二者在同一包内运行。
- 启动：`streamlit run scripts/run_ui.py --server.port 8502`；入口必须是 run_ui.py（sys.path 接线）。
- 关键 UI 文件：
  - `hotspot/ui/app.py`  主页面：会话/数据源/API 配置/追问/消息渲染
  - `scripts/run_ui.py`  启动入口（不能用 `streamlit run hotspot/ui/app.py`）
"""


def _read_guide() -> str:
    import re
    g = (ROOT / "scripts" / "build_dist.py").read_text(encoding="utf-8")
    m = re.search(r'RUN_GUIDE = """(.*?)"""', g, re.S)
    return m.group(1) if m else "# 运行说明（服务端）"


def build() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    # 1) 服务端源码/（目录）
    server = DEST / "服务端源码"
    _clean(server)
    _copy_items(ROOT, server)
    (server / "运行说明.md").write_text(_read_guide(), encoding="utf-8")

    # 2) 样例数据集/智引热点（独立快照，目录）
    snap = DEST / "样例数据集" / SNAP
    _clean(DEST / "样例数据集")
    snap.parent.mkdir(parents=True, exist_ok=True)
    for d in ("hotspot", "scripts"):
        shutil.copytree(ROOT / d, snap / d,
                        ignore=shutil.ignore_patterns(
                            *[p + "*" for p in EXCLUDE_PARTS],
                            *["*" + s for s in EXCLUDE_SUFFIX]),
                        dirs_exist_ok=True)
    _snapshot_workspace(ROOT / "workspace" / "sample", snap / "workspace")
    (snap / "config.yaml").write_text(_sample_config_text(), encoding="utf-8")
    (snap / "requirements.txt").write_text(
        (ROOT / "requirements.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (snap / "run_ui.py").write_text(
        (ROOT / "scripts" / "run_ui.py").read_text(encoding="utf-8"), encoding="utf-8")
    (snap / "运行说明-样例数据.md").write_text(SNAP_README, encoding="utf-8")

    # 3) 前端源码/（目录）
    front = DEST / "前端源码"
    _clean(front)
    shutil.copytree(ROOT / "hotspot" / "ui", front / "hotspot" / "ui",
                    ignore=shutil.ignore_patterns(
                        *[p + "*" for p in EXCLUDE_PARTS],
                        *["*" + s for s in EXCLUDE_SUFFIX]),
                    dirs_exist_ok=True)
    (front / "hotspot" / "ui").mkdir(parents=True, exist_ok=True)
    (front / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "run_ui.py", front / "scripts" / "run_ui.py")
    (front / "README-前端源码.md").write_text(FRONT_README, encoding="utf-8")

    # 4) 根 README
    (DEST / "README.md").write_text(_root_readme(), encoding="utf-8")
    print("G:\\work! 就绪（文件夹，未压缩）：", sorted(p.name for p in DEST.iterdir()))


def _root_readme() -> str:
    return """# 移动云杯 · 热点挖掘 —— 交付包（G:\\work!）

## 目录（本目录下均为**文件夹**，未压缩）
- **服务端源码/** —— 完整引擎源码（hotspot/ + scripts/，**已含 vendor 化的 dasc 与 md2vectors.py**）
  + 通用 config + requirements + 运行说明。
- **样例数据集/** —— 「智引热点」独立快照：源码副本 + 1200 篇会议 + 600 篇 arXiv 子集的
  精简 workspace（语料/向量/BM25），即用、不依赖 G 盘，可跑通检索/聚类做演示。
- **前端源码/** —— Streamlit UI 层（hotspot/ui/ + scripts/run_ui.py）+ 说明。
- **周报爬取/** —— arXiv 周报爬虫代码副本（抓 RSS→去重入库→取许可→RL 打标），
  来自 `D:\\大学编程学习\\arXiv周报\\程序`；运行细节见其 `运行说明-打包说明.md`。
- **README.md** —— 本文件。

## 客户端
本项目为「服务端 + 浏览器客户端」架构：**客户端就是浏览器**，无需单独安装包。
- 服务端启动：`streamlit run scripts/run_ui.py --server.port 8502`（服务端源码）或
  `streamlit run run_ui.py --server.port 8502`（样例快照）。
- 别人远程使用：给 8502 做内网穿透（frp/ngrok），浏览器访问公网地址即可，零安装。

## 快速开始（样例）——只用「样例数据集」即可完整演示
1. 进入 样例数据集/智引热点/，`pip install -r requirements.txt`；
2. **下载嵌入模型 bge-m3（运行必备）**：
   ```bash
   pip install huggingface_hub
   huggingface-cli download BAAI/bge-m3 --local-dir D:\hf_cache\hub\models--BAAI--bge-m3
   ```
   （详见下方「bge-m3 嵌入模型」；意图聚类/语义检索/SG 图都依赖它。）
3. `streamlit run run_ui.py --server.port 8502`，浏览器打开 http://127.0.0.1:8502，
   左侧切换「会议 24+25 / arXiv 周报」搜索。

## 用完整数据（服务端）
进入 服务端源码/，按 运行说明.md 把 G 盘语料、模型缓存、API key 就位后运行；
arXiv 周报走 `scripts/update_arxiv_weekly.py` 增量。

> 注意：样例数据的检索/聚类是**子集演示效果**；完整结论请在原始语料上运行服务端源码。

## bge-m3 嵌入模型：从哪来、怎么接
- **不在交付包内**。bge-m3（`BAAI/bge-m3`）是 HuggingFace 开源模型，需**联网下载一次**（约 2.3GB fp32）。
- 之后**加载是离线的**：`hotspot/vendor/md2vectors.py`（已随包 vendor）用 `_resolve_local_model` 把模型名
  解析成**本地 snapshot 绝对路径**（`$HF_HOME/hub/models--BAAI--bge-m3/snapshots/<commit>/`）再
  `SentenceTransformer(路径)` 加载，**不联网**；`HF_HOME` 未设时默认 `D:\hf_cache`，命中即用、零配置。
- 默认方式（下载一次到 D 盘默认位置）：
  ```bash
  pip install huggingface_hub sentence-transformers
  huggingface-cli download BAAI/bge-m3 --local-dir D:\\hf_cache\\hub\\models--BAAI--bge-m3
  ```
- 需要 `torch` + `sentence-transformers`；权重别放 C 盘（C 盘满会 Python 段错误）。
- 完整说明见 `服务端源码/运行说明.md` §2.1 与 `样例数据集/智引热点/运行说明-样例数据.md`。

## 常见问题：前端能改哪些配置？
- **运行态配置（可在浏览器改）**：左侧「API 配置」可填 `base_url` / `model` / `API key`，
  填了**自动保存**到本地 `workspace/settings.json`（本机持久化），启动时自动读回、提问时
  直接采用（key 留空则改用 `config.yaml` 里的 `llm.api_key`）。这类改的是**运行时端口配置**，
  与 `config.yaml` 分开、不会覆盖引擎默认。
- **引擎配置（`config.yaml`，仍需改服务器文件）**：语料路径、embed 参数、预算、评测端点等。
  `config.yaml` 只在服务端进程启动时读一次（`load_config()`），改完**重启 `streamlit`** 生效。
- **为什么前端不能直接改 `config.yaml`**：浏览器 JS 出于安全无法写服务器磁盘，所以引擎配置
  在前端不可写；前端只读/写它自己的 `settings.json` 运行态配置，二者互不影响。
"""


if __name__ == "__main__":
    build()
