# -*- coding: utf-8 -*-
"""
md2vectors.py - offline MD -> embedding tool for 3d-semantic-graph (and own projects)

Requirements (conda env gymna_env already has them):
    pip install sentence-transformers            # pulls torch? no: reuse env that has torch
Model BAAI/bge-m3 is cached under HF_HOME (D:\hf_cache, ~2.3GB); no internet needed after
the first download. C: is full, so the cache is deliberately kept off the system drive.

Usage
-----
1) Generic: vectorize every .md under DIR into a JSON file (own projects):
       python md2vectors.py embed DIR [--out vectors.json] [--model NAME]

2) Import into the 3d-semantic-graph plugin of an Obsidian vault:
       python md2vectors.py import VAULT_ROOT [--model NAME]
   where VAULT_ROOT is the folder that contains .obsidian/plugins/3d-semantic-graph.
   This writes uploaded-vectors.json into that plugin folder, backs up data.json,
   and flips data.json["uploadedVectorsFileName"] + embeddingProvider/embeddingModel
   (the plugin's semantic search requires both to match the payload). Works on a vault
   whose plugin has never saved settings; Obsidian creates no data.json until it does.

   QUIT OBSIDIAN FIRST. The plugin holds settings in memory and rewrites data.json on
   its next save, clobbering whatever this script wrote. Reopen after importing.

Notes
-----
* Kept paths are relative to the scanned DIR (Obsidian needs vault-relative paths;
  so for import, VAULT_ROOT must BE the scanned dir).
* Output JSON schema: {"version":1,"provider":"custom","model":...,"dimensions":N,
  "entries":{"rel/path.md":{"embedding":[...]}}}
"""
import argparse, io, json, os, re, sys, pathlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_MODEL = "BAAI/bge-m3"
SKIP_DIRS = {".obsidian", ".git", ".trash", ".claude", "node_modules"}

# The HF default cache is on C:, which is full -- keep the model cache on D: instead.
# Must run before huggingface_hub is imported (i.e. before the SentenceTransformer import).
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

# bge-m3 is XLM-R-large (568M params) and this box has no usable GPU (MX330 = 2GB VRAM,
# while fp32 weights alone are 2.3GB), so everything runs on CPU. Its 8k-token ceiling
# would take minutes per document there; cap it and keep the batch small.
BATCH_SIZE = int(os.environ.get("MD2V_BATCH", "4"))
MAX_SEQ_LENGTH = int(os.environ.get("MD2V_MAXLEN", "1024"))


def clean(text):
    """Strip YAML frontmatter; unwrap wiki/image/url links; keep plain body."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:]
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[\[([^]|]*)\|([^]]*)\]\]", r"\2", text)   # [[t|alias]] -> alias
    text = re.sub(r"\[\[([^]]*)\]\]", r"\1", text)             # [[t]] -> t
    text = re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", text)         # [a](url) -> a
    return text.strip()


def scan_md(root):
    out = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            if n.lower().endswith(".md"):
                full = os.path.join(base, n)
                rel = os.path.relpath(full, root).replace("\\", "/")
                out.append((rel, full))
    out.sort()
    return out


def model_cache_exists(model):
    # huggingface_hub cache layout for snapshot downloads. HF_HUB_CACHE defaults to
    # $HF_HOME/hub, and HF_HOME is redirected to D: at the top of this file.
    tag = "models--" + model.replace("/", "--")
    root = os.environ.get("HF_HOME") or str(pathlib.Path.home() / ".cache" / "huggingface")
    return (pathlib.Path(root) / "hub" / tag).exists()


def load_texts(files):
    docs, rels = [], []
    for rel, full in files:
        try:
            raw = pathlib.Path(full).read_text(encoding="utf-8", errors="replace")
        except Exception:
            raw = ""
        body = clean(raw)
        docs.append(body if body else rel)
        rels.append(rel)
    return rels, docs


# 进程内模型缓存：同一进程多次 embed 复用已加载的 SentenceTransformer，避免重复加载权重
# （评测/批量任务同进程反复调用时是关键加速；输出与不缓存完全一致，向后兼容）。
_MODEL_CACHE = {}


def _resolve_local_model(model_name):
    """把 HF 仓库 ID（如 `BAAI/bge-m3`）解析成**本地已缓存路径**，命中则返回该绝对路径。

    ⚠️ 为什么必须这么做：`SentenceTransformer("BAAI/bge-m3")` 按**模型名**加载时，是走
    huggingface_hub 的 `snapshot_download`——即使模型就在本地缓存，它也可能**先联网做元数据
    HEAD**（`--local-dir` 平铺的缓存布局 hugface_hub 认不出是"自己下载的"，会去
    huggingface.co 逐个 GET modules.json 等，网络不通就 1-2-4-8-8s 退避数轮=分钟级假死）。
    而**直接传本地 snapshot 绝对路径**给 `SentenceTransformer(path)` 则完全离线、不联网。

    本地布局判据（标准 HF 缓存）：`$HF_HOME/hub/models--<org>--<name>/snapshots/<commit>/`，
    commit 取 `refs/main` 的内容；找不到/不完整则返回 model_name 原值（让底层自己处理）。
    """
    if "/" not in model_name or "\x00" in model_name:
        return model_name                      # 不是 `org/name` 形式（可能是本地已存在的路径）
    try:
        org, name = model_name.split("/", 1)
        root = pathlib.Path(os.environ.get("HF_HOME") or str(pathlib.Path.home() / ".cache" / "huggingface"))
        tag = f"models--{org}--{name}"
        base = root / "hub" / tag
        if not (base / "refs" / "main").exists():
            return model_name
        commit = (base / "refs" / "main").read_text(encoding="utf-8").strip()
        snap = base / "snapshots" / commit
        # 完整性抽查：模型权重 + 配置必须都在本地路径下，否则回退让底层联网/报错
        needed = ("pytorch_model.bin", "config.json", "tokenizer.json", "modules.json")
        if snap.is_dir() and all((snap / f).exists() for f in needed):
            return str(snap)
    except Exception:                          # noqa: BLE001 —— 解析失败就回退原值
        pass
    return model_name


def embed(rels, docs, model_name):
    from sentence_transformers import SentenceTransformer
    m = _MODEL_CACHE.get(model_name)
    if m is None:
        # 优先用**本地缓存绝对路径**加载（离线、不联网）；解析不到才回退模型名 ID。
        local_path = _resolve_local_model(model_name)
        m = SentenceTransformer(local_path)
        m.max_seq_length = MAX_SEQ_LENGTH
        _MODEL_CACHE[model_name] = m
    if not docs:
        return {r: {"embedding": []} for r in rels}, 0
    v = m.encode(docs, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=True)
    # NB: 3D Semantic Graph (v1.5) expects entries as {"path.md": {"embedding": [...]}},
    # NOT a bare array. Wrap each vector accordingly.
    return {rels[i]: {"embedding": [round(float(x), 6) for x in v[i]]}
            for i in range(len(rels))}, int(v.shape[1])


def short_model(model_name):
    """Bare model name. Used for the payload AND for plugin settings, so the two cannot
    drift: the plugin's semantic search requires settings' provider+model to equal the
    payload's, or it refuses with a 'provider unavailable' notice."""
    return model_name.split("/")[-1]


def build_payload(entries, dim, model_name):
    return {"version": 1, "provider": "custom",
            "model": short_model(model_name), "dimensions": dim,
            "entries": entries}


def plugin_dir_of(vault):
    return os.path.join(vault, ".obsidian", "plugins", "3d-semantic-graph")


def cmd_embed(args):
    rels, docs = load_texts(scan_md(args.dir))
    if not rels:
        print("no .md files found under", args.dir)
        return 1
    print("found %d markdown notes under %s" % (len(rels), args.dir))
    entries, dim = embed(rels, docs, args.model)
    payload = build_payload(entries, dim, args.model)
    out = args.out or os.path.join(os.getcwd(), "md_vectors.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("wrote %s (%d dim, %d notes, %.0f KB)" % (
        out, dim, len(entries), os.path.getsize(out) / 1024))
    return 0


def cmd_import(args):
    vault = os.path.abspath(args.dir)
    pdir = plugin_dir_of(vault)
    if not os.path.isdir(pdir):
        print("plugin dir not found under vault: %s" % pdir)
        print("(expected %s)" % os.path.join(vault, ".obsidian", "plugins", "3d-semantic-graph"))
        return 1
    rels, docs = load_texts(scan_md(vault))
    if not rels:
        print("no .md files found under", vault)
        return 1
    print("found %d markdown notes under vault %s" % (len(rels), vault))
    entries, dim = embed(rels, docs, args.model)
    payload = build_payload(entries, dim, args.model)

    out = os.path.join(pdir, "uploaded-vectors.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("wrote %s (%.0f KB)" % (out, os.path.getsize(out) / 1024))

    dj = os.path.join(pdir, "data.json")
    bak = dj + ".bak"
    if os.path.exists(dj) and not os.path.exists(bak):
        pathlib.Path(bak).write_text(pathlib.Path(dj).read_text(encoding="utf-8"), encoding="utf-8")
    # A vault whose plugin never saved settings has no data.json at all -- start from {}
    # rather than crashing, and rebuild rather than abort if an existing one is unreadable.
    try:
        cfg = json.loads(pathlib.Path(dj).read_text(encoding="utf-8")) if os.path.exists(dj) else {}
    except (OSError, ValueError):
        print("warning: %s unreadable, rebuilding from defaults" % dj)
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    # Non-empty name is what makes the plugin load the file; its text is only a UI label
    # (the reader hardcodes <plugin dir>/uploaded-vectors.json).
    cfg["uploadedVectorsFileName"] = "uploaded-vectors.json"
    cfg["embeddingProvider"] = "custom"                       # must equal payload provider
    cfg["embeddingModel"] = short_model(args.model)           # must equal payload model
    cfg["uploadedVectorsRevision"] = int(cfg.get("uploadedVectorsRevision") or 0) + 1
    pathlib.Path(dj).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("patched %s -> uploadedVectorsFileName=uploaded-vectors.json, provider=custom, model=%s"
          % (dj, cfg["embeddingModel"]))
    if os.path.exists(bak):
        print("backup:", bak)
    print("next: reopen the vault in Obsidian. If Obsidian was running while this ran, quit")
    print("      it and re-run -- it caches settings in memory and overwrites data.json.")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="md2vectors.py",
                                 description="Offline MD->vector for own projects / Obsidian 3d-semantic-graph.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("embed", help="vectorize .md under DIR to a generic JSON (no Obsidian)")
    p1.add_argument("dir", help="directory to scan for .md files")
    p1.add_argument("--out", default=None, help="output JSON path")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.set_defaults(fn=cmd_embed)

    p2 = sub.add_parser("import", help="vectorize an Obsidian vault and write plugin vectors")
    p2.add_argument("dir", help="Obsidian vault root (the folder holding .obsidian)")
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.set_defaults(fn=cmd_import)

    args = ap.parse_args()

    # offline when the model is already cached; otherwise point at the China mirror.
    # huggingface.co itself is unreachable from here, and a stale HF_HUB_OFFLINE=1 already
    # in the environment would silently block the download, so clear it explicitly.
    if model_cache_exists(args.model):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
