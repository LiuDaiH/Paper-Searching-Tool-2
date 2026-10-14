# Linux 迁移评估（长期参考）

> 2026-10-14 记录（本是一件导师方向/运维参考，归入 `规划/`）。
> 结论：**迁移成本很低，技术栈全部跨平台，仅需改几处路径硬编码。**

## 1. 结论

- 技术栈全部跨平台：Python 3.13 / torch / sklearn / streamlit / plotly / MySQL / dasc 纯 Python，**没有 C 扩展或平台绑定的运行库** → "搬过去就能跑"。
- 需要动的地方**只集中在路径硬编码**，见 §2 表。

## 2. 需要动的地方

| 位置 | 现状（Windows 路径） | Linux 改法 | 量 |
|---|---|---|---|
| `hotspot/vendor/md2vectors.py:42` | `os.environ.setdefault("HF_HOME", r"D:\hf_cache")` | 本地路径或删掉靠环境变量 | 1 行 |
| `config.yaml` 语料路径 | `G:\rl_data24` / `G:\rl_data` / arXiv D 盘 | 挂载点 `/mnt/...` | 配置 |
| `scripts/run_deps.py:24` 默认根 | `G:\rl_data24\cache` | 传参支持，改默认 | 1 行 |
| `build_dist.py` / `build_release.py` | `G:\work!`、`D:\...` | **多在 docstring/生成文档文本**，非逻辑；不影响运行 | 0（文档） |

## 3. 注意点

- 构建脚本的 Windows 硬编码绝大部分在 docstring/示例文本里（Linux 上运行不报错，只是生成的交付 README 仍指 Windows 路径，可顺带改 `${HOME}` 模板，非必需）。
- `dasc/__init__.py` 的 GBK 控制台修复段（cp936 兜底）在 Linux（UTF-8）是死代码，`isatty()` 判据直接跳过，无副作用。

## 4. 迁移的实质价值

通常是**换头服务器部署**（Linux + CUDA），正好叠加 12G 显存/大内存 → 一次解决：
- 全库向量拼大矩阵 **GPU 提速**（算法档任务，当前 8GB 被迫 mmap 按需读）
- **本地 LLM**（Qwen 14B 量化，base_url 指 vLLM/Ollama）
- **社区 Jev**（151M，CPU/GPU 均跑得动）
- **MySQL** 账号系统 + **微信助手**常驻

## 5. 待办

- [ ] 整理成 `docs/linux迁移.md` 小抄（5 处路径改法 + 依赖清单 + 验证步骤）
- [ ] 部署到 12G 显存/大内存 Linux 机

## 相关

- `日志/2026-10-14-...md`（当日流水账）
- `审查分工.md`（算法档 GPU 提速 / 接口上下文 Linux 协调）
