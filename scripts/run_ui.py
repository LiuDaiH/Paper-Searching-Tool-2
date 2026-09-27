"""Streamlit UI 入口（`streamlit run scripts/run_ui.py`）。

`app.py` 在 `hotspot/ui/` 包内用了相对导入，不能直接 `streamlit run` 它；
此脚本把项目根加进 `sys.path` 后包式导入，再调 `main()`（含 st 代码）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hotspot  # noqa: E402 —— 装闸① + 接线 dasc
from hotspot.ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
