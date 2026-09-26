"""L5 推理层 —— 答案生成 + 引用校验（防幻觉最后防线）。

- :mod:`.answer` —— 带引用生成（强制 [P:id]/[C:cid]）
- :mod:`.verify` —— 引用校验（missing_id / unsupported 分开判）
"""
