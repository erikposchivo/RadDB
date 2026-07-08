"""
raddb/hc_mapping.py
-------------------
Canonical hydrometeor class constants for the MCH operational encoding.

Both HC_MCH and HC_PYART are stored in parquet on the same 1-based scale:
  parquet integer k  →  HC_MAP_DICT[k - 1]

HC_MCH raw (HYM file, after /25) is already 0-based (0-8).
HC_PYART (PyART native 1-9) is remapped to 0-based operational via
PYART_TO_OPE before archiving, then shifted +1 to match HC_MCH.

Import these constants wherever HC label/color information is needed.
"""

# 0-based operational classification (raw HC_MCH / 25 → this dict)
HC_MAP_DICT: dict[int, str] = {
    0: "None",
    1: "CR/VI",
    2: "AG",
    3: "LR",
    4: "RN",
    5: "RP",
    6: "WS",
    7: "IH/HDG",
    8: "MH",
}

# 1-based parquet labels (HC_MAP_DICT shifted by +1, matches stored integers 1–9)
HC_CLASSES: list[str] = [HC_MAP_DICT[k] for k in range(9)]   # index 0 → parquet 1

# Supervisor color palette (index-aligned with HC_CLASSES / HC_MAP_DICT 0-based)
HC_COLORS: list[str] = [
    "gray",         # 0 / parquet 1  — None
    "cyan",         # 1 / parquet 2  — CR/VI
    "deepskyblue",  # 2 / parquet 3  — AG
    "royalblue",    # 3 / parquet 4  — LR
    "midnightblue", # 4 / parquet 5  — RN
    "green",        # 5 / parquet 6  — RP
    "yellow",       # 6 / parquet 7  — WS
    "orangered",    # 7 / parquet 8  — IH/HDG
    "darkred",      # 8 / parquet 9  — MH
]

# label → color lookup (convenient for scatter plots)
HC_COLOR_BY_LABEL: dict[str, str] = dict(zip(HC_CLASSES, HC_COLORS))

# Remap PyART native integers (1–9) → MCH operational integers (1–8, 0-based index)
# PyART hydro_names order: ("AG","CR","LR","RP","RN","VI","WS","MH","IH")
PYART_TO_OPE: dict[int, int] = {
    1: 2,   # AG  → operational 2 (AG)
    2: 1,   # CR  → operational 1 (CR/VI)
    3: 3,   # LR  → operational 3 (LR)
    4: 5,   # RP  → operational 5 (RP)
    5: 4,   # RN  → operational 4 (RN)
    6: 1,   # VI  → operational 1 (CR/VI, merged with CR)
    7: 6,   # WS  → operational 6 (WS)
    8: 8,   # MH  → operational 8 (MH)
    9: 7,   # IH  → operational 7 (IH/HDG)
}
