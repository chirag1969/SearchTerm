#!/usr/bin/env python3
"""Preprocess Excel workbooks into JSON for faster loading.

This script uses pandas/numpy to parse the workbook once and exports the raw
sheet data in the same layout that the front-end expects. The JSON can then be
served next to ``index.html`` so the UI can skip the costly browser-side XLSX
parsing step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "pandas and numpy are required. Install them via 'pip install pandas numpy openpyxl'."
    ) from exc

HEADER_ROW_INDEX = 1
TOKEN_GROUPS: Dict[str, Sequence[str]] = {
    "date": ("date", "day", "reportingdate"),
    "store": ("store", "lo", "market", "site", "locale", "country", "account"),
    "lo": ("lo", "lineofbusiness", "lob", "store", "market"),
    "impressions": ("impressions", "impr"),
    "clicks": ("clicks",),
    "revenue": ("sales", "revenue", "totalsales", "orderedrevenue", "attributedsales"),
    "spend": ("spend", "adspend", "cost", "advertisingcost"),
    "category": ("category", "producttype", "cat"),
    "ttype": ("targetingtype", "tagetingtype", "adtype", "type"),
    "tsub": ("targetingsubtype", "tagetingsubtype", "matchtype", "subtype"),
    "tsub2": ("subtype2", "matchtype2", "tagetingsubtype2"),
    "term": ("customersearchterm", "searchterm", "term"),
    "customerType": ("customersearchtermtype", "searchtermtype"),
    "targetingAsin": ("targetingasin", "targetasin", "advertisedasin", "asin"),
    "targetingCat": ("targetingcat", "targetcat"),
    "placement": ("placement", "placements"),
    "campaign": ("campaignname", "campaign"),
    "adgroup": ("adgroupname", "ad group name", "adgroup"),
    "portfolio": ("portfolioname", "portfolio"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Path to the Excel workbook")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON path (defaults to ./preprocessed/<workbook>.json)",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional sheet name. When omitted the most relevant sheet is picked automatically.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output (default: 2).",
    )
    return parser.parse_args()


def normalize(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def find_index(headers: Sequence[str], tokens: Iterable[str]) -> int:
    normalized_headers = [normalize(h) for h in headers]
    token_list = []
    for token in tokens:
        norm_token = normalize(token)
        if norm_token:
            token_list.append(norm_token)
    if not token_list:
        return -1
    for token in token_list:
        try:
            return normalized_headers.index(token)
        except ValueError:
            continue
    best_idx = -1
    best_score = 0.0
    for idx, header in enumerate(normalized_headers):
        if not header:
            continue
        for token in token_list:
            pos = header.find(token)
            if pos == -1:
                continue
            score = len(token) / max(len(header), 1)
            if pos == 0:
                score += 0.15
            if pos + len(token) == len(header):
                score += 0.35
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx if best_score > 0 else -1


def score_headers(headers: Sequence[str]) -> int:
    score = 0
    for key in ("store", "revenue", "spend", "clicks", "impressions"):
        if find_index(headers, TOKEN_GROUPS[key]) >= 0:
            score += 1
    return score


def to_python_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.timezone.utc)
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip()
    return value


def trim_trailing(values: List[object]) -> List[object]:
    idx = len(values)
    while idx and values[idx - 1] is None:
        idx -= 1
    return values[:idx]


def dataframe_to_rows(df: pd.DataFrame) -> List[List[object]]:
    rows: List[List[object]] = []
    for record in df.itertuples(index=False, name=None):
        row = [to_python_value(value) for value in record]
        rows.append(trim_trailing(row))
    return rows


def pick_sheet(sheets: Dict[str, pd.DataFrame], preferred: str | None) -> str:
    if preferred and preferred in sheets:
        return preferred
    best_name = None
    best_score = -1
    for name, df in sheets.items():
        if df.shape[0] <= HEADER_ROW_INDEX:
            continue
        headers = [str(v or "").strip() for v in df.iloc[HEADER_ROW_INDEX].tolist()]
        score = score_headers(headers)
        if score > best_score:
            best_name = name
            best_score = score
    if not best_name:
        raise ValueError("No usable sheet found")
    return best_name


def build_output(sheet: pd.DataFrame, source_name: str, sheet_name: str) -> Dict[str, object]:
    sheet = sheet.copy()
    header_row = sheet.iloc[HEADER_ROW_INDEX].tolist()
    header_row = [to_python_value(v) for v in header_row]
    header_row = trim_trailing(header_row)
    target_width = len(header_row)
    if target_width == 0:
        raise ValueError("Header row appears to be empty")
    relevant = sheet.iloc[: HEADER_ROW_INDEX + 1]
    remaining = sheet.iloc[HEADER_ROW_INDEX + 1 :]
    rows = dataframe_to_rows(relevant)
    rows.extend(dataframe_to_rows(remaining))
    normalised_rows: List[List[object]] = []
    for row in rows:
        trimmed = row[:target_width]
        if len(trimmed) < target_width:
            trimmed = trimmed + [None] * (target_width - len(trimmed))
        normalised_rows.append(trimmed)
    return {
        "source": source_name,
        "sheet_name": sheet_name,
        "header_row_index": HEADER_ROW_INDEX,
        "generated_at": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
        "sheet_data": normalised_rows,
    }


def main() -> None:
    args = parse_args()
    workbook_path = args.workbook.resolve()
    if not workbook_path.exists():
        raise SystemExit(f"Workbook not found: {workbook_path}")

    sheets = pd.read_excel(
        workbook_path,
        sheet_name=None,
        header=None,
        engine="openpyxl",
        dtype=object,
    )
    sheet_name = pick_sheet(sheets, args.sheet)
    sheet_df = sheets[sheet_name]
    payload = build_output(sheet_df, workbook_path.name, sheet_name)

    if args.output is None:
        output_dir = workbook_path.parent / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (workbook_path.stem + ".json")
    else:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=args.indent)
        fp.write("\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
