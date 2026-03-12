from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from data_processing import (
    build_plot_payload,
    build_summary_payload,
    clean_dataframe,
    evaluate_dataframe_expression_outputs,
    load_csv,
    make_table_payload,
    normalize_column_names,
    refresh_summary_payload,
)

load_dotenv()

APP_SOURCE = "csv-agent-app"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FILE_LIMIT = 5
PAGE_SIZE_OPTIONS = (25, 50, 100)
CSV_ACCEPT = {
    "text/csv": [".csv"],
    "text/plain": [".csv"],
    "application/vnd.ms-excel": [".csv"],
    "application/csv": [".csv"],
}


class RenderPlotInput(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "histogram", "box", "pie"] = Field(
        ..., description="Plotly chart type to render in the left pane."
    )
    x: str = Field(..., description="Column name for the x axis or category dimension.")
    y: Optional[str] = Field(
        default=None,
        description="Optional numeric column for the y axis or values dimension.",
    )
    color: Optional[str] = Field(
        default=None,
        description="Optional column used to color or group the chart.",
    )
    aggregation: Optional[Literal["sum", "mean", "count", "median", "min", "max"]] = Field(
        default=None,
        description="Optional aggregation to apply before rendering. Use count for frequency charts.",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Optional limit to the top N rows/groups after sorting.",
    )
    sort_desc: bool = Field(
        default=True,
        description="Whether to sort descending before applying top_n.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional chart title. Leave empty to auto-generate one.",
    )

    @staticmethod
    def _coerce_column_name(value: Any) -> Optional[str]:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
            value = value[0].strip()
            return value or None
        return None

    @classmethod
    def _extract_plotly_args(cls, raw: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        figure = raw.get("figure")
        if isinstance(figure, Mapping):
            return cls._extract_plotly_args(figure)

        if "data" not in raw and "layout" not in raw:
            return None

        traces = raw.get("data") or []
        first_trace = traces[0] if isinstance(traces, list) and traces and isinstance(traces[0], Mapping) else {}

        trace_type = str(first_trace.get("type") or "").strip().lower()
        mode = str(first_trace.get("mode") or "").strip().lower()
        if trace_type == "scatter":
            chart_type = "line" if "lines" in mode and "markers" not in mode else "scatter"
        else:
            chart_type = trace_type or None

        if chart_type not in {"bar", "line", "scatter", "histogram", "box", "pie"}:
            return None

        x = cls._coerce_column_name(first_trace.get("x"))
        y = cls._coerce_column_name(first_trace.get("y"))
        if chart_type == "pie" and not x:
            x = cls._coerce_column_name(first_trace.get("labels") or first_trace.get("names"))

        title_value = None
        layout = raw.get("layout")
        if isinstance(layout, Mapping):
            layout_title = layout.get("title")
            if isinstance(layout_title, str):
                title_value = layout_title.strip() or None
            elif isinstance(layout_title, Mapping):
                title_value = cls._coerce_column_name(layout_title.get("text"))

        if not chart_type or not x:
            return None

        normalized: dict[str, Any] = {
            "chart_type": chart_type,
            "x": x,
            "y": y,
        }
        if title_value:
            normalized["title"] = title_value
        return normalized

    @model_validator(mode="before")
    @classmethod
    def normalize_tool_input(cls, raw: Any) -> Any:
        if isinstance(raw, Mapping):
            normalized = cls._extract_plotly_args(raw)
            if normalized:
                return normalized
        return raw


class PreprocessDataFrameInput(BaseModel):
    target_column: list[str] = Field(
        ...,
        description=(
            "Name of the new or existing column to create or update. "
            "Use a list when the preprocessing expression returns multiple derived columns."
        ),
    )
    expression: str = Field(
        ...,
        description=(
            "A pandas-style expression using dataframe columns, for example "
            "'out_tpd - out_tpd2' or '(out_ln * 1e9, out_lp * 1e9)'. "
            "Do not include assignment syntax."
        ),
    )
    preview_rows: Optional[int] = Field(
        default=None,
        ge=5,
        le=100,
        description=(
            "Optional number of rows per page to show immediately after preprocessing. "
            "If omitted, the app keeps the current rows-per-page setting."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_target_columns(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw

        payload = dict(raw)
        if "target_column" not in payload and "target_columns" in payload:
            payload["target_column"] = payload["target_columns"]

        target = payload.get("target_column")
        if isinstance(target, str):
            parts = [part.strip() for part in re.split(r"[,\n;]+", target) if part.strip()]
            payload["target_column"] = parts or [target.strip()]
        elif isinstance(target, tuple):
            payload["target_column"] = list(target)

        return payload

    @model_validator(mode="after")
    def validate_target_columns(self) -> "PreprocessDataFrameInput":
        cleaned = [str(column).strip() for column in self.target_column if str(column).strip()]
        if not cleaned:
            raise ValueError("At least one target column is required.")
        self.target_column = cleaned
        return self


class ChartOptionsChangedInput(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "histogram", "box", "pie"]
    x: str
    y: Optional[str] = None
    color: Optional[str] = None
    aggregation: Optional[Literal["sum", "mean", "count", "median", "min", "max"]] = None
    top_n: Optional[int] = None
    sort_desc: bool = True
    title: Optional[str] = None

    @model_validator(mode="after")
    def validate_fields(self) -> "ChartOptionsChangedInput":
        self.x = str(self.x).strip()
        if not self.x:
            raise ValueError("x is required.")

        if self.y is not None:
            self.y = str(self.y).strip() or None
        if self.color is not None:
            self.color = str(self.color).strip() or None
        if self.title is not None:
            self.title = str(self.title).strip() or None

        if self.top_n is not None and int(self.top_n) <= 0:
            raise ValueError("top_n must be a positive integer.")

        if self.chart_type == "pie" and not self.y and self.aggregation != "count":
            raise ValueError("Pie charts require a y column or aggregation='count'.")

        return self


MENTION_QUOTED_RE = re.compile(r'@"([^"]+)"')
MENTION_TOKEN_RE = re.compile(r"(?<!\w)@([^\s,;:!?()\[\]{}]+)")


async def send_window_payload(message_type: str, payload: dict[str, Any]) -> None:
    message = {"source": APP_SOURCE, "type": message_type, "payload": payload}
    await cl.send_window_message(json.dumps(message, default=str))


async def publish_ui_state() -> None:
    ui_state = cl.user_session.get("ui_state") or {}
    await send_window_payload("ui_state", ui_state)


def format_step_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(payload)


def _coerce_table_page_size(value: Any, *, default: int = 25) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(5, min(parsed, 100))


def _coerce_requested_page(value: Any, *, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _get_datasets() -> dict[str, dict[str, Any]]:
    datasets = cl.user_session.get("datasets")
    return datasets if isinstance(datasets, dict) else {}


def _set_datasets(datasets: dict[str, dict[str, Any]]) -> None:
    cl.user_session.set("datasets", datasets)


def _get_dataset_order() -> list[str]:
    dataset_order = cl.user_session.get("dataset_order")
    return list(dataset_order) if isinstance(dataset_order, list) else []


def _set_dataset_order(dataset_order: list[str]) -> None:
    cl.user_session.set("dataset_order", dataset_order)


def _get_active_dataset_key() -> Optional[str]:
    value = cl.user_session.get("active_dataset_key")
    return str(value) if value else None


def _set_active_dataset_key(dataset_key: Optional[str]) -> None:
    cl.user_session.set("active_dataset_key", dataset_key)


def _active_dataset_record() -> Optional[dict[str, Any]]:
    dataset_key = _get_active_dataset_key()
    if not dataset_key:
        return None
    return _get_datasets().get(dataset_key)


def _slugify_reference(value: str) -> str:
    cleaned = re.sub(r"[^\w]+", "_", str(value or "").strip().lower(), flags=re.UNICODE).strip("_")
    return cleaned or "file"


def _make_unique_display_name(original_name: str, existing_names: set[str]) -> str:
    candidate = str(original_name or "uploaded.csv").strip() or "uploaded.csv"
    existing_lower = {name.lower() for name in existing_names}
    if candidate.lower() not in existing_lower:
        return candidate

    path = Path(candidate)
    stem = path.stem or "uploaded"
    suffix = path.suffix or ".csv"
    index = 2
    while True:
        candidate = f"{stem} ({index}){suffix}"
        if candidate.lower() not in existing_lower:
            return candidate
        index += 1


def _make_unique_reference_alias(display_name: str, existing_aliases: set[str]) -> str:
    base_alias = _slugify_reference(Path(display_name).stem or display_name)
    existing_lower = {alias.lower() for alias in existing_aliases}
    if base_alias.lower() not in existing_lower:
        return base_alias

    index = 2
    while True:
        candidate = f"{base_alias}_{index}"
        if candidate.lower() not in existing_lower:
            return candidate
        index += 1


def _build_reference_aliases(display_name: str, mention: str) -> list[str]:
    stem = Path(display_name).stem
    candidates = {
        display_name,
        display_name.replace(" ", "_"),
        stem,
        stem.replace(" ", "_"),
        mention,
        f"{mention}.csv",
    }
    return sorted({candidate.strip().lower() for candidate in candidates if candidate and candidate.strip()})


def _dataset_lookup_maps() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for dataset_key in _get_dataset_order():
        record = _get_datasets().get(dataset_key)
        if not record:
            continue
        for alias in record.get("aliases") or []:
            lookup[str(alias).lower()] = dataset_key
    return lookup


def _match_dataset_reference(token: str) -> Optional[str]:
    cleaned = str(token or "").strip().strip(".,)").strip()
    if not cleaned:
        return None
    return _dataset_lookup_maps().get(cleaned.lower())


def extract_dataset_mentions(content: str) -> tuple[list[str], list[str], str]:
    text = str(content or "")
    spans: list[tuple[int, int]] = []
    resolved_keys: list[str] = []
    unresolved_tokens: list[str] = []
    resolved_spans: list[tuple[int, int]] = []

    def overlaps(candidate: tuple[int, int]) -> bool:
        start, end = candidate
        return any(not (end <= span_start or start >= span_end) for span_start, span_end in spans)

    for match in MENTION_QUOTED_RE.finditer(text):
        span = match.span()
        spans.append(span)
        token = match.group(1)
        dataset_key = _match_dataset_reference(token)
        if dataset_key:
            if dataset_key not in resolved_keys:
                resolved_keys.append(dataset_key)
            resolved_spans.append(span)
        else:
            unresolved_tokens.append(f'@"{token}"')

    for match in MENTION_TOKEN_RE.finditer(text):
        span = match.span()
        if overlaps(span):
            continue
        spans.append(span)
        token = match.group(1)
        dataset_key = _match_dataset_reference(token)
        if dataset_key:
            if dataset_key not in resolved_keys:
                resolved_keys.append(dataset_key)
            resolved_spans.append(span)
        else:
            unresolved_tokens.append(f"@{token}")

    if resolved_spans:
        parts: list[str] = []
        cursor = 0
        for start, end in sorted(resolved_spans):
            parts.append(text[cursor:start])
            parts.append(" ")
            cursor = end
        parts.append(text[cursor:])
        cleaned_content = "".join(parts)
    else:
        cleaned_content = text

    cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip(" ,")
    return resolved_keys, unresolved_tokens, cleaned_content


def _available_reference_summary() -> str:
    files = [
        f"`@{record['mention']}` for **{record['name']}**"
        for dataset_key in _get_dataset_order()
        if (record := _get_datasets().get(dataset_key))
    ]
    return ", ".join(files) if files else "No CSV files are registered yet."


def _get_dataset_or_error(dataset_key: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    datasets = _get_datasets()
    record = datasets.get(dataset_key)
    if not record:
        raise ValueError("The selected CSV file is no longer registered in this session.")
    return datasets, record


def _build_table_payload_for_record(record: dict[str, Any]) -> dict[str, Any]:
    df = record["df"]
    table_state = dict(record.get("table_state") or {})
    focus_columns = [
        str(column)
        for column in (table_state.get("focus_columns") or [])
        if isinstance(column, str)
    ]
    table_payload = make_table_payload(
        df,
        rows=_coerce_table_page_size(table_state.get("page_size"), default=25),
        page=_coerce_requested_page(table_state.get("page"), default=1),
        priority_columns=focus_columns,
        highlight_columns=focus_columns,
    )
    record["table_state"] = {
        "page_size": int(table_payload.get("page_size") or 25),
        "page": int(table_payload.get("page") or 1),
        "focus_columns": focus_columns,
    }
    return table_payload


def build_file_registry_payload() -> dict[str, Any]:
    datasets = _get_datasets()
    dataset_order = _get_dataset_order()
    active_dataset_key = _get_active_dataset_key()

    files: list[dict[str, Any]] = []
    for dataset_key in dataset_order:
        record = datasets.get(dataset_key)
        if not record:
            continue
        summary = dict(record.get("summary") or {})
        files.append(
            {
                "dataset_key": dataset_key,
                "name": record.get("name"),
                "mention": record.get("mention"),
                "rows": summary.get("rows"),
                "columns": summary.get("columns"),
                "has_chart": bool(record.get("chart")),
                "transformations": len(summary.get("transformations") or []),
                "is_active": dataset_key == active_dataset_key,
            }
        )

    return {
        "count": len(files),
        "limit": FILE_LIMIT,
        "can_add": len(files) < FILE_LIMIT,
        "active_dataset_key": active_dataset_key,
        "files": files,
    }


def _table_display_context(
    table_payload: Optional[dict[str, Any]],
    *,
    dataset_name: Optional[str] = None,
    mention: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if not table_payload:
        return None

    columns = [str(column) for column in (table_payload.get("columns") or [])]
    preview_rows = int(table_payload.get("preview_rows") or len(table_payload.get("rows") or []))
    total_rows = int(table_payload.get("total_rows") or preview_rows)
    total_columns = int(table_payload.get("total_columns") or len(columns))
    current_page = int(table_payload.get("page") or 1)
    total_pages = int(table_payload.get("total_pages") or 1)
    page_size = int(table_payload.get("page_size") or max(preview_rows, 1))
    row_start = int(table_payload.get("page_row_start") or 0)
    row_end = int(table_payload.get("page_row_end") or 0)
    highlighted = [str(column) for column in (table_payload.get("highlight_columns") or []) if column in columns]
    visible_prefix = columns[:8]
    visible_text = ", ".join(visible_prefix) if visible_prefix else "No visible columns recorded"
    if len(columns) > len(visible_prefix):
        visible_text += ", …"

    file_prefix = f"{dataset_name}" if dataset_name else "the active dataset"
    mention_text = f" (@{mention})" if mention else ""

    prompt_lines = [
        f"The displayed table belongs to {file_prefix}{mention_text}.",
        f"Visible table window: rows {row_start}-{row_end} of {total_rows}, page {current_page} of {total_pages}, {page_size} rows per page.",
        f"Visible columns in order: {visible_text}.",
    ]
    if highlighted:
        prompt_lines.append(
            "Focused columns currently highlighted in the table: " + ", ".join(highlighted) + "."
        )

    return {
        "kind": "table",
        "title": f"Table preview for {file_prefix}",
        "description": f"Table page {current_page} of {total_pages} with rows {row_start}-{row_end} and {total_columns} columns.",
        "details": [
            f"Dataset: {file_prefix}{mention_text}.",
            f"Total rows: {total_rows}.",
            f"Total columns: {total_columns}.",
            f"Current page: {current_page} of {total_pages}.",
            f"Visible row window: {row_start}-{row_end}.",
            f"Rows per page: {page_size}.",
        ],
        "prompt_lines": prompt_lines,
    }


def _chart_display_context(
    chart_payload: Optional[dict[str, Any]],
    *,
    dataset_name: Optional[str] = None,
    mention: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if not chart_payload:
        return None

    chart_type = str(chart_payload.get("chart_type") or "chart").strip().lower()
    x = chart_payload.get("x")
    y = chart_payload.get("y")
    aggregation = chart_payload.get("aggregation")
    top_n = chart_payload.get("top_n")
    color = chart_payload.get("color")
    title = str(chart_payload.get("title") or f"{chart_type.title()} chart").strip()
    file_prefix = f"{dataset_name}" if dataset_name else "the active dataset"
    mention_text = f" (@{mention})" if mention else ""

    description_parts = [chart_type.title(), f"for {file_prefix}{mention_text}"]
    if x:
        description_parts.append(f"x = {x}")
    if y:
        description_parts.append(f"y = {y}")
    description = ", ".join(description_parts)

    prompt_lines = [
        f"The displayed chart belongs to {file_prefix}{mention_text}.",
        f"Active chart title: {title}.",
        f"Chart type: {chart_type}.",
    ]
    if x:
        prompt_lines.append(f"X axis or category column: {x}.")
    if y:
        prompt_lines.append(f"Y axis or value column: {y}.")
    if aggregation:
        prompt_lines.append(f"Aggregation applied before plotting: {aggregation}.")
    if color:
        prompt_lines.append(f"Color or grouping column: {color}.")
    if top_n:
        prompt_lines.append(f"Top-N filter applied: {top_n}.")

    details = [
        f"Dataset: {file_prefix}{mention_text}.",
        f"Chart type: {chart_type}.",
    ]
    if x:
        details.append(f"X axis or category: {x}.")
    if y:
        details.append(f"Y axis or value: {y}.")
    if aggregation:
        details.append(f"Aggregation: {aggregation}.")
    if color:
        details.append(f"Grouped or colored by: {color}.")
    if top_n:
        details.append(f"Top-N filter applied: {top_n}.")

    return {
        "kind": "chart",
        "title": title,
        "description": description,
        "details": details,
        "prompt_lines": prompt_lines,
    }


def build_workspace_context(ui_state: Optional[dict[str, Any]]) -> dict[str, Any]:
    state = dict(ui_state or {})
    active_view = str(state.get("active_view") or "").strip().lower()
    file_registry = dict(state.get("file_registry") or {})
    active_file = None
    other_files: list[str] = []
    for file_item in file_registry.get("files") or []:
        if file_item.get("is_active"):
            active_file = file_item
        else:
            other_files.append(f"{file_item.get('name')} (@{file_item.get('mention')})")

    dataset_name = state.get("dataset_name") or (active_file or {}).get("name")
    mention = (active_file or {}).get("mention")
    table_context = _table_display_context(state.get("table"), dataset_name=dataset_name, mention=mention)
    chart_context = _chart_display_context(state.get("chart"), dataset_name=dataset_name, mention=mention)

    if active_view == "chart" and chart_context:
        current_display = chart_context
        alternate_display = table_context
    elif active_view == "table" and table_context:
        current_display = table_context
        alternate_display = chart_context
    else:
        current_display = table_context or chart_context
        alternate_display = chart_context if current_display and current_display.get("kind") == "table" else table_context
        if current_display:
            active_view = str(current_display.get("kind") or active_view)

    prompt_lines: list[str] = []
    files = file_registry.get("files") or []
    if files:
        prompt_lines.append(
            "Registered CSV files in this session: "
            + "; ".join(
                f"{file_item.get('name')} (@{file_item.get('mention')})"
                for file_item in files
            )
            + "."
        )
    else:
        prompt_lines.append("There are no CSV files registered in this session yet.")

    if active_file:
        prompt_lines.append(
            f"Current active dataset: {active_file.get('name')} (@{active_file.get('mention')})."
        )
    if other_files:
        prompt_lines.append("Other available datasets: " + "; ".join(other_files) + ".")

    if current_display:
        prompt_lines.append(f"Current active view in the left pane: {current_display['kind']}.")
        prompt_lines.extend(current_display.get("prompt_lines") or [])
    else:
        prompt_lines.append("There is no table or chart currently displayed in the left pane.")

    if alternate_display:
        prompt_lines.append(f"Other available view for the active dataset: {alternate_display['title']}.")

    summary = dict(state.get("summary") or {})
    transformations = list(summary.get("transformations") or [])[-5:]
    if transformations:
        prompt_lines.append(
            "Recent user-requested dataframe transformations on the active dataset: "
            + "; ".join(transformations)
            + "."
        )

    return {
        "active_view": active_view or None,
        "active_file": active_file,
        "current_display": current_display,
        "alternate_display": alternate_display,
        "transformations": transformations,
        "prompt_context": "\n".join(prompt_lines),
    }


def build_public_ui_state(
    *,
    status_override: Optional[str] = None,
    active_view_override: Optional[str] = None,
) -> dict[str, Any]:
    previous_ui_state = dict(cl.user_session.get("ui_state") or {})
    datasets = _get_datasets()
    active_dataset_key = _get_active_dataset_key()
    file_registry = build_file_registry_payload()

    status = status_override or previous_ui_state.get("status") or "Upload up to five CSV files to begin."
    desired_view = str(
        active_view_override
        or cl.user_session.get("active_view")
        or previous_ui_state.get("active_view")
        or "table"
    ).strip().lower()
    if desired_view not in {"table", "chart"}:
        desired_view = "table"

    if not active_dataset_key or active_dataset_key not in datasets:
        return {
            "dataset_name": None,
            "summary": None,
            "table": None,
            "chart": None,
            "active_view": "table",
            "status": status,
            "file_registry": file_registry,
        }

    record = datasets[active_dataset_key]
    table_payload = _build_table_payload_for_record(record)
    datasets[active_dataset_key] = record
    _set_datasets(datasets)
    chart_payload = record.get("chart")
    if desired_view == "chart" and not chart_payload:
        desired_view = "table"

    return {
        "dataset_name": record.get("name"),
        "summary": record.get("summary"),
        "table": table_payload,
        "chart": chart_payload,
        "active_view": desired_view,
        "status": status,
        "file_registry": file_registry,
    }


async def sync_ui(*, status: Optional[str] = None, active_view: Optional[str] = None) -> None:
    ui_state = build_public_ui_state(status_override=status, active_view_override=active_view)
    cl.user_session.set("active_view", ui_state.get("active_view") or "table")
    cl.user_session.set("ui_state", ui_state)
    cl.user_session.set("workspace_context", build_workspace_context(ui_state))
    await publish_ui_state()


async def initialize_empty_state() -> None:
    cl.user_session.set("datasets", {})
    cl.user_session.set("dataset_order", [])
    cl.user_session.set("active_dataset_key", None)
    cl.user_session.set("active_view", "table")
    cl.user_session.set("ui_state", {})
    cl.user_session.set("workspace_context", {})
    await sync_ui(status="Upload up to five CSV files to begin.", active_view="table")


def build_agent_request(
    user_request: str,
    *,
    target_record: dict[str, Any],
    explicit_reference: bool,
    multiple_references: Optional[list[str]] = None,
) -> str:
    workspace_context = cl.user_session.get("workspace_context") or build_workspace_context(
        cl.user_session.get("ui_state") or {}
    )
    prompt_context = workspace_context.get("prompt_context") or "There is no current table or chart context available."

    target_note = (
        f"Selected dataset for this request: {target_record['name']} (@{target_record['mention']}). "
        "Use only this dataframe when calling tools."
    )
    reference_note = (
        "The app selected this dataset because the user explicitly referenced it with @filename syntax."
        if explicit_reference
        else "The app selected the currently active dataset because the user did not specify an @filename."
    )
    extra_note = ""
    if multiple_references:
        extra_note = (
            "The user mentioned multiple files in one message. For tool-based analysis, "
            f"the app chose the first referenced file: {target_record['name']}. "
            f"Other referenced files were: {', '.join(multiple_references)}."
        )

    return (
        "Application workspace context (supplied by the app, not by the user):\n"
        "The user is working inside a split-screen multi-CSV exploration app. Use the workspace context below to resolve references such as 'this chart', 'that table', 'show it again', or 'use the highlighted column'.\n\n"
        f"{prompt_context}\n\n"
        f"{target_note}\n"
        f"{reference_note}\n"
        f"{extra_note}\n\n"
        "User request:\n"
        f"{user_request}"
    )


async def build_dataframe_agent(dataset_key: str, df):
    model = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

    @tool
    async def show_cleaned_table(rows: Optional[int] = None) -> str:
        """Display the cleaned dataframe preview in the left pane. Use this whenever the user asks to see rows, a preview, the cleaned data, or the table for the selected dataset."""
        datasets, record = _get_dataset_or_error(dataset_key)
        current_table_state = dict(record.get("table_state") or {})
        requested_rows = _coerce_table_page_size(rows, default=current_table_state.get("page_size", 25))

        async with cl.Step(name="show_cleaned_table", type="tool", show_input="json") as step:
            step.input = format_step_payload({"rows": requested_rows, "dataset": record.get("name")})
            step.output = "Refreshing the cleaned table preview in the left pane…"
            await step.update()

            record["table_state"] = {
                "page_size": requested_rows,
                "page": 1,
                "focus_columns": list(current_table_state.get("focus_columns") or []),
            }
            datasets[dataset_key] = record
            _set_datasets(datasets)
            _set_active_dataset_key(dataset_key)
            await sync_ui(
                status=(
                    f"Showing {record['name']} as a cleaned table with {requested_rows} rows per page."
                ),
                active_view="table",
            )
            table_payload = dict((cl.user_session.get("ui_state") or {}).get("table") or {})
            result = (
                f"Displayed **{record['name']}** as a cleaned table in the left pane. "
                f"The current page covers rows {table_payload.get('page_row_start', 0)}-{table_payload.get('page_row_end', 0)} "
                f"out of {table_payload.get('total_rows', 0)} with {table_payload.get('page_size', requested_rows)} rows per page."
            )
            step.output = result
            return result

    @tool(args_schema=RenderPlotInput)
    async def render_plot(
        chart_type: str,
        x: str,
        y: Optional[str] = None,
        color: Optional[str] = None,
        aggregation: Optional[str] = None,
        top_n: Optional[int] = None,
        sort_desc: bool = True,
        title: Optional[str] = None,
    ) -> str:
        """Render a Plotly chart in the left pane for the selected dataset. Pass only dataframe-oriented arguments such as chart_type, x, y, color, aggregation, top_n, and title; do not pass a prebuilt Plotly figure or figure JSON."""
        datasets, record = _get_dataset_or_error(dataset_key)
        step_payload = {
            "dataset": record.get("name"),
            "chart_type": chart_type,
            "x": x,
            "y": y,
            "color": color,
            "aggregation": aggregation,
            "top_n": top_n,
            "sort_desc": sort_desc,
            "title": title,
        }

        async with cl.Step(name="render_plot", type="tool", show_input="json") as step:
            step.input = format_step_payload(step_payload)
            step.output = "Building the Plotly figure and syncing the left workspace…"
            await step.update()

            plot_payload = build_plot_payload(
                df=record["df"],
                chart_type=chart_type,
                x=x,
                y=y,
                color=color,
                aggregation=aggregation,
                top_n=top_n,
                sort_desc=sort_desc,
                title=title,
            )
            record["chart"] = plot_payload
            datasets[dataset_key] = record
            _set_datasets(datasets)
            _set_active_dataset_key(dataset_key)
            await sync_ui(
                status=f"Rendered a {chart_type} chart for {record['name']}.",
                active_view="chart",
            )
            result = (
                f"Rendered a **{chart_type}** chart for **{record['name']}** in the left pane "
                f"using x='{x}' and y='{plot_payload.get('y')}'."
            )
            step.output = result
            return result

    @tool(args_schema=PreprocessDataFrameInput)
    async def preprocess_dataframe(
        target_column: list[str], expression: str, preview_rows: Optional[int] = None
    ) -> str:
        """Create or update one or more dataframe columns on the selected dataset as a preprocessing step and immediately refresh the left pane. Use this for requests like computing differences, sums, ratios, flags, unit conversions, splits, or other derived columns before further analysis or plotting."""
        datasets, record = _get_dataset_or_error(dataset_key)
        current_table_state = dict(record.get("table_state") or {})
        requested_rows = _coerce_table_page_size(preview_rows, default=current_table_state.get("page_size", 25))
        step_payload = {
            "dataset": record.get("name"),
            "target_column": target_column,
            "expression": expression,
            "preview_rows": requested_rows,
        }

        async with cl.Step(name="preprocess_dataframe", type="tool", show_input="json") as step:
            step.input = format_step_payload(step_payload)
            step.output = "Evaluating the preprocessing expression and refreshing the dataframe preview…"
            await step.update()

            normalized_targets = normalize_column_names(target_column)[0]
            computed_outputs, normalized_expression = evaluate_dataframe_expression_outputs(
                record["df"],
                expression,
                expected_outputs=len(normalized_targets),
            )

            for normalized_target, computed_series in zip(normalized_targets, computed_outputs):
                record["df"][normalized_target] = computed_series

            transformation_history = list(record.get("transformation_history") or [])
            if len(normalized_targets) == 1:
                transformation_history.append(
                    f"Created or updated `{normalized_targets[0]}` as `{normalized_expression}`"
                )
            else:
                transformation_history.append(
                    "Created or updated "
                    + ", ".join(f"`{column}`" for column in normalized_targets)
                    + f" as `{normalized_expression}`"
                )
            record["transformation_history"] = transformation_history[-12:]
            record["summary"] = refresh_summary_payload(
                dataset_name=record["name"],
                df=record["df"],
                existing_summary=record.get("summary"),
                transformations=record["transformation_history"],
            )
            record["chart"] = None
            record["table_state"] = {
                "page_size": requested_rows,
                "page": 1,
                "focus_columns": normalized_targets,
            }
            record["agent"] = await build_dataframe_agent(dataset_key, record["df"])
            datasets[dataset_key] = record
            _set_datasets(datasets)
            _set_active_dataset_key(dataset_key)
            await sync_ui(
                status=(
                    f"Updated {record['name']} and surfaced {', '.join(normalized_targets)} in the table preview."
                ),
                active_view="table",
            )

            non_null_details = ", ".join(
                f"`{column}`: {int(record['df'][column].notna().sum())} non-null"
                for column in normalized_targets
            )
            if len(normalized_targets) == 1:
                result = (
                    f"Created or updated `{normalized_targets[0]}` in **{record['name']}** using `{normalized_expression}`. "
                    f"The left pane has been refreshed, with `{normalized_targets[0]}` moved to the front of the table preview. "
                    f"It currently has {int(record['df'][normalized_targets[0]].notna().sum())} non-null values across {len(record['df'])} rows."
                )
            else:
                result = (
                    "Created or updated "
                    + ", ".join(f"`{column}`" for column in normalized_targets)
                    + f" in **{record['name']}** using `{normalized_expression}`. "
                    + "The left pane has been refreshed, with those columns moved to the front of the table preview and highlighted together. "
                    + f"Non-null counts — {non_null_details}."
                )
            step.output = result
            return result

    agent_prefix = """
You are an embedded dataframe analyst inside a multi-CSV exploration application.
You have access to one cleaned pandas DataFrame named `df`, representing the single selected dataset for this request.

Rules:
1. Use `show_cleaned_table` whenever the user asks to see a preview, rows, the cleaned data, or a table. If they specify how many rows to show, pass that as the rows-per-page value.
2. Use `render_plot` whenever the user asks for a chart, plot, graph, visualization, distribution, trend, or comparison for the selected dataset.
3. Use `preprocess_dataframe` whenever the user asks to create, update, engineer, derive, manipulate, or preprocess dataframe columns before analysis. This includes differences, sums, ratios, flags, bins, unit conversions, splits, or similar feature engineering.
4. When calling `preprocess_dataframe`, pass only the target column name(s) and the right-hand-side expression. For a single derived column, use target_column=['tpd_delta'] and expression='out_tpd - out_tpd2'. For multiple derived columns, use a list of names and a tuple/list expression with the same number of outputs, such as target_column=['ln_nm', 'lp_nm'] and expression='(out_ln * 1e9, out_lp * 1e9)'. Never include assignment syntax and never send full Python statements.
5. When calling `render_plot`, pass only named dataframe arguments like `chart_type`, `x`, `y`, `color`, `aggregation`, `top_n`, and `title`. Never pass a full Plotly figure, `data`/`layout` object, or JSON blob.
6. Only reference columns that exist exactly as named in `df`, except for a new target column when preprocessing.
7. Use `aggregation='count'` for frequency-style bar or pie charts.
8. After you update the left pane, briefly explain what was shown and mention one or two notable takeaways.
9. If a request is ambiguous, pick the most defensible transformation or chart for the selected dataset and state your assumption.
10. Each request also includes a workspace-context block describing what is currently displayed on the left pane and which CSV file is active. Use that context to resolve follow-up references such as 'this chart', 'that table', 'show it again', or 'use the highlighted column'.
""".strip()

    return create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_type="tool-calling",
        prefix=agent_prefix,
        include_df_in_prompt=True,
        number_of_head_rows=8,
        max_iterations=8,
        verbose=False,
        allow_dangerous_code=True,
        extra_tools=[show_cleaned_table, render_plot, preprocess_dataframe],
    )


async def _create_dataset_record(uploaded_file: Any) -> dict[str, Any]:
    datasets = _get_datasets()
    existing_names = {record.get("name") for record in datasets.values() if record.get("name")}
    existing_aliases = {record.get("mention") for record in datasets.values() if record.get("mention")}

    display_name = _make_unique_display_name(uploaded_file.name, existing_names)
    mention = _make_unique_reference_alias(display_name, existing_aliases)

    raw_df = load_csv(uploaded_file.path)
    cleaned_df, cleaning_report = clean_dataframe(raw_df)
    if cleaned_df.empty:
        raise ValueError("The CSV became empty after preprocessing. Please upload a file with usable rows and columns.")

    summary = build_summary_payload(display_name, cleaning_report)
    summary["transformations"] = []
    dataset_key = f"csv_{uuid.uuid4().hex[:10]}"
    record: dict[str, Any] = {
        "key": dataset_key,
        "name": display_name,
        "mention": mention,
        "aliases": _build_reference_aliases(display_name, mention),
        "df": cleaned_df,
        "summary": summary,
        "chart": None,
        "table_state": {"page_size": 25, "page": 1, "focus_columns": []},
        "transformation_history": [],
    }
    record["agent"] = await build_dataframe_agent(dataset_key, cleaned_df)
    return record


async def register_uploaded_files(files: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    datasets = _get_datasets()
    dataset_order = _get_dataset_order()
    remaining_capacity = max(0, FILE_LIMIT - len(dataset_order))

    successes: list[dict[str, Any]] = []
    failures: list[str] = []
    if remaining_capacity <= 0:
        return successes, [f"The workspace already has {FILE_LIMIT} CSV files. Delete one before uploading another."]

    selected_files = list(files)[:remaining_capacity]
    skipped_count = max(0, len(list(files)) - len(selected_files))

    for uploaded_file in selected_files:
        try:
            record = await _create_dataset_record(uploaded_file)
            datasets[record["key"]] = record
            dataset_order.append(record["key"])
            successes.append(record)
        except Exception as exc:
            failures.append(f"`{uploaded_file.name}` — {exc}")

    if skipped_count:
        failures.append(
            f"Skipped {skipped_count} additional file(s) because the workspace limit is {FILE_LIMIT}."
        )

    _set_datasets(datasets)
    _set_dataset_order(dataset_order)
    if successes:
        _set_active_dataset_key(successes[-1]["key"])
        cl.user_session.set("active_view", "table")

    return successes, failures


async def prompt_for_csv_upload(*, source: str) -> None:
    current_count = len(_get_dataset_order())
    remaining_capacity = FILE_LIMIT - current_count
    if remaining_capacity <= 0:
        await sync_ui(status=f"The workspace already has {FILE_LIMIT} files. Delete one to add another.")
        if source != "window":
            await cl.Message(content=f"The workspace is full. Delete one of the existing {FILE_LIMIT} CSV files before uploading another.").send()
        return

    await sync_ui(status="Waiting for CSV upload…")
    files = await cl.AskFileMessage(
        content=(
            f"Upload one or more CSV files. You can keep up to {FILE_LIMIT} files in this workspace, "
            f"and you currently have room for {remaining_capacity} more."
        ),
        accept=CSV_ACCEPT,
        max_files=remaining_capacity,
        timeout=180,
    ).send()

    if not files:
        await sync_ui(status="CSV upload was canceled. You can add files anytime from the workspace.")
        return

    successes, failures = await register_uploaded_files(files)
    if successes:
        active_record = successes[-1]
        await sync_ui(
            status=f"Registered {len(successes)} CSV file(s). Active file: {active_record['name']}.",
            active_view="table",
        )
        message_lines = [
            f"Registered **{len(successes)}** CSV file(s). New active file: **{active_record['name']}**."
        ]
        for record in successes:
            summary = dict(record.get("summary") or {})
            message_lines.append(
                f"- **{record['name']}** → use `@{record['mention']}` · {summary.get('rows', 0)} rows · {summary.get('columns', 0)} columns"
            )
        if failures:
            message_lines.append("\nSome files could not be loaded:")
            message_lines.extend([f"- {item}" for item in failures])
        if len(_get_dataset_order()) >= FILE_LIMIT:
            message_lines.append(f"\nThe workspace is full at {FILE_LIMIT} files. Delete one to upload another.")
        await cl.Message(content="\n".join(message_lines)).send()
    else:
        await sync_ui(status="No CSV files were added.")
        if failures:
            await cl.Message(
                content="I couldn't load those CSV files:\n" + "\n".join(f"- {item}" for item in failures)
            ).send()


async def activate_dataset(dataset_key: str, *, status: Optional[str] = None) -> None:
    datasets, record = _get_dataset_or_error(dataset_key)
    _set_datasets(datasets)
    _set_active_dataset_key(dataset_key)
    await sync_ui(
        status=status or f"Viewing {record['name']} in the table workspace.",
        active_view="table",
    )


async def delete_dataset(dataset_key: str) -> None:
    datasets = _get_datasets()
    dataset_order = _get_dataset_order()
    record = datasets.get(dataset_key)
    if not record:
        return

    deleted_name = record.get("name") or "the selected CSV"
    deleted_index = dataset_order.index(dataset_key) if dataset_key in dataset_order else -1

    datasets.pop(dataset_key, None)
    dataset_order = [key for key in dataset_order if key != dataset_key]
    _set_datasets(datasets)
    _set_dataset_order(dataset_order)

    if not dataset_order:
        _set_active_dataset_key(None)
        cl.user_session.set("active_view", "table")
        await sync_ui(status=f"Removed {deleted_name}. The workspace is now empty.", active_view="table")
        return

    next_index = min(max(deleted_index, 0), len(dataset_order) - 1)
    next_dataset_key = dataset_order[next_index]
    _set_active_dataset_key(next_dataset_key)
    await sync_ui(status=f"Removed {deleted_name}.", active_view="table")


def parse_window_message(message: str) -> tuple[str, dict[str, Any]]:
    if message == "SYNC_VIEW":
        return "SYNC_VIEW", {}
    try:
        parsed = json.loads(message)
    except Exception:
        return message, {}
    if not isinstance(parsed, dict):
        return message, {}
    payload = parsed.get("payload")
    return str(parsed.get("type") or ""), payload if isinstance(payload, dict) else {}


@cl.on_chat_start
async def on_chat_start() -> None:
    await initialize_empty_state()
    await cl.Message(
        content=(
            "This app can hold up to **five CSV files** in one session. "
            "Use the left workspace to switch files or delete them, and mention a file in chat with `@filename` style references such as `@sales_2025` or `@\"Sales Q1.csv\"`."
        )
    ).send()
    await prompt_for_csv_upload(source="chat_start")


@cl.on_window_message
async def on_window_message(message: str) -> None:
    message_type, payload = parse_window_message(message)

    if message_type == "SYNC_VIEW":
        await sync_ui()
        return

    if message_type == "REQUEST_UPLOAD_FILES":
        await prompt_for_csv_upload(source="window")
        return

    if message_type == "ACTIVE_DATASET_CHANGED":
        dataset_key = str(payload.get("dataset_key") or "").strip()
        if not dataset_key:
            return
        try:
            await activate_dataset(dataset_key)
        except Exception:
            return
        return

    if message_type == "DELETE_DATASET":
        dataset_key = str(payload.get("dataset_key") or "").strip()
        if not dataset_key:
            return
        await delete_dataset(dataset_key)
        return

    if message_type == "ACTIVE_VIEW_CHANGED":
        requested_view = str(payload.get("active_view") or "").strip().lower()
        if requested_view not in {"table", "chart"}:
            return
        active_record = _active_dataset_record()
        if active_record is None:
            return
        if requested_view == "chart" and not active_record.get("chart"):
            return
        await sync_ui(active_view=requested_view)
        return

    if message_type == "TABLE_PAGE_CHANGED":
        active_key = _get_active_dataset_key()
        if not active_key:
            return
        datasets, record = _get_dataset_or_error(active_key)
        table_state = dict(record.get("table_state") or {})
        table_state["page"] = _coerce_requested_page(payload.get("page"), default=table_state.get("page", 1))
        record["table_state"] = table_state
        datasets[active_key] = record
        _set_datasets(datasets)
        await sync_ui(active_view="table")
        return

    if message_type == "CHART_OPTIONS_CHANGED":
        active_key = _get_active_dataset_key()
        if not active_key:
            return

        datasets, record = _get_dataset_or_error(active_key)
        try:
            options = ChartOptionsChangedInput.model_validate(payload)
            plot_payload = build_plot_payload(
                record["df"],
                chart_type=options.chart_type,
                x=options.x,
                y=options.y,
                color=options.color,
                aggregation=options.aggregation,
                top_n=options.top_n,
                sort_desc=options.sort_desc,
                title=options.title,
            )
        except (ValidationError, ValueError) as exc:
            await sync_ui(status=f"Failed to update chart options: {exc}", active_view="chart")
            return

        record["chart"] = plot_payload
        datasets[active_key] = record
        _set_datasets(datasets)
        await sync_ui(status=f"Updated chart options for {record['name']}.", active_view="chart")
        return

    if message_type == "TABLE_PAGE_SIZE_CHANGED":
        active_key = _get_active_dataset_key()
        if not active_key:
            return
        datasets, record = _get_dataset_or_error(active_key)
        table_state = dict(record.get("table_state") or {})
        table_state["page_size"] = _coerce_table_page_size(
            payload.get("page_size"), default=table_state.get("page_size", 25)
        )
        table_state["page"] = 1
        record["table_state"] = table_state
        datasets[active_key] = record
        _set_datasets(datasets)
        await sync_ui(
            status=(
                f"Updated the table pagination for {record['name']} to {table_state['page_size']} rows per page."
            ),
            active_view="table",
        )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    content = (message.content or "").strip()
    lower_content = content.lower()

    if lower_content in {"add csv", "add file", "upload csv", "upload more", "upload more csv", "add more csv"}:
        await prompt_for_csv_upload(source="chat")
        return

    if lower_content in {"restart", "reset", "clear workspace", "new csv", "load new csv"}:
        await initialize_empty_state()
        await prompt_for_csv_upload(source="chat")
        return

    if not _get_dataset_order():
        await cl.Message(
            content="No CSV files are registered yet. Upload one or more CSV files first."
        ).send()
        return

    resolved_keys, unresolved_tokens, cleaned_content = extract_dataset_mentions(content)
    if unresolved_tokens:
        await cl.Message(
            content=(
                "I couldn't match these file references: "
                + ", ".join(f"`{token}`" for token in unresolved_tokens)
                + "\n\nAvailable references: "
                + _available_reference_summary()
            )
        ).send()
        return

    multiple_reference_names: list[str] = []
    if len(resolved_keys) > 1:
        datasets = _get_datasets()
        multiple_reference_names = [
            str(datasets[key].get("name")) for key in resolved_keys[1:] if key in datasets
        ]
        first_record = datasets.get(resolved_keys[0])
        if first_record:
            await cl.Message(
                content=(
                    f"I can drive the left-pane tools against one CSV at a time. "
                    f"I'll use **{first_record['name']}** for this request because it was mentioned first."
                )
            ).send()

    target_dataset_key = resolved_keys[0] if resolved_keys else _get_active_dataset_key()
    if not target_dataset_key:
        await cl.Message(content="No active CSV file is selected yet. Upload a CSV first.").send()
        return

    try:
        datasets, target_record = _get_dataset_or_error(target_dataset_key)
    except Exception as exc:
        await cl.Message(content=f"I couldn't access that CSV file: {exc}").send()
        return

    if not cleaned_content:
        await activate_dataset(
            target_dataset_key,
            status=f"Selected {target_record['name']} as the active file.",
        )
        await cl.Message(
            content=(
                f"Switched the workspace to **{target_record['name']}**. "
                f"You can target it again with `@{target_record['mention']}`."
            )
        ).send()
        return

    agent = target_record.get("agent")
    if agent is None:
        await cl.Message(content="That CSV file is loaded, but its dataframe agent is unavailable.").send()
        return

    _set_active_dataset_key(target_dataset_key)
    analysis_status = f"Analyzing {target_record['name']}…"
    await sync_ui(status=analysis_status)

    try:
        callback = cl.LangchainCallbackHandler()
        agent_input = build_agent_request(
            cleaned_content,
            target_record=target_record,
            explicit_reference=bool(resolved_keys),
            multiple_references=multiple_reference_names,
        )
        async with cl.Step(name="analyze_dataframe_request", type="run", show_input=True) as step:
            step.input = content
            step.output = (
                "Planning the analysis, considering the current workspace view and selected CSV file, "
                "and invoking tools if needed…"
            )
            await step.update()

            result = await agent.ainvoke(
                {"input": agent_input},
                config={
                    "callbacks": [callback],
                    "run_name": "csv_agent_analysis",
                },
            )
            answer = result.get("output") if isinstance(result, dict) else str(result)
            step.output = answer

        ui_state = cl.user_session.get("ui_state") or {}
        if ui_state.get("status") == analysis_status:
            active_record = _active_dataset_record() or target_record
            await sync_ui(status=f"Ready. Active file: {active_record['name']}.")
        await cl.Message(content=answer).send()
    except Exception as exc:
        await sync_ui(status=f"Analysis failed: {exc}")
        await cl.Message(content=f"I ran into an analysis error: {exc}").send()
