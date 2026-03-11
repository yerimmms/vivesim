from __future__ import annotations

import ast
import json
import keyword
import math
import operator
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    "-",
    "--",
    "—",
    "missing",
}

AGGREGATIONS = {"sum", "mean", "count", "median", "min", "max"}
CHART_TYPES = {"bar", "line", "scatter", "histogram", "box", "pie"}
_DF_BRACKET_ACCESS_RE = re.compile(r"""df\[\s*([\"'])(.+?)\1\s*\]""")
_DF_ATTR_ACCESS_RE = re.compile(r"\bdf\.([A-Za-z_][A-Za-z0-9_]*)\b")
_ASSIGNMENT_RE = re.compile(r"(?<![<>=!])=(?!=)")


def _dedupe_name(name: str, seen: set[str]) -> str:
    if name not in seen:
        seen.add(name)
        return name

    index = 2
    while f"{name}_{index}" in seen:
        index += 1
    unique_name = f"{name}_{index}"
    seen.add(unique_name)
    return unique_name


def normalize_column_names(columns: List[Any]) -> Tuple[List[str], Dict[str, str]]:
    seen: set[str] = set()
    normalized: List[str] = []
    mapping: Dict[str, str] = {}

    for raw in columns:
        original = str(raw)
        name = original.strip().lower()
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        name = name or "column"
        unique_name = _dedupe_name(name, seen)
        normalized.append(unique_name)
        if original != unique_name:
            mapping[original] = unique_name

    return normalized, mapping


def normalize_single_column_name(name: str) -> str:
    normalized, _ = normalize_column_names([name])
    return normalized[0]


def _standardize_string_series(series: pd.Series) -> pd.Series:
    standardized = series.astype("string").str.strip()
    if not len(standardized):
        return standardized

    lowered = standardized.str.lower()
    standardized = standardized.mask(lowered.isin(MISSING_TOKENS), pd.NA)
    return standardized


def _try_parse_boolean(series: pd.Series) -> Optional[pd.Series]:
    non_null = series.dropna().astype(str).str.strip().str.lower()
    if non_null.empty:
        return None

    truthy = {"true", "yes", "y", "1"}
    falsy = {"false", "no", "n", "0"}
    allowed = truthy | falsy
    if not set(non_null.unique()).issubset(allowed):
        return None

    mapped = series.astype("string").str.strip().str.lower().map(
        {**{v: True for v in truthy}, **{v: False for v in falsy}}
    )
    return mapped.astype("boolean")


def _try_parse_numeric(series: pd.Series) -> Optional[pd.Series]:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[$€,]", "", regex=True)
        .str.replace("%", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    success_rate = float(parsed.notna().sum()) / float(non_null.shape[0])
    if success_rate < 0.8:
        return None

    percent_ratio = float(non_null.str.endswith("%").sum()) / float(non_null.shape[0])
    if percent_ratio >= 0.8:
        parsed = parsed / 100.0

    non_null_parsed = parsed.dropna()
    if not non_null_parsed.empty and np.allclose(non_null_parsed, np.round(non_null_parsed), atol=1e-9):
        return parsed.astype("Int64")
    return parsed.astype("Float64")


def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    success_rate = float(parsed.notna().sum()) / float(non_null.shape[0])
    if success_rate < 0.8:
        return None
    return parsed


def infer_series_type(series: pd.Series) -> tuple[pd.Series, Optional[str]]:
    if (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        return series, None

    standardized = _standardize_string_series(series)

    parsed_boolean = _try_parse_boolean(standardized)
    if parsed_boolean is not None:
        return parsed_boolean, "boolean"

    parsed_numeric = _try_parse_numeric(standardized)
    if parsed_numeric is not None:
        return parsed_numeric, "numeric"

    parsed_datetime = _try_parse_datetime(standardized)
    if parsed_datetime is not None:
        return parsed_datetime, "datetime"

    return standardized, None


def _column_type_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_columns = [
        column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    datetime_columns = [
        column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])
    ]
    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns and column not in datetime_columns
    ]
    return numeric_columns, datetime_columns, categorical_columns


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "original_shape": tuple(df.shape),
        "empty_rows_removed": 0,
        "empty_columns_removed": 0,
        "duplicate_rows_removed": 0,
        "column_renames": {},
        "converted_columns": {},
    }

    cleaned = df.copy()

    normalized_columns, rename_mapping = normalize_column_names(list(cleaned.columns))
    cleaned.columns = normalized_columns
    report["column_renames"] = rename_mapping

    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = _standardize_string_series(cleaned[column])

    before_rows = len(cleaned)
    cleaned = cleaned.dropna(axis=0, how="all")
    report["empty_rows_removed"] = before_rows - len(cleaned)

    before_cols = cleaned.shape[1]
    cleaned = cleaned.dropna(axis=1, how="all")
    report["empty_columns_removed"] = before_cols - cleaned.shape[1]

    for column in cleaned.columns:
        converted, conversion_type = infer_series_type(cleaned[column])
        cleaned[column] = converted
        if conversion_type:
            report["converted_columns"][column] = conversion_type

    before_dupes = len(cleaned)
    cleaned = cleaned.drop_duplicates(ignore_index=True)
    report["duplicate_rows_removed"] = before_dupes - len(cleaned)

    report["final_shape"] = tuple(cleaned.shape)
    report["missing_cells"] = int(cleaned.isna().sum().sum())
    numeric_columns, datetime_columns, categorical_columns = _column_type_lists(cleaned)
    report["numeric_columns"] = numeric_columns
    report["datetime_columns"] = datetime_columns
    report["categorical_columns"] = categorical_columns

    return cleaned, report


def load_csv(path: str) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
    ]

    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                on_bad_lines="skip",
                **attempt,
            )
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_error = exc
    assert last_error is not None
    raise last_error


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return ts.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_table_payload(
    df: pd.DataFrame,
    rows: int = 25,
    priority_columns: Optional[List[str]] = None,
    highlight_columns: Optional[List[str]] = None,
    page: int = 1,
) -> dict[str, Any]:
    page_size = max(1, int(rows or 25))
    total_rows = int(len(df))
    total_pages = max(1, math.ceil(total_rows / page_size)) if total_rows else 1
    current_page = max(1, min(int(page or 1), total_pages))

    start_index = (current_page - 1) * page_size
    end_index = min(start_index + page_size, total_rows)
    preview = df.iloc[start_index:end_index].copy()

    ordered_columns = list(df.columns)
    if priority_columns:
        preferred = [column for column in priority_columns if column in df.columns]
        ordered_columns = preferred + [column for column in ordered_columns if column not in preferred]
    if ordered_columns:
        preview = preview.loc[:, ordered_columns]

    preview = preview.where(pd.notnull(preview), None)
    serialized_rows = [
        {column: _serialize_value(value) for column, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]
    row_numbers = list(range(start_index + 1, end_index + 1)) if total_rows else []

    return {
        "columns": list(preview.columns),
        "rows": serialized_rows,
        "preview_rows": len(preview),
        "total_rows": total_rows,
        "total_columns": int(df.shape[1]),
        "highlight_columns": [
            column for column in (highlight_columns or []) if column in preview.columns
        ],
        "page": current_page,
        "page_size": page_size,
        "total_pages": total_pages,
        "page_row_start": row_numbers[0] if row_numbers else 0,
        "page_row_end": row_numbers[-1] if row_numbers else 0,
        "row_numbers": row_numbers,
    }


def build_summary_payload(dataset_name: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "rows": report["final_shape"][0],
        "columns": report["final_shape"][1],
        "missing_cells": report["missing_cells"],
        "duplicate_rows_removed": report["duplicate_rows_removed"],
        "empty_rows_removed": report["empty_rows_removed"],
        "empty_columns_removed": report["empty_columns_removed"],
        "numeric_columns": report["numeric_columns"],
        "datetime_columns": report["datetime_columns"],
        "categorical_columns": report["categorical_columns"],
        "column_renames": report["column_renames"],
        "converted_columns": report["converted_columns"],
        "transformations": list(report.get("transformations") or []),
    }


def refresh_summary_payload(
    dataset_name: str,
    df: pd.DataFrame,
    existing_summary: Optional[dict[str, Any]] = None,
    transformations: Optional[list[str]] = None,
) -> dict[str, Any]:
    summary = dict(existing_summary or {})
    numeric_columns, datetime_columns, categorical_columns = _column_type_lists(df)
    summary.update(
        {
            "dataset_name": dataset_name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "numeric_columns": numeric_columns,
            "datetime_columns": datetime_columns,
            "categorical_columns": categorical_columns,
        }
    )
    summary.setdefault("duplicate_rows_removed", 0)
    summary.setdefault("empty_rows_removed", 0)
    summary.setdefault("empty_columns_removed", 0)
    summary.setdefault("column_renames", {})
    summary.setdefault("converted_columns", {})
    if transformations is not None:
        summary["transformations"] = list(transformations)
    else:
        summary.setdefault("transformations", [])
    return summary


def _validate_columns(df: pd.DataFrame, *columns: Optional[str]) -> None:
    missing = [column for column in columns if column and column not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(f"Unknown column(s): {', '.join(missing)}. Available columns: {available}")


def _scalar_numeric(value: Any, *, name: str) -> float | int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric scalar.")
    return value


def _series_round(value: Any, ndigits: Any = 0) -> Any:
    precision = int(_scalar_numeric(ndigits, name="round precision"))
    if isinstance(value, pd.Series):
        return value.round(precision)
    return round(value, precision)


def _series_abs(value: Any) -> Any:
    return value.abs() if isinstance(value, pd.Series) else abs(value)


def _series_clip(value: Any, lower: Any = None, upper: Any = None) -> Any:
    if isinstance(lower, np.generic):
        lower = lower.item()
    if isinstance(upper, np.generic):
        upper = upper.item()
    if isinstance(value, pd.Series):
        return value.clip(lower=lower, upper=upper)
    if lower is not None:
        value = max(value, lower)
    if upper is not None:
        value = min(value, upper)
    return value





def _coerce_optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    numeric = _scalar_numeric(value, name=name)
    integer_value = int(numeric)
    if integer_value != numeric:
        raise ValueError(f"{name} must be an integer.")
    return integer_value


class _SafeStringAccessor:
    def __init__(self, series: pd.Series):
        self._series = series.astype("string")

    def apply_subscript(self, key: Any) -> pd.Series:
        if isinstance(key, slice):
            return self._series.str.slice(key.start, key.stop, key.step)
        if isinstance(key, int):
            return self._series.str.get(key)
        raise ValueError(
            "Only integer indexing and slice syntax like column.str[:2] are supported on the .str accessor."
        )

_ALLOWED_EVAL_FUNCTIONS: dict[str, Any] = {
    "round": _series_round,
    "abs": _series_abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "exp": np.exp,
    "floor": np.floor,
    "ceil": np.ceil,
    "clip": _series_clip,
}


_ALLOWED_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


_ALLOWED_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}


_ALLOWED_COMPARATORS: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _safe_alias_for_column(index: int) -> str:
    return f"col_{index}"


def _rewrite_expression_with_aliases(df: pd.DataFrame, expression: str) -> tuple[str, dict[str, str]]:
    alias_by_column = {str(column): _safe_alias_for_column(index) for index, column in enumerate(df.columns)}

    def replace_backtick(match: re.Match[str]) -> str:
        column_name = match.group(1).strip()
        if column_name not in alias_by_column:
            raise ValueError(
                f"Unknown column '{column_name}' in preprocessing expression. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return alias_by_column[column_name]

    rewritten = re.sub(r"`([^`]+)`", replace_backtick, expression)

    reserved_names = set(_ALLOWED_EVAL_FUNCTIONS) | {"and", "or", "not", "True", "False", "None", "nan"}
    for column_name, alias in sorted(alias_by_column.items(), key=lambda item: len(item[0]), reverse=True):
        if column_name in reserved_names or keyword.iskeyword(column_name):
            continue
        rewritten = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(column_name)}(?![A-Za-z0-9_])",
            alias,
            rewritten,
        )

    return rewritten, alias_by_column




def _evaluate_subscript_key(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Slice):
        start = _coerce_optional_int(
            _evaluate_ast_expression(node.lower, context) if node.lower is not None else None,
            name="string slice start",
        )
        stop = _coerce_optional_int(
            _evaluate_ast_expression(node.upper, context) if node.upper is not None else None,
            name="string slice stop",
        )
        step = _coerce_optional_int(
            _evaluate_ast_expression(node.step, context) if node.step is not None else None,
            name="string slice step",
        )
        return slice(start, stop, step)

    value = _evaluate_ast_expression(node, context)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("String indexing only supports integers or slices.")
    return int(value)

def _evaluate_ast_expression(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_ast_expression(node.body, context)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)) or node.value is None:
            return node.value
        raise ValueError(f"Unsupported constant value {node.value!r} in preprocessing expression.")

    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        raise ValueError(f"Unknown name '{node.id}' in preprocessing expression.")

    if isinstance(node, ast.BinOp):
        operator_fn = _ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f"Operator '{type(node.op).__name__}' is not supported in preprocessing expressions.")
        return operator_fn(
            _evaluate_ast_expression(node.left, context),
            _evaluate_ast_expression(node.right, context),
        )

    if isinstance(node, ast.UnaryOp):
        operator_fn = _ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f"Unary operator '{type(node.op).__name__}' is not supported.")
        return operator_fn(_evaluate_ast_expression(node.operand, context))

    if isinstance(node, ast.BoolOp):
        values = [_evaluate_ast_expression(value, context) for value in node.values]
        if not values:
            raise ValueError("Boolean expressions must contain at least one value.")
        result = values[0]
        for value in values[1:]:
            if isinstance(node.op, ast.And):
                result = result & value
            elif isinstance(node.op, ast.Or):
                result = result | value
            else:
                raise ValueError(f"Boolean operator '{type(node.op).__name__}' is not supported.")
        return result

    if isinstance(node, ast.Compare):
        left = _evaluate_ast_expression(node.left, context)
        combined_result = None
        for operator_node, comparator in zip(node.ops, node.comparators):
            operator_fn = _ALLOWED_COMPARATORS.get(type(operator_node))
            if operator_fn is None:
                raise ValueError(f"Comparison operator '{type(operator_node).__name__}' is not supported.")
            right = _evaluate_ast_expression(comparator, context)
            current_result = operator_fn(left, right)
            combined_result = current_result if combined_result is None else (combined_result & current_result)
            left = right
        if combined_result is None:
            raise ValueError("Comparison expressions must contain at least one comparator.")
        return combined_result

    if isinstance(node, ast.Attribute):
        base_value = _evaluate_ast_expression(node.value, context)
        if isinstance(base_value, pd.Series) and node.attr == "str":
            return _SafeStringAccessor(base_value)
        raise ValueError(
            f"Attribute access '.{node.attr}' is not supported in preprocessing expressions. "
            "Only the pandas string accessor '.str' is currently allowed."
        )

    if isinstance(node, ast.Subscript):
        base_value = _evaluate_ast_expression(node.value, context)
        if isinstance(base_value, _SafeStringAccessor):
            key = _evaluate_subscript_key(node.slice, context)
            return base_value.apply_subscript(key)
        raise ValueError(
            "Subscript access is not supported in preprocessing expressions except for the pandas string accessor, "
            "for example column.str[:2]."
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to supported preprocessing functions are allowed.")
        function_name = node.func.id
        function = _ALLOWED_EVAL_FUNCTIONS.get(function_name)
        if function is None:
            raise ValueError(f"'{function_name}' is not a supported function.")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in preprocessing functions.")
        arguments = [_evaluate_ast_expression(argument, context) for argument in node.args]
        return function(*arguments)

    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_ast_expression(element, context) for element in node.elts)

    if isinstance(node, ast.List):
        return [_evaluate_ast_expression(element, context) for element in node.elts]

    raise ValueError(
        f"Unsupported syntax '{type(node).__name__}' in preprocessing expression. "
        "Use arithmetic, comparisons, boolean conditions, and supported functions only."
    )


def _safe_evaluate_expression(df: pd.DataFrame, expression: str) -> Any:
    rewritten_expression, alias_by_column = _rewrite_expression_with_aliases(df, expression)
    try:
        parsed = ast.parse(rewritten_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid preprocessing expression syntax: {exc.msg}") from exc

    context: dict[str, Any] = {alias: df[column_name] for column_name, alias in alias_by_column.items()}
    context.update(_ALLOWED_EVAL_FUNCTIONS)
    context["nan"] = np.nan
    return _evaluate_ast_expression(parsed, context)


def _normalize_eval_expression(expression: str) -> str:
    normalized = str(expression or "").strip()
    if not normalized:
        raise ValueError("The preprocessing expression cannot be empty.")

    lowered = normalized.lower()
    forbidden_fragments = ["__", "import ", "lambda", "open(", "exec(", "eval(", "subprocess", "os.", "sys."]
    if any(fragment in lowered for fragment in forbidden_fragments):
        raise ValueError("The preprocessing expression must be a simple dataframe expression, not arbitrary Python code.")
    if "@" in normalized:
        raise ValueError("External variables are not supported in preprocessing expressions.")
    if "\n" in normalized or ";" in normalized:
        raise ValueError("The preprocessing expression must stay on one line and cannot contain statement separators.")
    if _ASSIGNMENT_RE.search(normalized):
        raise ValueError("Pass only the right-hand side expression. Do not include an assignment like x = ...")

    normalized = _DF_BRACKET_ACCESS_RE.sub(lambda match: f"`{match.group(2).strip()}`", normalized)
    normalized = _DF_ATTR_ACCESS_RE.sub(lambda match: f"`{match.group(1)}`", normalized)
    return normalized


def _coerce_expression_output_to_series(value: Any, df: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("Each preprocessing output must resolve to exactly one column.")
        value = value.iloc[:, 0]

    if isinstance(value, pd.Series):
        result = value
    elif np.isscalar(value):
        result = pd.Series([value] * len(df), index=df.index)
    else:
        try:
            result = pd.Series(value, index=df.index)
        except Exception as exc:
            raise ValueError(
                "The preprocessing expression did not produce a scalar or a Series-like result."
            ) from exc

    if len(result) != len(df):
        raise ValueError(
            f"The preprocessing expression produced {len(result)} values for a dataframe with {len(df)} rows."
        )

    result = result.reindex(df.index)
    if pd.api.types.is_numeric_dtype(result):
        result = result.replace([np.inf, -np.inf], np.nan)

    converted, _ = infer_series_type(result)
    return converted



def evaluate_dataframe_expression_outputs(
    df: pd.DataFrame,
    expression: str,
    *,
    expected_outputs: Optional[int] = None,
) -> tuple[list[pd.Series], str]:
    normalized_expression = _normalize_eval_expression(expression)
    try:
        raw_result = _safe_evaluate_expression(df, normalized_expression)
    except Exception as exc:
        raise ValueError(f"Could not evaluate preprocessing expression '{expression}': {exc}") from exc

    if isinstance(raw_result, pd.DataFrame):
        outputs = [raw_result.iloc[:, index] for index in range(raw_result.shape[1])]
    elif isinstance(raw_result, (list, tuple)):
        outputs = list(raw_result)
    else:
        outputs = [raw_result]

    coerced_outputs = [_coerce_expression_output_to_series(value, df) for value in outputs]

    if expected_outputs is not None and len(coerced_outputs) != int(expected_outputs):
        raise ValueError(
            f"The preprocessing expression produced {len(coerced_outputs)} output column(s), "
            f"but {int(expected_outputs)} target column(s) were requested."
        )

    return coerced_outputs, normalized_expression



def evaluate_dataframe_expression(df: pd.DataFrame, expression: str) -> tuple[pd.Series, str]:
    outputs, normalized_expression = evaluate_dataframe_expression_outputs(
        df,
        expression,
        expected_outputs=1,
    )
    return outputs[0], normalized_expression


def build_plot_payload(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: Optional[str] = None,
    color: Optional[str] = None,
    aggregation: Optional[str] = None,
    title: Optional[str] = None,
    top_n: Optional[int] = None,
    sort_desc: bool = True,
) -> dict[str, Any]:
    chart_type = chart_type.lower().strip()
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unsupported chart type '{chart_type}'. Supported: {sorted(CHART_TYPES)}")

    aggregation = aggregation.lower().strip() if aggregation else None
    if aggregation and aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. Supported: {sorted(AGGREGATIONS)}"
        )

    _validate_columns(df, x, y, color)

    plot_df = df.copy()
    resolved_y = y

    if chart_type in {"bar", "line", "pie"} and (aggregation or chart_type == "pie"):
        if aggregation == "count" or (chart_type == "pie" and not y):
            plot_df = plot_df.groupby(x, dropna=False).size().reset_index(name="count")
            resolved_y = "count"
        else:
            if not y:
                raise ValueError(f"'{chart_type}' charts with aggregation require a y column.")
            plot_df = plot_df.groupby(x, dropna=False)[y].agg(aggregation or "sum").reset_index()
            resolved_y = y
        color = None

    if chart_type == "pie" and resolved_y is None:
        raise ValueError("Pie charts require either a y column or aggregation='count'.")

    if top_n and resolved_y and resolved_y in plot_df.columns:
        plot_df = plot_df.sort_values(resolved_y, ascending=not sort_desc).head(top_n)

    if not title:
        if chart_type in {"bar", "line", "pie"} and aggregation:
            title = f"{aggregation.title()} of {resolved_y} by {x}" if resolved_y else f"Count by {x}"
        elif chart_type == "histogram":
            title = f"Distribution of {x}"
        else:
            title = f"{chart_type.title()} chart of {resolved_y or x}"

    common_args = {"title": title}
    if color:
        common_args["color"] = color

    if chart_type == "bar":
        if not resolved_y:
            raise ValueError("Bar charts require a y column or an aggregation.")
        fig = px.bar(plot_df, x=x, y=resolved_y, **common_args)
    elif chart_type == "line":
        if not resolved_y:
            raise ValueError("Line charts require a y column or an aggregation.")
        fig = px.line(plot_df, x=x, y=resolved_y, **common_args)
    elif chart_type == "scatter":
        if not y:
            raise ValueError("Scatter charts require both x and y columns.")
        fig = px.scatter(plot_df, x=x, y=y, **common_args)
    elif chart_type == "histogram":
        fig = px.histogram(plot_df, x=x, **common_args)
    elif chart_type == "box":
        if not y:
            raise ValueError("Box plots require both x and y columns.")
        fig = px.box(plot_df, x=x, y=y, **common_args)
    else:  # pie
        fig = px.pie(plot_df, names=x, values=resolved_y, title=title)

    fig.update_layout(margin=dict(l=30, r=30, t=60, b=30), autosize=True)

    figure_payload = fig.to_plotly_json()
    layout_payload = figure_payload.get("layout")
    if isinstance(layout_payload, dict):
        layout_payload.pop("template", None)

    return {
        "chart_type": chart_type,
        "figure": json.loads(json.dumps(figure_payload, default=str)),
        "config": {"responsive": True, "displaylogo": False},
        "title": title,
        "x": x,
        "y": resolved_y,
        "color": color,
        "aggregation": aggregation,
        "top_n": top_n,
    }
