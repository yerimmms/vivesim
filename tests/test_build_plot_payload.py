import pandas as pd

from data_processing import build_plot_payload


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def test_build_plot_payload_includes_sort_desc_and_respects_descending_top_n():
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10, 30, 20],
        }
    )

    payload = build_plot_payload(
        df=df,
        chart_type="bar",
        x="category",
        y="value",
        top_n=2,
        sort_desc=True,
        aggregation=None,
    )

    assert payload["sort_desc"] is True
    assert payload["top_n"] == 2

    first_trace = payload["figure"]["data"][0]
    assert _as_list(first_trace["x"]) == ["B", "C"]
    assert first_trace["y"] is not None


def test_build_plot_payload_respects_ascending_top_n_when_sort_desc_false():
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10, 30, 20],
        }
    )

    payload = build_plot_payload(
        df=df,
        chart_type="bar",
        x="category",
        y="value",
        top_n=2,
        sort_desc=False,
        aggregation=None,
    )

    assert payload["sort_desc"] is False

    first_trace = payload["figure"]["data"][0]
    assert _as_list(first_trace["x"]) == ["A", "C"]
    assert first_trace["y"] is not None


def test_build_plot_payload_pie_defaults_to_count_when_y_is_omitted():
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "C"],
            "value": [1, 2, 3, 4],
        }
    )

    payload = build_plot_payload(
        df=df,
        chart_type="pie",
        x="category",
        y=None,
        aggregation=None,
    )

    assert payload["y"] == "count"
    assert payload["aggregation"] is None
