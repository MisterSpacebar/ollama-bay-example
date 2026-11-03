"""Server-side chart rendering/data-prep for dataset preview/exploration.

'hist' still renders a static matplotlib PNG (Streamlit has no built-in histogram widget).
'line'/'scatter'/'bar' instead prepare plain row data for Streamlit's native interactive
chart widgets (st.line_chart/st.bar_chart/st.scatter_chart), which support hover/zoom/pan
that a static image can't.
"""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MAX_POINTS = 5000


def _downsample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_POINTS:
        return df
    step = max(1, len(df) // MAX_POINTS)
    return df.iloc[::step]


def _is_year_like(col_name: str, series: pd.Series) -> bool:
    """Heuristic for 'this x column is a calendar year, not a continuous quantity' -- Vega-Lite
    (the renderer behind Streamlit's native charts) otherwise treats year as a plain large
    number and applies thousands-separator formatting (e.g. '2,023'), which looks wrong."""
    if "year" in col_name.lower():
        return True
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty or not (numeric == numeric.round()).all():
        return False
    return bool(numeric.between(1500, 2100).all())


def prepare_native_chart_data(
    df: pd.DataFrame, chart_type: str, y: list[str], x: str | None = None, group_by: str | None = None
) -> pd.DataFrame:
    """Build a small, plot-ready dataframe for line/scatter/bar charts (NOT hist -- that one
    still goes through make_chart/matplotlib). Mirrors the same aggregation/downsampling
    make_chart used to do directly, so the native widgets get the same well-behaved data."""
    if not y:
        raise ValueError("At least one y column is required")
    cols = [c for c in [x, group_by, *y] if c]
    plot_df = df[cols].copy()
    for col in y:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    if chart_type == "line":
        plot_df = _downsample(plot_df.sort_values(x) if x else plot_df)
        # If x repeats (e.g. 'year' with many daily rows per year), a raw line through every
        # row zigzags through the full y-range at each x and visually reads like a bar chart
        # instead of a trend -- aggregate (mean) per x/group_by combo first so each x value
        # maps to one point, which is what a "line over time" chart actually implies.
        if x and plot_df[x].duplicated().any():
            group_cols = [group_by, x] if group_by else [x]
            plot_df = plot_df.groupby(group_cols, as_index=False)[y].mean()
    elif chart_type == "scatter":
        if x is None:
            raise ValueError("scatter requires an x column")
        plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
        plot_df = _downsample(plot_df)
    elif chart_type == "bar":
        group_cols = [group_by, x] if group_by and x else [x] if x else None
        if group_cols:
            plot_df = plot_df.groupby(group_cols, as_index=False)[y].mean()
    else:
        raise ValueError(f"Unsupported native chart_type: {chart_type!r}. Use line, scatter, or bar.")

    plot_df = plot_df.dropna(subset=[c for c in [x, *y] if c])
    # Years are a discrete label, not a continuous quantity -- convert to a plain string
    # AFTER numeric sorting/aggregation above, so the axis shows '2023' instead of '2,023'.
    # Skip this for scatter, where x needs to stay numeric for a real position-based plot.
    if x and chart_type in ("line", "bar") and _is_year_like(x, plot_df[x]):
        plot_df[x] = plot_df[x].astype("Int64").astype(str)
    return plot_df


def make_chart(df: pd.DataFrame, chart_type: str, y: list[str], x: str | None = None, group_by: str | None = None) -> bytes:
    if not y:
        raise ValueError("At least one y column is required")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if chart_type == "line":
        plot_df = _downsample(df.sort_values(x) if x else df)
        # If x repeats (e.g. 'year' with many daily rows per year), a raw line through every
        # row zigzags through the full y-range at each x and visually reads like a bar chart
        # instead of a trend -- aggregate (mean) per x/group_by combo first so each x value
        # maps to one point, which is what a "line over time" chart actually implies.
        if x and plot_df[x].duplicated().any():
            group_cols = [group_by, x] if group_by else [x]
            numeric_y = plot_df[y].apply(pd.to_numeric, errors="coerce")
            plot_df = pd.concat([plot_df[group_cols], numeric_y], axis=1).groupby(group_cols, as_index=False).mean()
        if group_by:
            for key, grp in plot_df.groupby(group_by):
                for col in y:
                    ax.plot(grp[x] if x else grp.index, pd.to_numeric(grp[col], errors="coerce"), label=f"{key} · {col}")
        else:
            for col in y:
                ax.plot(plot_df[x] if x else plot_df.index, pd.to_numeric(plot_df[col], errors="coerce"), label=col)
        ax.set_xlabel(x or "row index")
        ax.legend(fontsize=8)
    elif chart_type == "scatter":
        if x is None:
            raise ValueError("scatter requires an x column")
        plot_df = _downsample(df)
        ax.scatter(pd.to_numeric(plot_df[x], errors="coerce"), pd.to_numeric(plot_df[y[0]], errors="coerce"), s=8, alpha=0.5)
        ax.set_xlabel(x)
    elif chart_type == "bar":
        if x:
            grouped = pd.to_numeric(df[y[0]], errors="coerce").groupby(df[x]).mean()
        else:
            grouped = pd.to_numeric(df[y[0]], errors="coerce")
        grouped.plot(kind="bar", ax=ax)
        ax.set_xlabel(x or "")
    elif chart_type == "hist":
        ax.hist(pd.to_numeric(df[y[0]], errors="coerce").dropna(), bins=30)
        ax.set_xlabel(y[0])
    else:
        raise ValueError(f"Unknown chart_type: {chart_type!r}. Use line, scatter, bar, or hist.")

    ax.set_ylabel(", ".join(y))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
