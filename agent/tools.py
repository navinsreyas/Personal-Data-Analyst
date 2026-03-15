import json
import os

import pandas as pd
from smolagents import tool


@tool
def load_csv_data(file_path: str) -> str:
    """Load a CSV file and return its structure and contents as a JSON string.

    Returns a JSON object with keys:
      - rows:    list of row dicts (full dataset)
      - columns: list of column name strings
      - shape:   [row_count, col_count]
      - preview: per-column summary (top values for categoricals,
                 mean/min/max for numerics)

    Always call this tool first before any analysis step.

    Args:
        file_path: Absolute path to the CSV file to load
    """
    df = pd.read_csv(file_path, low_memory=False)

    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(50)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().sum() / max(len(sample), 1) > 0.8:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except Exception:
            pass

    ROW_LIMIT = 2000
    is_large  = len(df) > ROW_LIMIT

    preview = {}
    for col in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                non_null = df[col].dropna()
                preview[col] = {
                    "min": str(non_null.min()),
                    "max": str(non_null.max()),
                    "null_count": int(df[col].isna().sum()),
                }
            elif df[col].dtype == object:
                preview[col] = df[col].value_counts().head(5).to_dict()
            else:
                preview[col] = {
                    "mean": round(float(df[col].mean()), 4),
                    "min":  round(float(df[col].min()),  4),
                    "max":  round(float(df[col].max()),  4),
                    "null_count": int(df[col].isna().sum()),
                }
        except Exception:
            preview[col] = "could not preview"

    return json.dumps(
        {
            "rows":       df.to_dict("records"),
            "columns":    df.columns.tolist(),
            "shape":      list(df.shape),
            "preview":    preview,
            "is_sampled": False,
        },
        default=str,
    )


@tool
def calculate_statistics(data_json: str, column: str) -> str:
    """Calculate descriptive statistics for a numeric column.

    Returns count, mean, std, min, 25th/50th/75th percentiles, and max.
    Use this for questions about averages, distributions, or ranges.

    Args:
        data_json: JSON string as returned by load_csv_data
        column:    Name of the numeric column to analyse
    """
    data = json.loads(data_json)
    df   = pd.DataFrame(data["rows"])
    desc = df[column].describe().to_dict()
    return json.dumps({k: round(float(v), 4) for k, v in desc.items()})


@tool
def group_and_aggregate(
    data_json: str,
    group_column: str,
    value_column: str,
    agg_function: str
) -> str:
    """Group the dataset by one column and aggregate another.

    Use for: 'average sales by region', 'total revenue by product',
    'count of records by state'. Results sorted descending.

    Args:
        data_json:     JSON string as returned by load_csv_data
        group_column:  Column to group by (e.g. 'region', 'geolocation_state')
        value_column:  Column to aggregate (e.g. 'sales', 'units_sold')
        agg_function:  One of: 'mean', 'sum', 'count', 'min', 'max'
    """
    data   = json.loads(data_json)
    df     = pd.DataFrame(data["rows"])
    result = (
        df.groupby(group_column)[value_column]
        .agg(agg_function)
        .sort_values(ascending=False)
    )
    return json.dumps({
        str(k): round(float(v), 4) for k, v in result.items()
    })


@tool
def filter_rows(
    data_json: str,
    column: str,
    operator: str,
    value: float
) -> str:
    """Filter dataset rows based on a numeric condition.

    Returns filtered rows as JSON with 'rows' and 'count' keys.
    Use this before aggregating when you need only a data subset.

    Args:
        data_json: JSON string as returned by load_csv_data
        column:    Column to filter on
        operator:  One of: '>', '<', '>=', '<=', '==', '!='
        value:     Numeric threshold to compare against
    """
    data = json.loads(data_json)
    df   = pd.DataFrame(data["rows"])
    ops  = {
        ">":  df[column] >  value,
        "<":  df[column] <  value,
        ">=": df[column] >= value,
        "<=": df[column] <= value,
        "==": df[column] == value,
        "!=": df[column] != value,
    }
    mask     = ops.get(operator)
    filtered = df[mask] if mask is not None else df
    return json.dumps({
        "rows":  filtered.to_dict("records"),
        "count": len(filtered),
    })


@tool
def generate_chart(
    data_json: str,
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
) -> str:
    """Generate a chart from data and save it as a PNG file.

    Use this tool when the user asks for a trend, distribution,
    comparison chart, or when visualisation would help the answer.
    Always call this AFTER load_csv_data or group_and_aggregate.

    Returns a JSON string with:
      - chart_path: absolute path to the saved PNG file
      - message:    confirmation string

    Args:
        data_json:  JSON string as returned by load_csv_data
                    OR a JSON object with keys matching x_column/y_column
        chart_type: One of: 'bar', 'line', 'histogram', 'pie'
        x_column:   Column name for the x-axis (or category axis)
        y_column:   Column name for the y-axis (or value axis)
        title:      Chart title shown above the plot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(data_json)

    if "rows" in data:
        df = pd.DataFrame(data["rows"])
    else:
        df = pd.DataFrame(
            list(data.items()), columns=[x_column, y_column]
        )

    df = df.dropna(subset=[x_column, y_column])

    if len(df) > 20:
        df = df.head(20)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    if chart_type == "bar":
        ax.bar(
            df[x_column].astype(str),
            df[y_column],
            color="#4f8ef7",
            edgecolor="#0e1117",
        )
        plt.xticks(rotation=45, ha="right", color="white")
        ax.set_xlabel(x_column, color="white")
        ax.set_ylabel(y_column, color="white")

    elif chart_type == "line":
        ax.plot(
            df[x_column].astype(str),
            df[y_column],
            color="#4f8ef7",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.xticks(rotation=45, ha="right", color="white")
        ax.set_xlabel(x_column, color="white")
        ax.set_ylabel(y_column, color="white")
        ax.fill_between(
            range(len(df)), df[y_column],
            alpha=0.15, color="#4f8ef7"
        )

    elif chart_type == "histogram":
        ax.hist(
            df[y_column].dropna(),
            bins=20,
            color="#4f8ef7",
            edgecolor="#0e1117",
        )
        ax.set_xlabel(y_column, color="white")
        ax.set_ylabel("Count", color="white")

    elif chart_type == "pie":
        wedges, texts, autotexts = ax.pie(
            df[y_column],
            labels=df[x_column].astype(str),
            autopct="%1.1f%%",
            colors=plt.cm.Set2.colors,
        )
        for t in texts + autotexts:
            t.set_color("white")

    else:
        ax.bar(
            df[x_column].astype(str),
            df[y_column],
            color="#4f8ef7",
        )
        plt.xticks(rotation=45, ha="right", color="white")

    ax.set_title(title, fontsize=13, pad=12, color="white")
    plt.tight_layout()

    charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
    os.makedirs(charts_dir, exist_ok=True)

    import time as _time
    chart_filename = f"chart_{int(_time.time() * 1000)}.png"
    chart_path     = os.path.join(charts_dir, chart_filename)

    plt.savefig(chart_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return json.dumps({
        "chart_path": chart_path,
        "message":    f"Chart saved: {chart_filename}",
    })
