"""Streamlit dashboard for the Twitter/X sentiment tracker."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.aggregator import build_distribution_frame, build_timeseries_frame
from analytics.metrics import add_rolling_average, build_summary_stats
from app.config import load_config
from app.logger import configure_logging
from app.main import build_pipeline
from app.schemas import SearchParameters
from visualization.plots import build_sentiment_distribution_chart, build_sentiment_trend_chart


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@st.cache_resource
def load_dashboard_pipeline(config_path: str):
    """Create and cache the pipeline used by the dashboard."""
    config = load_config(config_path)
    configure_logging(config.logging.level, config.logging.format)
    pipeline = build_pipeline(config)
    return config, pipeline


def main() -> None:
    """Render the Streamlit UI."""
    st.set_page_config(page_title="Twitter/X Sentiment Tracker", layout="wide")
    config, pipeline = load_dashboard_pipeline(str(DEFAULT_CONFIG_PATH))
    st.title(config.dashboard.title)
    st.caption("Collect tweets with the X API, analyze sentiment, and explore trends.")

    with st.sidebar:
        st.header("Search")
        query = st.text_input("Keyword Query", value=config.search.keyword or "")
        hashtags_input = st.text_input(
            "Hashtags (comma-separated)",
            value=", ".join(config.search.hashtags),
        )
        default_start = config.search.start_date or (date.today() - timedelta(days=1))
        default_end = config.search.end_date or date.today()
        date_range = st.date_input("Date Range", value=(default_start, default_end))
        max_results = st.number_input(
            "Max Results",
            min_value=10,
            max_value=1000,
            value=config.search.max_results,
            step=10,
        )
        model_name = st.selectbox(
            "Sentiment Model",
            options=["vader", "ml"],
            index=0 if config.model.active_model == "vader" else 1,
        )
        granularity = st.selectbox("Trend Granularity", options=["hourly", "daily"], index=1)
        distribution_chart = st.selectbox("Distribution Chart", options=["pie", "bar"], index=0)
        run_button = st.button("Run Tracking Pipeline", use_container_width=True)

    if run_button:
        try:
            start_date, end_date = _normalize_date_range(date_range)
            search_parameters = SearchParameters(
                keyword=query or None,
                hashtags=[value.strip() for value in hashtags_input.split(",") if value.strip()],
                start_date=start_date,
                end_date=end_date,
                max_results=int(max_results),
            )
            result = pipeline.run(search_parameters=search_parameters, model_name=model_name)
            st.session_state["run_id"] = result.run_id
            st.session_state["result"] = result
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    run_id = st.session_state.get("run_id")
    result = st.session_state.get("result")

    if not run_id or not result:
        st.info("Run the pipeline to populate charts and metrics.")
        return

    summary = build_summary_stats(
        pipeline.repository.fetch_summary_stats(run_id),
        build_distribution_frame(pipeline.repository.fetch_sentiment_distribution(run_id)),
    )
    time_series = add_rolling_average(
        build_timeseries_frame(pipeline.repository.fetch_time_series(granularity, run_id), granularity)
    )
    distribution = build_distribution_frame(pipeline.repository.fetch_sentiment_distribution(run_id))
    recent_tweets = pd.DataFrame(pipeline.repository.fetch_recent_tweets(run_id, limit=100))

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Tweets", int(summary["total_tweets"]))
    metric_columns[1].metric("Average Sentiment", f'{summary["average_sentiment"]:.3f}')
    metric_columns[2].metric("Average Confidence", f'{summary["average_confidence"]:.2%}')
    metric_columns[3].metric("Dominant Sentiment", str(summary["dominant_sentiment"]).title())

    chart_columns = st.columns((2, 1))
    chart_columns[0].plotly_chart(build_sentiment_trend_chart(time_series), use_container_width=True)
    chart_columns[1].plotly_chart(
        build_sentiment_distribution_chart(distribution, chart_type=distribution_chart),
        use_container_width=True,
    )

    st.subheader("Run Summary")
    st.write(
        {
            "run_id": result.run_id,
            "model_name": result.model_name,
            "source_query": result.source_query,
            "collected_count": result.collected_count,
            "stored_count": result.stored_count,
            "raw_export_path": result.raw_export_path,
            "processed_export_path": result.processed_export_path,
        }
    )

    st.subheader("Recent Tweets")
    st.dataframe(recent_tweets, use_container_width=True)


def _normalize_date_range(date_range) -> tuple[date | None, date | None]:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        return date_range[0], date_range[1]
    if isinstance(date_range, list) and len(date_range) == 2:
        return date_range[0], date_range[1]
    if isinstance(date_range, date):
        return date_range, date_range
    return None, None


if __name__ == "__main__":
    main()
