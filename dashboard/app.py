"""Streamlit dashboard for the Twitter/X sentiment tracker."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.aggregator import (
    build_distribution_frame,
    build_run_history_frame,
    build_source_breakdown_frame,
    build_timeseries_frame,
)
from analytics.metrics import add_rolling_average, build_summary_stats
from app.config import load_config
from app.logger import configure_logging
from app.main import build_pipeline
from app.schemas import SearchParameters
from visualization.plots import (
    build_sentiment_distribution_chart,
    build_sentiment_trend_chart,
    build_source_breakdown_chart,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
BACKEND_OPTIONS = ["auto", "twscrape", "x_api", "news_rss", "snscrape"]


@st.cache_resource
def load_dashboard_pipeline(config_path: str):
    """Create and cache the pipeline used by the dashboard."""
    config = load_config(config_path)
    configure_logging(config.logging.level, config.logging.format)
    pipeline = build_pipeline(config)
    return config, pipeline


@st.cache_data(show_spinner=False)
def load_model_metadata(model_path: str) -> dict:
    """Load persisted ML model metadata when available."""
    metadata_path = Path(model_path).with_suffix(".metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_export_frame(path: str | None) -> pd.DataFrame:
    """Load an exported CSV preview on demand."""
    if not path:
        return pd.DataFrame()
    export_path = Path(path)
    if not export_path.exists():
        return pd.DataFrame()
    return pd.read_csv(export_path)


def main() -> None:
    """Render the Streamlit UI."""
    st.set_page_config(page_title="Twitter/X Sentiment Tracker", layout="wide")
    _apply_dashboard_theme()
    config, pipeline = load_dashboard_pipeline(str(DEFAULT_CONFIG_PATH))

    st.title(config.dashboard.title)
    st.caption(
        "Run sentiment tracking across X and fallback sources, inspect run provenance, "
        "and review exports and model quality from one control surface."
    )

    if flash_message := st.session_state.pop("flash_success", None):
        st.success(flash_message)
    if flash_warning := st.session_state.pop("flash_warning", None):
        st.warning(flash_warning)

    recent_runs = pipeline.repository.fetch_recent_runs(limit=25)
    run_history_frame = build_run_history_frame(recent_runs)
    run_ids = [int(row["id"]) for row in recent_runs]
    if run_ids and "run_id" not in st.session_state:
        st.session_state["run_id"] = run_ids[0]

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
        backend = st.selectbox(
            "Ingestion Backend",
            options=BACKEND_OPTIONS,
            index=BACKEND_OPTIONS.index(config.ingestion.backend),
            help="Choose the preferred source. Auto can still fall back if the primary path fails.",
        )
        granularity = st.selectbox("Trend Granularity", options=["hourly", "daily"], index=1)
        distribution_chart = st.selectbox("Distribution Chart", options=["pie", "bar"], index=0)
        run_button = st.button("Run Tracking Pipeline", use_container_width=True)

        st.divider()
        st.subheader("Run History")
        if run_ids:
            default_run_id = st.session_state.get("run_id", run_ids[0])
            selected_run_id = st.selectbox(
                "Inspect Stored Run",
                options=run_ids,
                index=run_ids.index(default_run_id) if default_run_id in run_ids else 0,
                format_func=lambda value: _format_run_option(value, recent_runs),
            )
            st.session_state["run_id"] = int(selected_run_id)
        else:
            st.caption("No stored runs yet.")

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
            result = pipeline.run(
                search_parameters=search_parameters,
                model_name=model_name,
                backend_override=backend,
            )
            st.session_state["run_id"] = result.run_id
            st.session_state["flash_success"] = (
                f"Run {result.run_id} completed using {result.served_backend} "
                f"(requested {result.requested_backend}, resolved {result.resolved_backend})."
            )
            if result.fallback_used:
                st.session_state["flash_warning"] = (
                    f"Run {result.run_id} fell back to {result.served_backend} after "
                    f"{result.resolved_backend} did not return usable data."
                )
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Run the pipeline to populate charts and metrics.")
        return

    run_details = pipeline.repository.fetch_run_details(int(run_id))
    if not run_details:
        st.warning("The selected run could not be found in the database.")
        return

    summary = build_summary_stats(
        pipeline.repository.fetch_summary_stats(int(run_id)),
        build_distribution_frame(pipeline.repository.fetch_sentiment_distribution(int(run_id))),
    )
    time_series = add_rolling_average(
        build_timeseries_frame(
            pipeline.repository.fetch_time_series(granularity, int(run_id)),
            granularity,
        )
    )
    distribution = build_distribution_frame(
        pipeline.repository.fetch_sentiment_distribution(int(run_id))
    )
    source_breakdown = build_source_breakdown_frame(
        pipeline.repository.fetch_source_breakdown(int(run_id), limit=20)
    )
    recent_tweets = pd.DataFrame(pipeline.repository.fetch_recent_tweets(int(run_id), limit=100))
    processed_export_preview = load_export_frame(run_details.get("processed_export_path"))
    model_metadata = load_model_metadata(str(config.model.ml_model_path))

    _render_run_header(run_details)

    if run_details["status"] == "failed" and run_details.get("error_message"):
        st.error(run_details["error_message"])
    elif summary["total_tweets"] == 0:
        st.warning(
            "This run completed without stored items. Review the run provenance below to see "
            "which backend was attempted and whether a fallback was used."
        )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Items", int(summary["total_tweets"]))
    metric_columns[1].metric("Average Sentiment", f'{summary["average_sentiment"]:.3f}')
    metric_columns[2].metric("Average Confidence", f'{summary["average_confidence"]:.2%}')
    metric_columns[3].metric("Dominant Sentiment", str(summary["dominant_sentiment"]).title())

    overview_tab, sources_tab, data_tab, model_tab = st.tabs(
        ["Overview", "Sources", "Data", "Model"]
    )

    with overview_tab:
        chart_columns = st.columns((2, 1))
        chart_columns[0].plotly_chart(
            build_sentiment_trend_chart(time_series),
            use_container_width=True,
        )
        chart_columns[1].plotly_chart(
            build_sentiment_distribution_chart(distribution, chart_type=distribution_chart),
            use_container_width=True,
        )
        st.subheader("Run Metadata")
        st.json(
            {
                "run_id": run_details["id"],
                "status": run_details["status"],
                "source_query": run_details["source_query"],
                "model_name": run_details["model_name"],
                "requested_backend": run_details["requested_backend"],
                "resolved_backend": run_details["resolved_backend"],
                "served_backend": run_details["served_backend"],
                "fallback_used": bool(run_details["fallback_used"]),
                "total_collected": run_details["total_collected"],
                "total_stored": run_details["total_stored"],
                "started_at": run_details["started_at"],
                "completed_at": run_details["completed_at"],
            }
        )

    with sources_tab:
        source_chart_col, history_col = st.columns((3, 2))
        source_chart_col.plotly_chart(
            build_source_breakdown_chart(source_breakdown),
            use_container_width=True,
        )
        history_col.subheader("Recent Runs")
        history_col.dataframe(
            run_history_frame[
                [
                    "id",
                    "status",
                    "model_name",
                    "served_backend",
                    "fallback_used",
                    "total_stored",
                    "started_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Source Breakdown")
        st.dataframe(source_breakdown, use_container_width=True, hide_index=True)

    with data_tab:
        download_columns = st.columns(2)
        _render_export_download(
            download_columns[0],
            "Download Raw Export",
            run_details.get("raw_export_path"),
        )
        _render_export_download(
            download_columns[1],
            "Download Processed Export",
            run_details.get("processed_export_path"),
        )

        st.subheader("Recent Items")
        st.dataframe(recent_tweets, use_container_width=True, hide_index=True)

        st.subheader("Processed Export Preview")
        if processed_export_preview.empty:
            st.info("No processed export preview is available for this run.")
        else:
            st.dataframe(processed_export_preview.head(100), use_container_width=True, hide_index=True)

    with model_tab:
        st.subheader("Active Model Configuration")
        st.write(
            {
                "selected_model": run_details["model_name"],
                "configured_default": config.model.active_model,
                "artifact_path": str(config.model.ml_model_path),
                "auto_train_if_missing": config.model.auto_train_if_missing,
            }
        )
        if not model_metadata:
            st.info("No ML model metadata file is available yet.")
        else:
            candidate_scores = pd.DataFrame(model_metadata.get("candidate_scores", []))
            metadata_columns = st.columns(3)
            metadata_columns[0].metric(
                "Selected ML Candidate",
                str(model_metadata.get("selected_model", "unknown")),
            )
            metadata_columns[1].metric(
                "Training Rows",
                int(model_metadata.get("training_rows", 0)),
            )
            metadata_columns[2].metric(
                "Feature Pipeline Count",
                len(model_metadata.get("candidate_scores", [])),
            )
            st.json(model_metadata)
            if not candidate_scores.empty:
                st.subheader("Candidate Validation Scores")
                st.dataframe(candidate_scores, use_container_width=True, hide_index=True)


def _apply_dashboard_theme() -> None:
    """Apply a compact editorial dashboard theme on top of Streamlit defaults."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(19,25,34,0.98), rgba(12,16,23,0.96));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
        }
        div[data-testid="stMetricLabel"] {
            color: #94a3b8;
            letter-spacing: 0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_run_header(run_details: dict) -> None:
    """Render a high-signal run summary banner."""
    status = str(run_details.get("status", "unknown")).title()
    served_backend = str(run_details.get("served_backend", "unknown"))
    requested_backend = str(run_details.get("requested_backend", "unknown"))
    resolved_backend = str(run_details.get("resolved_backend", "unknown"))
    fallback_text = "Yes" if bool(run_details.get("fallback_used")) else "No"
    st.markdown(
        f"""
        <div style="
            margin: 0.5rem 0 1.25rem 0;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.94));
            border: 1px solid rgba(148,163,184,0.18);
        ">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;">
                Run {run_details.get("id")}
            </div>
            <div style="font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem;">
                {status} via {served_backend}
            </div>
            <div style="margin-top: 0.35rem; color: #cbd5e1;">
                Requested <strong>{requested_backend}</strong>, resolved <strong>{resolved_backend}</strong>,
                fallback used <strong>{fallback_text}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_export_download(container, label: str, export_path: str | None) -> None:
    """Show a CSV download button when an export exists."""
    with container:
        if not export_path:
            st.info(f"{label}: not available for this run.")
            return
        path = Path(export_path)
        if not path.exists():
            st.warning(f"{label}: file is missing on disk.")
            return
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            use_container_width=True,
        )


def _format_run_option(run_id: int, recent_runs: list[dict]) -> str:
    """Build a compact label for the run-history selector."""
    row = next((item for item in recent_runs if int(item["id"]) == int(run_id)), None)
    if row is None:
        return f"Run {run_id}"
    served_backend = row.get("served_backend") or "unknown"
    model_name = row.get("model_name") or "unknown"
    status = row.get("status") or "unknown"
    stored = row.get("total_stored") or 0
    return f"Run {row['id']} • {status} • {served_backend} • {model_name} • {stored} stored"


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
