# Project Graph

This graph summarizes the current Python package structure and the main runtime flow in this repository.

## Package Dependency Graph

```mermaid
flowchart LR
    dashboard[dashboard] --> analytics[analytics]
    dashboard --> app[app]
    dashboard --> visualization[visualization]

    app --> ingestion[ingestion]
    app --> processing[processing]
    app --> sentiment[sentiment]
    app --> storage[storage]

    ingestion --> app
    sentiment --> app
    storage --> app

    tests[tests] --> processing
    tests --> sentiment
```

## Runtime Flow

```mermaid
flowchart TD
    cli[app.main] --> pipeline[app.pipeline]
    dashboard_ui[dashboard.app] --> pipeline

    pipeline --> scraper[ingestion.scraper]
    pipeline --> cleaner[processing.cleaner]
    pipeline --> tokenizer[processing.tokenizer]
    pipeline --> selector[sentiment.model_selector]
    selector --> vader[sentiment.vader_model]
    selector --> ml[sentiment.ml_model]
    pipeline --> repository[storage.repository]
    repository --> database[storage.database]

    repository --> analytics[analytics.aggregator / analytics.metrics]
    analytics --> charts[visualization.plots]
    charts --> dashboard_ui
```

## Layer Summary

- `app/` wires configuration, schemas, logging, CLI entrypoints, and pipeline orchestration.
- `ingestion/` pulls tweets with retry handling.
- `processing/` cleans and tokenizes tweet text.
- `sentiment/` selects and runs either the VADER or scikit-learn model.
- `storage/` persists runs and tweets in SQLite and serves query results back to the app.
- `analytics/` converts stored results into time series, distributions, and summary metrics.
- `visualization/` builds Plotly charts for the dashboard.
- `dashboard/` provides the Streamlit UI on top of the same pipeline.
