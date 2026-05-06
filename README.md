# Twitter/X Sentiment Tracking System

A production-oriented Python pipeline for collecting tweets with the official X API or fallback scraping backends, plus a Google News RSS safety net when X is unavailable, cleaning and tokenizing text, scoring sentiment with interchangeable models, storing results in SQLite, and exploring trends in a Streamlit dashboard.

## Architecture

The system is organized into modular layers:

1. `ingestion/`: Collects tweets from Twitter/X using the official X API or fallback scraping backends with retry logic and validated search parameters.
2. `processing/`: Cleans text and tokenizes it with spaCy.
3. `sentiment/`: Provides a rule-based VADER model and a trainable scikit-learn model behind a config-driven selector.
4. `storage/`: Persists tweets, run metadata, and run-to-tweet mappings in SQLite through a repository layer.
5. `analytics/`: Aggregates hourly and daily sentiment metrics and computes rolling averages and distributions.
6. `visualization/`: Builds Plotly figures for trends and sentiment breakdowns.
7. `dashboard/`: Exposes the pipeline through a Streamlit interface.

## Repository Layout

```text
project_root/
├── analytics/
├── app/
├── config/
├── dashboard/
├── data/
├── ingestion/
├── models/
├── processing/
├── sentiment/
├── storage/
├── tests/
├── visualization/
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Optional: install a richer spaCy model. The system defaults to `spacy.blank("en")`, so this is not required:

```bash
python -m spacy download en_core_web_sm
```

3. Review or update the example configuration at `config/config.yaml`.

## Configuration

The pipeline is driven by YAML configuration. The provided example includes:

- ingestion backend selection
- default search parameters
- retry settings
- model selection
- database path
- export directories
- logging configuration

### X API Authentication

The example configuration in `config/config.yaml` is currently set to `ingestion.backend: "twscrape"` for authenticated scraping. If you want to use the official X API recent-search endpoint instead, change it to `x_api` and provide an app-only bearer token.

Put your token in `.env` at the repository root:

```dotenv
Bearer_Token=your_x_bearer_token
```

You can also provide it through environment variables:

```bash
export Bearer_Token='your_x_bearer_token'
```

Alternative names also work:

- `X_BEARER_TOKEN`
- `BEARER_TOKEN`

If you want backend auto-selection, set `ingestion.backend: "auto"` explicitly. The backend chooser prefers `x_api`, then `twscrape`, then `snscrape`; any of those can still fall back to `news_rss` if X blocks the request or returns no items.

### Scraping Fallbacks

If you do not want to use the official API, the project still supports authenticated `twscrape`:

```bash
export TWITTER_COOKIES='auth_token=...; ct0=...'
export TWITTER_USERNAME='your_x_username'
```

or

```bash
export TWITTER_USERNAME='your_x_username'
export TWITTER_PASSWORD='your_x_password'
export TWITTER_EMAIL='you@example.com'
export TWITTER_EMAIL_PASSWORD='your_email_password'
```

Optional variables:

- `TWITTER_PROXY`
- `TWITTER_MFA_CODE`

If X keeps failing or you want a fully free path, you can also set:

```yaml
ingestion:
  backend: "news_rss"
```

That backend searches Google News RSS across multiple topic combinations derived from the configured keyword and hashtags and returns those headlines through the same `RawTweet` schema.

## Run Ingestion

Run the full ingestion and sentiment pipeline using the defaults from `config/config.yaml`:

```bash
python -m app.main run
```

If you want to override the backend explicitly:

```bash
python -m app.main run --backend news_rss
```

Override the search parameters from the CLI:

```bash
python -m app.main run \
  --query "openai" \
  --hashtags "AI,LLM" \
  --start-date 2026-05-01 \
  --end-date 2026-05-06 \
  --max-results 250 \
  --backend twscrape \
  --model vader
```

Outputs are written to:

- `data/raw/`: CSV snapshots of scraped tweets
- `data/processed/`: CSV snapshots of analyzed tweets
- `data/twitter_sentiment.db`: SQLite database with tweet and run data

## Train the ML Model

Train the scikit-learn sentiment model from the bundled sample dataset or your own labeled CSV with `text` and `label` columns:

```bash
python -m app.main train-ml
```

When no `--training-data` path is provided, the command now builds a combined corpus from all trainable CSVs in `data/processed/` plus the configured seed dataset, and only falls back to the seed dataset alone if no processed export is suitable.

The trainer now also:

- merges all trainable processed exports plus the seed labeled dataset
- deduplicates repeated texts across runs
- downweights weaker pseudo-labels from processed exports using sentiment confidence/score
- evaluates multiple candidate pipelines and selects the best one by validation macro F1
- writes model metadata to `models/sentiment_model.metadata.json`

Custom dataset example:

```bash
python -m app.main train-ml --training-data /absolute/path/to/training_data.csv
```

The trainer accepts either:

- `text` and `label`
- `cleaned_text` and `sentiment_label`
- `content` and `sentiment_label`

The trained model is saved to `models/sentiment_model.joblib` by default.

## Run the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard supports:

- keyword and hashtag inputs
- date range filtering
- backend selection per run
- dynamic model selection
- hourly or daily trend views
- interactive Plotly charts
- run provenance showing requested/resolved/served backend and fallback usage
- run history inspection for the most recent stored runs
- CSV export downloads and processed-export previews
- source breakdown analysis and recent analyzed tweets
- ML model metadata and validation-score inspection

## Testing

Run the unit tests with:

```bash
pytest
```

## Notes

- As of May 6, 2026, X frequently rejects unauthenticated `snscrape` search requests with `blocked (404)` responses.
- `x_api` uses the official recent-search endpoint and requires a bearer token. Recent search only covers roughly the last 7 days.
- `twscrape` requires an authenticated X account or fresh cookies to access search reliably.
- `news_rss` is a non-X fallback. It returns topical headlines, not actual tweets, but it keeps the downstream sentiment pipeline usable when X blocks search.
- The rule-based model uses the standalone `vaderSentiment` package, so no separate lexicon download is required.
- The sample training dataset in `data/processed/training_sample.csv` is intended for bootstrapping and testing.
