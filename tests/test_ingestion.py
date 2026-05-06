"""Tests for the ingestion layer."""

from __future__ import annotations

import importlib.machinery
import io
import logging
import ssl
from datetime import date, datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from urllib import error as urllib_error

import pytest

from app.schemas import RawTweet, SearchParameters
from ingestion.retry import execute_with_retry
from ingestion.scraper import (
    NonRetryableIngestionError,
    TwitterScraper,
    _extract_twscrape_operation_id,
    _extract_twscrape_script_urls,
    _ensure_snscrape_import_compatibility,
    _load_project_dotenv_values,
    _load_twscrape_credentials_from_env,
    _load_x_bearer_token,
    _load_snscrape_twitter_module,
)


async def _async_return(value):
    return value


def test_build_query_includes_keyword_hashtags_and_date_filters() -> None:
    query = TwitterScraper.build_query(
        SearchParameters(
            keyword="openai",
            hashtags=["AI", "llm"],
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 6),
            max_results=25,
        )
    )

    assert query == "openai #AI #llm since:2026-05-01 until:2026-05-07"


def test_build_x_api_query_excludes_legacy_date_operators() -> None:
    query = TwitterScraper.build_x_api_query(
        SearchParameters(
            keyword="openai",
            hashtags=["AI", "llm"],
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 6),
            max_results=25,
        )
    )

    assert query == "openai #AI #llm"


def test_build_news_queries_fans_out_broader_topic_variants() -> None:
    queries = TwitterScraper.build_news_queries(
        SearchParameters(
            keyword="openai",
            hashtags=["AI", "llm"],
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 6),
            max_results=25,
        )
    )

    assert queries == [
        "openai AI llm",
        "openai AI",
        "openai llm",
        "AI llm",
        "openai OR AI OR llm",
        "openai",
        "AI",
        "llm",
    ]


def test_snscrape_twitter_module_imports_with_compatibility_shim() -> None:
    added_find_module = not hasattr(importlib.machinery.FileFinder, "find_module")

    try:
        _ensure_snscrape_import_compatibility()
        assert hasattr(importlib.machinery.FileFinder, "find_module")

        module = _load_snscrape_twitter_module()

        assert hasattr(module, "TwitterSearchScraper")
    finally:
        _load_snscrape_twitter_module.cache_clear()
        if added_find_module:
            delattr(importlib.machinery.FileFinder, "find_module")


def test_auto_backend_prefers_twscrape_when_cookies_env_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "ingestion.scraper._project_dotenv_path",
        lambda: tmp_path / ".env.missing",
    )
    _load_project_dotenv_values.cache_clear()
    monkeypatch.delenv("Bearer_Token", raising=False)
    monkeypatch.setenv("TWITTER_COOKIES", "auth_token=abc; ct0=xyz")

    scraper = TwitterScraper(
        backend="auto",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "twscrape_accounts.db",
        logger=logging.getLogger("test.ingestion"),
    )

    try:
        assert scraper._resolve_backend() == "twscrape"
    finally:
        _load_project_dotenv_values.cache_clear()


def test_auto_backend_falls_back_to_snscrape_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "ingestion.scraper._project_dotenv_path",
        lambda: tmp_path / ".env.missing",
    )
    _load_project_dotenv_values.cache_clear()
    for key in [
        "TWITTER_COOKIES",
        "TWITTER_USERNAME",
        "TWITTER_PASSWORD",
        "TWITTER_EMAIL",
        "TWITTER_EMAIL_PASSWORD",
        "Bearer_Token",
    ]:
        monkeypatch.delenv(key, raising=False)

    scraper = TwitterScraper(
        backend="auto",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "missing_accounts.db",
        logger=logging.getLogger("test.ingestion"),
    )

    try:
        assert scraper._resolve_backend() == "snscrape"
    finally:
        _load_project_dotenv_values.cache_clear()


def test_auto_backend_prefers_x_api_when_bearer_token_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "ingestion.scraper._project_dotenv_path",
        lambda: tmp_path / ".env.missing",
    )
    _load_project_dotenv_values.cache_clear()
    monkeypatch.setenv("Bearer_Token", "test-token")

    scraper = TwitterScraper(
        backend="auto",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "missing_accounts.db",
        logger=logging.getLogger("test.ingestion"),
    )

    try:
        assert scraper._resolve_backend() == "x_api"
    finally:
        _load_project_dotenv_values.cache_clear()


def test_bearer_token_loads_from_project_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("Bearer_Token=test-token-from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("Bearer_Token", raising=False)
    monkeypatch.setattr("ingestion.scraper._project_dotenv_path", lambda: env_file)
    _load_project_dotenv_values.cache_clear()

    try:
        assert _load_x_bearer_token() == "test-token-from-dotenv"
    finally:
        _load_project_dotenv_values.cache_clear()


def test_twscrape_cookie_parts_load_from_project_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "auth_token=test-auth-token\n"
        "cto=test-csrf-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("ingestion.scraper._project_dotenv_path", lambda: env_file)
    for key in [
        "TWITTER_COOKIES",
        "TWITTER_AUTH_TOKEN",
        "TWITTER_CT0",
        "TWITTER_CTO",
        "auth_token",
        "ct0",
        "cto",
    ]:
        monkeypatch.delenv(key, raising=False)
    _load_project_dotenv_values.cache_clear()

    try:
        credentials = _load_twscrape_credentials_from_env()
        assert credentials is not None
        assert credentials.cookies == "auth_token=test-auth-token; ct0=test-csrf-token"
    finally:
        _load_project_dotenv_values.cache_clear()


def test_extract_twscrape_script_urls_from_current_x_html_shape() -> None:
    html = """
    <link rel="preload" href="https://abs.twimg.com/responsive-web/client-web/vendor.abc123.js" />
    <link rel="preload" href="https://abs.twimg.com/responsive-web/client-web/main.def456.js" />
    <link rel="preload" href="https://abs.twimg.com/responsive-web/client-web/main.def456.js" />
    """

    assert _extract_twscrape_script_urls(html) == [
        "https://abs.twimg.com/responsive-web/client-web/vendor.abc123.js",
        "https://abs.twimg.com/responsive-web/client-web/main.def456.js",
    ]


def test_extract_twscrape_operation_id_from_bundle() -> None:
    bundle = 'e.exports={queryId:"xrS3h-srT2mQT-g3lKsUjA",operationName:"SearchTimeline",operationType:"query"}'
    assert _extract_twscrape_operation_id(bundle, "SearchTimeline") == "xrS3h-srT2mQT-g3lKsUjA"


def test_x_api_request_uses_ssl_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scraper = TwitterScraper(
        backend="x_api",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )
    sentinel_context = object()
    captured: dict[str, object] = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"data": [], "meta": {}}'

    def fake_urlopen(request, timeout, context):
        captured["request"] = request
        captured["timeout"] = timeout
        captured["context"] = context
        return DummyResponse()

    monkeypatch.setattr("ingestion.scraper._build_ssl_context", lambda: sentinel_context)
    monkeypatch.setattr("ingestion.scraper.urllib_request.urlopen", fake_urlopen)

    payload = scraper._perform_x_api_request({"query": "openai"}, "token")

    assert payload == {"data": [], "meta": {}}
    assert captured["timeout"] == 20
    assert captured["context"] is sentinel_context


def test_x_api_ssl_verification_failure_is_not_retryable(tmp_path: Path) -> None:
    scraper = TwitterScraper(
        backend="x_api",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )

    def fake_urlopen(*args, **kwargs):
        raise urllib_error.URLError(
            ssl.SSLCertVerificationError("certificate verify failed")
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("ingestion.scraper.urllib_request.urlopen", fake_urlopen)
        with pytest.raises(NonRetryableIngestionError, match="TLS verification failed"):
            scraper._perform_x_api_request({"query": "openai"}, "token")


def test_x_api_402_error_is_not_retryable(tmp_path: Path) -> None:
    scraper = TwitterScraper(
        backend="x_api",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )

    def fake_urlopen(*args, **kwargs):
        raise urllib_error.HTTPError(
            url="https://api.x.com/2/tweets/search/recent",
            code=402,
            msg="Payment Required",
            hdrs=None,
            fp=io.BytesIO(
                b'{"title":"CreditsDepleted","detail":"Account has no credits left."}'
            ),
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("ingestion.scraper.urllib_request.urlopen", fake_urlopen)
        with pytest.raises(NonRetryableIngestionError, match="CreditsDepleted"):
            scraper._perform_x_api_request({"query": "openai"}, "token")


def test_fetch_tweets_falls_back_to_news_rss_when_primary_returns_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scraper = TwitterScraper(
        backend="twscrape",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )
    calls: list[str] = []
    fallback_tweet = RawTweet(
        tweet_id="news-1",
        created_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
        username="axios",
        display_name="Axios",
        content="OpenAI launches self-serve ad platform",
        url="https://example.com/openai-ad-platform",
        raw_payload={"source_type": "google_news_rss"},
    )

    monkeypatch.setattr(scraper, "_resolve_backend", lambda: "twscrape")
    monkeypatch.setattr(scraper, "_log_backend_query", lambda backend, params: None)

    def fake_execute_backend(backend: str, params: SearchParameters) -> list[RawTweet]:
        calls.append(backend)
        return [] if backend == "twscrape" else [fallback_tweet]

    monkeypatch.setattr(scraper, "_execute_backend_fetch", fake_execute_backend)

    tweets = scraper.fetch_tweets(
        SearchParameters(
            keyword="openai",
            hashtags=["AI"],
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 6),
            max_results=25,
        )
    )

    assert calls == ["twscrape", "news_rss"]
    assert tweets == [fallback_tweet]


def test_fetch_batch_records_requested_resolved_and_served_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scraper = TwitterScraper(
        backend="auto",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )
    fallback_tweet = RawTweet(
        tweet_id="news-1",
        created_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
        username="axios",
        display_name="Axios",
        content="OpenAI launches self-serve ad platform",
        url="https://example.com/openai-ad-platform",
        raw_payload={"source_type": "google_news_rss"},
    )

    monkeypatch.setattr(scraper, "_log_backend_query", lambda backend, params: None)
    monkeypatch.setattr(
        scraper,
        "resolve_backend_request",
        lambda backend_override=None: ("auto", "twscrape"),
    )
    monkeypatch.setattr(
        scraper,
        "_execute_backend_fetch",
        lambda backend, params: [] if backend == "twscrape" else [fallback_tweet],
    )

    batch = scraper.fetch_batch(
        SearchParameters(
            keyword="openai",
            hashtags=["AI"],
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 6),
            max_results=25,
        )
    )

    assert batch.requested_backend == "auto"
    assert batch.resolved_backend == "twscrape"
    assert batch.served_backend == "news_rss"
    assert batch.fallback_used is True
    assert batch.tweets == [fallback_tweet]


def test_fetch_with_news_rss_parses_and_filters_items(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scraper = TwitterScraper(
        backend="news_rss",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )
    captured_urls: list[str] = []
    xml_payload = b"""
    <rss>
      <channel>
        <item>
          <title>OpenAI launches self-serve ad platform - Axios</title>
          <link>https://example.com/openai-ad-platform</link>
          <guid>guid-1</guid>
          <pubDate>Wed, 06 May 2026 03:49:16 GMT</pubDate>
          <source url="https://www.axios.com">Axios</source>
          <description><![CDATA[<a href="https://example.com/openai-ad-platform">OpenAI launches self-serve ad platform - Axios</a>]]></description>
        </item>
        <item>
          <title>Too old to include - Example</title>
          <link>https://example.com/old</link>
          <guid>guid-2</guid>
          <pubDate>Tue, 07 Apr 2026 07:00:00 GMT</pubDate>
          <source url="https://example.com">Example</source>
          <description>Too old</description>
        </item>
      </channel>
    </rss>
    """

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return xml_payload

    def fake_urlopen(request, timeout, context):
        captured_urls.append(request.full_url)
        return DummyResponse()

    monkeypatch.setattr("ingestion.scraper.urllib_request.urlopen", fake_urlopen)

    tweets = scraper._fetch_with_news_rss(
        SearchParameters(
            keyword="openai",
            hashtags=["AI"],
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 6),
            max_results=10,
        )
    )

    assert len(tweets) == 1
    assert tweets[0].display_name == "Axios"
    assert tweets[0].username == "axios"
    assert tweets[0].url == "https://example.com/openai-ad-platform"
    assert tweets[0].raw_payload["source_type"] == "google_news_rss"
    assert captured_urls


def test_twscrape_cloudflare_login_block_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scraper = TwitterScraper(
        backend="twscrape",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )
    account = SimpleNamespace(active=False, error_msg=None)
    saved: dict[str, object] = {}

    class DummyResponse:
        status_code = 403
        text = (
            "<html><title>Attention Required! | Cloudflare</title>"
            "<h1>Sorry, you have been blocked</h1>"
            "<div>Please enable cookies.</div></html>"
        )

    class DummyExc(Exception):
        def __init__(self):
            self.response = DummyResponse()

    async def fake_login(account, cfg):
        raise DummyExc()

    async def fake_save(account):
        saved["active"] = account.active
        saved["error_msg"] = account.error_msg

    fake_api = SimpleNamespace(
        pool=SimpleNamespace(
            _login_config=object(),
            save=fake_save,
        )
    )

    monkeypatch.setattr("ingestion.scraper._load_twscrape_login_function", lambda: fake_login)

    with pytest.raises(NonRetryableIngestionError, match="Cloudflare"):
        import asyncio

        asyncio.run(scraper._login_twscrape_account(fake_api, account))

    assert saved["active"] is False
    assert saved["error_msg"] == "Cloudflare blocked login flow"


def test_twscrape_no_account_error_maps_cookie_auth_failure(tmp_path: Path) -> None:
    scraper = TwitterScraper(
        backend="twscrape",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )

    class DummyExc(Exception):
        pass

    fake_api = SimpleNamespace(
        pool=SimpleNamespace(
            get_all=lambda: _async_return(
                [
                    SimpleNamespace(username="twitter_session", active=False, error_msg="(32) Could not authenticate you")
                ]
            )
        )
    )

    import asyncio

    with pytest.raises(NonRetryableIngestionError, match="Refresh auth_token and ct0"):
        asyncio.run(scraper._raise_twscrape_pool_error(fake_api, DummyExc()))


def test_blocked_snscrape_error_is_not_retried(tmp_path: Path) -> None:
    scraper = TwitterScraper(
        backend="snscrape",
        max_retries=3,
        backoff_factor=1.5,
        twscrape_accounts_db=tmp_path / "unused.db",
        logger=logging.getLogger("test.ingestion"),
    )
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        raise RuntimeError(
            "4 requests to https://twitter.com/i/api/graphql/.../SearchTimeline "
            "failed, giving up."
        )

    with pytest.raises(RuntimeError, match="failed, giving up"):
        execute_with_retry(
            operation=operation,
            max_retries=3,
            backoff_factor=1.5,
            logger=logging.getLogger("test.retry"),
            operation_name="snscrape_fetch",
            retry_if=scraper._should_retry_fetch_error,
        )

    assert attempts["count"] == 1
