"""Twitter/X ingestion logic backed by selectable API and scraping backends."""

from __future__ import annotations

import asyncio
import hashlib
import html
import importlib
import importlib.machinery
import json
import logging
import os
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import xml.etree.ElementTree as ET

from app.schemas import RawTweet, SearchParameters
from ingestion.retry import execute_with_retry


class IngestionError(RuntimeError):
    """Base error for ingestion backend failures."""


class NonRetryableIngestionError(IngestionError):
    """Raised when retrying the same request is known to be pointless."""


@dataclass(frozen=True)
class TwscrapeCredentials:
    """Credential material used to bootstrap a twscrape account pool."""

    username: str
    password: str
    email: str
    email_password: str
    cookies: str | None = None
    proxy: str | None = None
    mfa_code: str | None = None


def _project_dotenv_path() -> Path:
    """Return the project-level dotenv path."""
    return Path(__file__).resolve().parents[1] / ".env"


@lru_cache(maxsize=1)
def _load_project_dotenv_values() -> dict[str, str]:
    """Load simple KEY=VALUE pairs from the project's .env file."""
    env_file = _project_dotenv_path()
    if not env_file.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _ensure_snscrape_import_compatibility() -> None:
    """Restore the legacy finder API expected by snscrape on Python 3.12+."""
    if hasattr(importlib.machinery.FileFinder, "find_module"):
        return

    def find_module(self, fullname: str, path=None):
        spec = self.find_spec(fullname)
        return None if spec is None else spec.loader

    importlib.machinery.FileFinder.find_module = find_module  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _load_snscrape_twitter_module() -> ModuleType:
    """Import the snscrape Twitter module only when the pipeline needs it."""
    _ensure_snscrape_import_compatibility()
    return importlib.import_module("snscrape.modules.twitter")


@lru_cache(maxsize=1)
def _load_twscrape_api_class():
    """Import twscrape only when an authenticated backend is needed."""
    _patch_twscrape_runtime()
    return importlib.import_module("twscrape").API


@lru_cache(maxsize=1)
def _load_twscrape_login_function():
    """Import twscrape's login coroutine only when account bootstrap is needed."""
    return importlib.import_module("twscrape.login").login


def _extract_twscrape_script_urls(text: str) -> list[str]:
    """Extract current responsive-web script URLs from an X HTML document."""
    pattern = r"https://abs\.twimg\.com/responsive-web/client-web/[^\"']+\.js"
    seen: set[str] = set()
    urls: list[str] = []
    for match in re.findall(pattern, text):
        if match not in seen:
            seen.add(match)
            urls.append(match)
    return urls


def _extract_twscrape_operation_id(bundle_text: str, operation_name: str) -> str | None:
    """Extract a GraphQL queryId from the current X frontend bundle."""
    pattern = rf'queryId:"([^"]+)",operationName:"{re.escape(operation_name)}"'
    match = re.search(pattern, bundle_text)
    return match.group(1) if match else None


@lru_cache(maxsize=1)
def _load_current_twscrape_search_operation_id() -> str | None:
    """Fetch the current SearchTimeline queryId from X's main frontend bundle."""
    headers = {"User-Agent": "Mozilla/5.0"}
    ssl_context = _build_ssl_context()

    try:
        html_request = urllib_request.Request("https://x.com/tesla", headers=headers)
        with urllib_request.urlopen(html_request, timeout=20, context=ssl_context) as response:
            html = response.read().decode("utf-8", errors="replace")

        main_bundle_url = next(
            (url for url in _extract_twscrape_script_urls(html) if "/main." in url),
            None,
        )
        if main_bundle_url is None:
            return None

        bundle_request = urllib_request.Request(main_bundle_url, headers=headers)
        with urllib_request.urlopen(bundle_request, timeout=20, context=ssl_context) as response:
            bundle_text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    return _extract_twscrape_operation_id(bundle_text, "SearchTimeline")


def _patch_twscrape_runtime() -> None:
    """Patch twscrape's brittle XClId script discovery for the current X frontend."""
    xclid = importlib.import_module("twscrape.xclid")
    api = importlib.import_module("twscrape.api")
    if getattr(xclid, "_twitter_sentiment_patched", False):
        return

    original_get_scripts_list = xclid.get_scripts_list

    def patched_get_scripts_list(text: str):
        urls = _extract_twscrape_script_urls(text)
        if urls:
            for url in urls:
                yield url
            return

        yield from original_get_scripts_list(text)

    async def patched_parse_anim_idx(text: str) -> list[int]:
        scripts = list(patched_get_scripts_list(text))
        preferred = [x for x in scripts if "/ondemand.s." in x]
        if not preferred:
            preferred = [x for x in scripts if "/main." in x] or scripts

        for script_url in preferred:
            script_text = await xclid.get_tw_page_text(script_url)
            items = [int(match.group(2)) for match in xclid.INDICES_REGEX.finditer(script_text)]
            if items:
                return items

        raise Exception("Couldn't get XClientTxId indices")

    xclid.get_scripts_list = patched_get_scripts_list
    xclid.parse_anim_idx = patched_parse_anim_idx
    if search_query_id := _load_current_twscrape_search_operation_id():
        api.OP_SearchTimeline = f"{search_query_id}/SearchTimeline"
    xclid._twitter_sentiment_patched = True


def _is_snscrape_blocked_error(exc: Exception) -> bool:
    """Return whether snscrape hit X's blocked unauthenticated search path."""
    text = str(exc)
    return "SearchTimeline" in text and (
        "blocked (404)" in text or "failed, giving up" in text
    )


def _is_cloudflare_block_page(body: str) -> bool:
    """Detect Cloudflare block pages returned during twscrape login."""
    normalized = body.lower()
    return (
        "cloudflare" in normalized
        and (
            "attention required!" in normalized
            or "sorry, you have been blocked" in normalized
            or "please enable cookies" in normalized
        )
    )


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        value = _load_project_dotenv_values().get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _load_x_bearer_token() -> str | None:
    """Read the X API bearer token from env or .env."""
    for name in ("Bearer_Token", "X_BEARER_TOKEN", "BEARER_TOKEN"):
        value = _env(name)
        if value:
            return value
    return None


def _load_twitter_cookie_string() -> str | None:
    """Read Twitter/X session cookies from a combined string or separate parts."""
    cookies = _env("TWITTER_COOKIES")
    if cookies:
        return cookies

    auth_token = _env("TWITTER_AUTH_TOKEN") or _env("auth_token")
    ct0 = _env("TWITTER_CT0") or _env("ct0") or _env("TWITTER_CTO") or _env("cto")
    if auth_token and ct0:
        return f"auth_token={auth_token}; ct0={ct0}"

    return None


def _build_ssl_context() -> ssl.SSLContext:
    """Create an HTTPS context with a reliable CA bundle when available."""
    try:
        certifi = importlib.import_module("certifi")
    except ImportError:
        return ssl.create_default_context()

    return ssl.create_default_context(cafile=certifi.where())


def _load_twscrape_credentials_from_env() -> TwscrapeCredentials | None:
    """Read twscrape credentials from environment variables."""
    cookies = _load_twitter_cookie_string()
    username = _env("TWITTER_USERNAME")
    password = _env("TWITTER_PASSWORD")
    email = _env("TWITTER_EMAIL")
    email_password = _env("TWITTER_EMAIL_PASSWORD")
    proxy = _env("TWITTER_PROXY")
    mfa_code = _env("TWITTER_MFA_CODE")

    if cookies:
        return TwscrapeCredentials(
            username=username or "twitter_session",
            password=password or "unused-password",
            email=email or "unused@example.com",
            email_password=email_password or "unused-email-password",
            cookies=cookies,
            proxy=proxy,
            mfa_code=mfa_code,
        )

    if all([username, password, email, email_password]):
        return TwscrapeCredentials(
            username=username,
            password=password,
            email=email,
            email_password=email_password,
            proxy=proxy,
            mfa_code=mfa_code,
        )

    return None


class TwitterScraper:
    """Collect tweets from Twitter/X using the official API or scraping backends."""

    def __init__(
        self,
        backend: str,
        max_retries: int,
        backoff_factor: float,
        twscrape_accounts_db,
        logger: logging.Logger,
    ) -> None:
        self.backend = backend
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.twscrape_accounts_db = twscrape_accounts_db
        self.logger = logger

    def fetch_tweets(self, search_parameters: SearchParameters) -> list[RawTweet]:
        """Fetch tweets for the provided search parameters."""
        search_parameters.validate()
        primary_backend = self._resolve_backend()
        backends = [primary_backend]
        if primary_backend != "news_rss":
            backends.append("news_rss")

        last_error: Exception | None = None

        for index, backend in enumerate(backends):
            self._log_backend_query(backend, search_parameters)
            try:
                tweets = self._execute_backend_fetch(backend, search_parameters)
            except Exception as exc:
                last_error = exc
                if index + 1 < len(backends):
                    self.logger.warning(
                        "%s backend failed for %s. Falling back to %s. Error: %s",
                        backend,
                        search_parameters.source_query,
                        backends[index + 1],
                        exc,
                    )
                    continue
                raise

            if tweets:
                if backend != primary_backend:
                    self.logger.info(
                        "Fallback backend %s returned %s items for %s",
                        backend,
                        len(tweets),
                        search_parameters.source_query,
                    )
                return tweets

            if index + 1 < len(backends):
                self.logger.warning(
                    "%s backend returned 0 items for %s. Falling back to %s.",
                    backend,
                    search_parameters.source_query,
                    backends[index + 1],
                )
                continue

        if last_error is not None:
            raise last_error

        return []

    @staticmethod
    def build_query(search_parameters: SearchParameters) -> str:
        """Build the TwitterSearchScraper query string."""
        search_parameters.validate()
        parts: list[str] = []
        if search_parameters.keyword:
            parts.append(search_parameters.keyword.strip())
        parts.extend(search_parameters.normalized_hashtags())
        if search_parameters.start_date:
            parts.append(f"since:{search_parameters.start_date.isoformat()}")
        if search_parameters.end_date:
            until_date = search_parameters.end_date + timedelta(days=1)
            parts.append(f"until:{until_date.isoformat()}")
        return " ".join(parts)

    @staticmethod
    def build_x_api_query(search_parameters: SearchParameters) -> str:
        """Build an X API v2 recent-search query string."""
        search_parameters.validate()
        parts: list[str] = []
        if search_parameters.keyword:
            parts.append(search_parameters.keyword.strip())
        parts.extend(search_parameters.normalized_hashtags())
        return " ".join(parts)

    @staticmethod
    def build_news_queries(search_parameters: SearchParameters) -> list[str]:
        """Build a fanout of broad topical queries for RSS/news fallback."""
        search_parameters.validate()
        keyword = search_parameters.keyword.strip() if search_parameters.keyword else ""
        hashtag_topics = [
            hashtag.lstrip("#").strip()
            for hashtag in search_parameters.normalized_hashtags()
            if hashtag.lstrip("#").strip()
        ]
        topic_tokens = [token for token in [keyword, *hashtag_topics] if token]

        queries: list[str] = []
        if topic_tokens:
            queries.append(" ".join(topic_tokens))
        if keyword:
            for hashtag_topic in hashtag_topics:
                queries.append(f"{keyword} {hashtag_topic}")
        if hashtag_topics:
            queries.append(" ".join(hashtag_topics))
        if len(topic_tokens) > 1:
            queries.append(" OR ".join(topic_tokens))
        queries.extend(topic_tokens)

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized_query = " ".join(query.split())
            if not normalized_query:
                continue
            key = normalized_query.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized_query)

        return deduped

    def _resolve_backend(self) -> str:
        """Choose the configured backend, or infer one in auto mode."""
        if self.backend != "auto":
            return self.backend

        if _load_x_bearer_token() is not None:
            return "x_api"
        if _load_twscrape_credentials_from_env() is not None or self.twscrape_accounts_db.exists():
            return "twscrape"

        return "snscrape"

    def _log_backend_query(self, backend: str, search_parameters: SearchParameters) -> None:
        """Log the query or query family used by the selected backend."""
        self.logger.info("Using %s ingestion backend", backend)
        if backend == "x_api":
            self.logger.info("Using X API query: %s", self.build_x_api_query(search_parameters))
            return
        if backend == "snscrape":
            self.logger.info("Using snscrape query: %s", self.build_query(search_parameters))
            return
        if backend == "twscrape":
            self.logger.info("Using twscrape query: %s", self.build_query(search_parameters))
            return

        queries = self.build_news_queries(search_parameters)
        preview = ", ".join(queries[:4])
        self.logger.info(
            "Using Google News RSS query fanout (%s queries): %s%s",
            len(queries),
            preview,
            " ..." if len(queries) > 4 else "",
        )

    def _execute_backend_fetch(
        self,
        backend: str,
        search_parameters: SearchParameters,
    ) -> list[RawTweet]:
        """Execute a concrete backend with retry semantics."""
        query = (
            self.build_x_api_query(search_parameters)
            if backend == "x_api"
            else self.build_query(search_parameters)
        )

        def operation() -> list[RawTweet]:
            if backend == "x_api":
                return self._fetch_with_x_api(search_parameters)
            if backend == "twscrape":
                return asyncio.run(self._fetch_with_twscrape(query, search_parameters.max_results))
            if backend == "snscrape":
                return self._fetch_with_snscrape(query, search_parameters.max_results)
            if backend == "news_rss":
                return self._fetch_with_news_rss(search_parameters)
            raise NonRetryableIngestionError(f"Unsupported ingestion backend: {backend}")

        return execute_with_retry(
            operation=operation,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
            logger=self.logger,
            operation_name=f"{backend}_fetch",
            retry_if=self._should_retry_fetch_error,
        )

    def _should_retry_fetch_error(self, exc: Exception) -> bool:
        """Skip outer retries for known upstream rejections and config errors."""
        return not (
            isinstance(exc, NonRetryableIngestionError) or _is_snscrape_blocked_error(exc)
        )

    def _fetch_with_snscrape(self, query: str, max_results: int) -> list[RawTweet]:
        """Fetch tweets using snscrape's unauthenticated search flow."""
        sntwitter = _load_snscrape_twitter_module()
        tweets: list[RawTweet] = []
        scraper = sntwitter.TwitterSearchScraper(query)

        try:
            for index, tweet in enumerate(scraper.get_items()):
                if index >= max_results:
                    break
                tweets.append(self._raw_tweet_from_scraped_tweet(tweet))
        except Exception as exc:
            if _is_snscrape_blocked_error(exc):
                raise NonRetryableIngestionError(
                    "X blocked unauthenticated snscrape search requests. "
                    "Set ingestion.backend to 'twscrape' and provide TWITTER_COOKIES "
                    "or TWITTER_USERNAME/TWITTER_PASSWORD/TWITTER_EMAIL/TWITTER_EMAIL_PASSWORD."
                ) from exc
            raise

        return tweets

    def _fetch_with_x_api(self, search_parameters: SearchParameters) -> list[RawTweet]:
        """Fetch tweets using X API v2 recent search and a bearer token."""
        bearer_token = _load_x_bearer_token()
        if not bearer_token:
            raise NonRetryableIngestionError(
                "No X API bearer token was found. Set Bearer_Token=... in .env or export "
                "Bearer_Token/X_BEARER_TOKEN/BEARER_TOKEN before running the pipeline."
            )

        query = self.build_x_api_query(search_parameters)
        start_time, end_time = self._build_x_api_time_bounds(search_parameters)
        tweets: list[RawTweet] = []
        next_token: str | None = None

        while len(tweets) < search_parameters.max_results:
            remaining = search_parameters.max_results - len(tweets)
            params = {
                "query": query,
                "max_results": str(min(100, max(10, remaining))),
                "tweet.fields": "created_at,lang,public_metrics,author_id",
                "expansions": "author_id",
                "user.fields": "username,name",
            }
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time
            if next_token:
                params["next_token"] = next_token

            response = self._perform_x_api_request(params, bearer_token)
            includes = response.get("includes", {})
            users = {
                str(user.get("id")): user
                for user in includes.get("users", [])
                if user.get("id") is not None
            }

            batch = response.get("data", [])
            for tweet in batch:
                tweets.append(self._raw_tweet_from_x_api_tweet(tweet, users))
                if len(tweets) >= search_parameters.max_results:
                    break

            next_token = response.get("meta", {}).get("next_token")
            if not next_token or not batch:
                break

        return tweets

    def _fetch_with_news_rss(self, search_parameters: SearchParameters) -> list[RawTweet]:
        """Fetch topical text items from Google News RSS as a free fallback source."""
        headers = {"User-Agent": "Mozilla/5.0"}
        ssl_context = _build_ssl_context()
        seen_ids: set[str] = set()
        tweets: list[RawTweet] = []

        for query in self.build_news_queries(search_parameters):
            url = "https://news.google.com/rss/search?" + urllib_parse.urlencode(
                {
                    "q": query,
                    "hl": "en-US",
                    "gl": "US",
                    "ceid": "US:en",
                }
            )
            request = urllib_request.Request(url, headers=headers)
            try:
                with urllib_request.urlopen(request, timeout=20, context=ssl_context) as response:
                    payload = response.read()
            except urllib_error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {400, 401, 403, 404}:
                    raise NonRetryableIngestionError(
                        f"Google News RSS request failed with HTTP {exc.code}: {body}"
                    ) from exc
                raise IngestionError(
                    f"Google News RSS request failed with HTTP {exc.code}: {body}"
                ) from exc
            except urllib_error.URLError as exc:
                if isinstance(exc.reason, ssl.SSLCertVerificationError):
                    raise NonRetryableIngestionError(
                        "Google News RSS TLS verification failed. "
                        "Install/update certifi in this environment or repair the local CA trust store."
                    ) from exc
                raise IngestionError(f"Google News RSS request failed: {exc.reason}") from exc

            try:
                parsed = ET.fromstring(payload)
            except Exception as exc:
                raise NonRetryableIngestionError(
                    "Google News RSS returned malformed XML."
                ) from exc

            for item in parsed.findall("./channel/item"):
                raw_tweet = self._raw_tweet_from_news_item(item, query)
                if raw_tweet is None:
                    continue
                if not self._news_item_in_requested_window(raw_tweet.created_at, search_parameters):
                    continue
                if raw_tweet.tweet_id in seen_ids:
                    continue
                seen_ids.add(raw_tweet.tweet_id)
                tweets.append(raw_tweet)
                if len(tweets) >= search_parameters.max_results:
                    return tweets

        return tweets

    def _perform_x_api_request(self, params: dict[str, str], bearer_token: str) -> dict[str, Any]:
        """Execute a recent-search request against X API v2."""
        url = f"https://api.x.com/2/tweets/search/recent?{urllib_parse.urlencode(params)}"
        request = urllib_request.Request(
            url,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "TwitterSentimentAnalysis/1.0",
            },
        )
        ssl_context = _build_ssl_context()

        try:
            with urllib_request.urlopen(request, timeout=20, context=ssl_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = self._build_x_api_error_message(exc.code, body)
            if exc.code in {400, 401, 402, 403, 404}:
                raise NonRetryableIngestionError(message) from exc
            raise IngestionError(message) from exc
        except urllib_error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise NonRetryableIngestionError(
                    "X API TLS verification failed for api.x.com. "
                    "Install/update certifi in this environment or repair the local CA trust store."
                ) from exc
            raise IngestionError(f"X API request failed: {exc.reason}") from exc

        if "errors" in payload and payload["errors"]:
            detail = "; ".join(
                str(item.get("detail") or item.get("message") or item)
                for item in payload["errors"]
            )
            raise NonRetryableIngestionError(f"X API request failed: {detail}")

        return payload

    def _build_x_api_time_bounds(
        self,
        search_parameters: SearchParameters,
    ) -> tuple[str | None, str | None]:
        """Build X API start/end timestamps from date-only search parameters."""
        now = datetime.now(timezone.utc)
        start_time: datetime | None = None
        end_time: datetime | None = None

        if search_parameters.start_date:
            start_time = datetime.combine(
                search_parameters.start_date,
                time.min,
                tzinfo=timezone.utc,
            )
        if search_parameters.end_date:
            end_time = datetime.combine(
                search_parameters.end_date + timedelta(days=1),
                time.min,
                tzinfo=timezone.utc,
            )
            end_time = min(end_time, now)

        recent_search_floor = now - timedelta(days=7)
        if start_time and start_time < recent_search_floor:
            raise NonRetryableIngestionError(
                "X API recent search only supports the last 7 days. "
                f"start_date={search_parameters.start_date.isoformat()} is too old."
            )
        if start_time and start_time > now:
            raise NonRetryableIngestionError(
                "start_date is in the future for X API recent search."
            )
        if end_time and start_time and start_time >= end_time:
            raise NonRetryableIngestionError(
                "The requested X API time window is empty after applying recent-search limits."
            )

        return (
            start_time.isoformat().replace("+00:00", "Z") if start_time else None,
            end_time.isoformat().replace("+00:00", "Z") if end_time else None,
        )

    @staticmethod
    def _build_x_api_error_message(status_code: int, body: str) -> str:
        """Convert an X API error response into a concise message."""
        detail = body
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if payload.get("errors"):
                detail = "; ".join(
                    str(item.get("detail") or item.get("message") or item)
                    for item in payload["errors"]
                )
            elif payload.get("title") or payload.get("detail"):
                detail = " - ".join(
                    str(part)
                    for part in [payload.get("title"), payload.get("detail")]
                    if part
                )
        return f"X API request failed with HTTP {status_code}: {detail}"

    async def _fetch_with_twscrape(self, query: str, max_results: int) -> list[RawTweet]:
        """Fetch tweets using twscrape's authenticated GraphQL client."""
        api = await self._build_twscrape_api()
        tweets: list[RawTweet] = []

        try:
            async for tweet in api.search(query, limit=max_results):
                tweets.append(self._raw_tweet_from_scraped_tweet(tweet))
                if len(tweets) >= max_results:
                    break
        except Exception as exc:
            if type(exc).__name__ == "NoAccountError":
                await self._raise_twscrape_pool_error(api, exc)
            raise

        return tweets

    async def _build_twscrape_api(self):
        """Construct and bootstrap a twscrape API client."""
        api_class = _load_twscrape_api_class()
        api = api_class(
            pool=str(self.twscrape_accounts_db),
            raise_when_no_account=True,
        )

        credentials = _load_twscrape_credentials_from_env()
        accounts = await api.pool.get_all()
        if credentials is not None and credentials.cookies:
            await self._bootstrap_twscrape_account(api, credentials)
            accounts = await api.pool.get_all()

        if any(account.active for account in accounts):
            return api

        if credentials is None:
            raise NonRetryableIngestionError(
                "No active twscrape account is available. "
                f"Populate {self.twscrape_accounts_db} with an active account or set "
                "TWITTER_COOKIES, or set TWITTER_USERNAME/TWITTER_PASSWORD/"
                "TWITTER_EMAIL/TWITTER_EMAIL_PASSWORD."
            )

        await self._bootstrap_twscrape_account(api, credentials)
        accounts = await api.pool.get_all()
        if not any(account.active for account in accounts):
            raise NonRetryableIngestionError(
                "twscrape account bootstrap did not produce an active account. "
                "Refresh TWITTER_COOKIES or verify the TWITTER_* login credentials."
            )

        return api

    async def _bootstrap_twscrape_account(self, api: Any, credentials: TwscrapeCredentials) -> None:
        """Create or refresh the configured twscrape account."""
        existing_account = await api.pool.get_account(credentials.username)
        if existing_account is not None and (credentials.cookies or not existing_account.active):
            await api.pool.delete_accounts(credentials.username)
            existing_account = None

        if existing_account is None:
            await api.pool.add_account(
                username=credentials.username,
                password=credentials.password,
                email=credentials.email,
                email_password=credentials.email_password,
                proxy=credentials.proxy,
                cookies=credentials.cookies,
                mfa_code=credentials.mfa_code,
            )

        account = await api.pool.get_account(credentials.username)
        if account is None:
            raise NonRetryableIngestionError(
                f"Unable to create twscrape account entry for {credentials.username!r}."
            )

        if account.active:
            return

        if credentials.cookies:
            raise NonRetryableIngestionError(
                "TWITTER_COOKIES was provided, but the session did not become active. "
                "Provide a fresh cookie string that includes ct0 and auth_token."
            )

        await self._login_twscrape_account(api, account)

    async def _login_twscrape_account(self, api: Any, account: Any) -> None:
        """Run twscrape login with actionable error mapping for blocked flows."""
        login_func = _load_twscrape_login_function()

        try:
            await login_func(account, cfg=api.pool._login_config)
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            body = getattr(response, "text", "") if response is not None else ""

            account.active = False

            if body and _is_cloudflare_block_page(body):
                account.error_msg = "Cloudflare blocked login flow"
                await api.pool.save(account)
                raise NonRetryableIngestionError(
                    "X blocked the twscrape login flow with Cloudflare from this IP/session. "
                    "Use fresh TWITTER_COOKIES from a browser session, or try a different "
                    "IP/TWITTER_PROXY. Username/password login will not work until the block is cleared."
                ) from exc

            account.error_msg = (
                f"HTTP {status_code}" if status_code is not None else str(exc)
            )
            await api.pool.save(account)
            raise NonRetryableIngestionError(
                "twscrape login failed before an active session was established. "
                "Verify the TWITTER_* credentials, or prefer TWITTER_COOKIES from a browser session."
            ) from exc

        account.error_msg = None
        await api.pool.save(account)

    async def _raise_twscrape_pool_error(self, api: Any, exc: Exception) -> None:
        """Translate twscrape pool failures into actionable ingestion errors."""
        accounts = await api.pool.get_all()
        error_messages = [str(account.error_msg or "") for account in accounts]

        if any("Could not authenticate you" in message for message in error_messages):
            raise NonRetryableIngestionError(
                "X rejected the provided TWITTER_COOKIES with 401 '(32) Could not authenticate you)'. "
                "Refresh auth_token and ct0 from a logged-in browser session; the current cookies are "
                "expired, mismatched, or not valid for authenticated GraphQL requests."
            ) from exc

        if any("Cloudflare blocked login flow" in message for message in error_messages):
            raise NonRetryableIngestionError(
                "X blocked the twscrape login flow with Cloudflare from this IP/session. "
                "Use fresh TWITTER_COOKIES from a browser session, or try a different "
                "IP/TWITTER_PROXY. Username/password login will not work until the block is cleared."
            ) from exc

        raise NonRetryableIngestionError(
            "No twscrape account is currently available for SearchTimeline. "
            "Reset stale locks or refresh the configured session credentials."
        ) from exc

    @staticmethod
    def _news_item_in_requested_window(
        created_at: datetime,
        search_parameters: SearchParameters,
    ) -> bool:
        """Apply local date filtering to fallback RSS items."""
        item_date = created_at.astimezone(timezone.utc).date()
        if search_parameters.start_date and item_date < search_parameters.start_date:
            return False
        if search_parameters.end_date and item_date > search_parameters.end_date:
            return False
        return True

    @staticmethod
    def _clean_news_text(value: str | None) -> str:
        """Strip simple HTML from Google News RSS text fields."""
        if not value:
            return ""
        text = html.unescape(value)
        text = re.sub(r"<[^>]+>", " ", text)
        return " ".join(text.split())

    def _raw_tweet_from_news_item(self, item: Any, query: str) -> RawTweet | None:
        """Convert a Google News RSS item into the RawTweet schema."""
        title = self._clean_news_text(item.findtext("title"))
        if not title:
            return None

        link = self._clean_news_text(item.findtext("link"))
        guid = self._clean_news_text(item.findtext("guid"))
        source = self._clean_news_text(item.findtext("source")) or "Google News"
        description = self._clean_news_text(item.findtext("description"))
        published_text = self._clean_news_text(item.findtext("pubDate"))

        if published_text:
            created_at = parsedate_to_datetime(published_text)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            created_at = created_at.astimezone(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

        content = title
        if description and description.casefold() != title.casefold():
            content = f"{title}. {description}"

        identifier = guid or link or title
        tweet_id = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
        username = re.sub(r"[^a-z0-9_]+", "_", source.casefold()).strip("_") or "google_news"

        return RawTweet(
            tweet_id=tweet_id,
            created_at=created_at,
            username=username,
            display_name=source,
            content=content,
            url=link,
            like_count=0,
            retweet_count=0,
            lang="en",
            raw_payload={
                "tweet_id": tweet_id,
                "created_at": created_at.isoformat(),
                "username": username,
                "display_name": source,
                "content": content,
                "url": link,
                "lang": "en",
                "source_type": "google_news_rss",
                "query": query,
                "title": title,
                "description": description,
                "guid": guid,
                "link": link,
                "source": source,
                "published_at": published_text,
            },
        )

    @staticmethod
    def _raw_tweet_from_scraped_tweet(tweet: Any) -> RawTweet:
        """Convert a snscrape- or twscrape-style tweet object into RawTweet."""
        user = getattr(tweet, "user", None)
        content = getattr(tweet, "rawContent", None) or getattr(tweet, "renderedContent", "")
        raw_payload = json.loads(tweet.json()) if hasattr(tweet, "json") else {
            "tweet_id": str(getattr(tweet, "id", "")),
            "created_at": getattr(tweet, "date").isoformat(),
            "username": getattr(user, "username", "unknown"),
            "display_name": getattr(user, "displayname", "unknown"),
            "content": content,
            "url": getattr(tweet, "url", ""),
            "like_count": getattr(tweet, "likeCount", 0) or 0,
            "retweet_count": getattr(tweet, "retweetCount", 0) or 0,
            "lang": getattr(tweet, "lang", "unknown") or "unknown",
        }

        return RawTweet(
            tweet_id=str(getattr(tweet, "id", "")),
            created_at=getattr(tweet, "date"),
            username=getattr(user, "username", "unknown"),
            display_name=getattr(user, "displayname", "unknown"),
            content=content,
            url=getattr(tweet, "url", ""),
            like_count=getattr(tweet, "likeCount", 0) or 0,
            retweet_count=getattr(tweet, "retweetCount", 0) or 0,
            lang=getattr(tweet, "lang", "unknown") or "unknown",
            raw_payload=raw_payload,
        )

    @staticmethod
    def _raw_tweet_from_x_api_tweet(tweet: dict[str, Any], users: dict[str, dict[str, Any]]) -> RawTweet:
        """Convert an X API v2 tweet object into RawTweet."""
        author = users.get(str(tweet.get("author_id")), {})
        metrics = tweet.get("public_metrics", {})
        created_at = datetime.fromisoformat(str(tweet["created_at"]).replace("Z", "+00:00"))
        username = str(author.get("username", "unknown"))
        display_name = str(author.get("name", "unknown"))
        tweet_id = str(tweet.get("id", ""))

        raw_payload = {
            "tweet": tweet,
            "author": author,
        }

        return RawTweet(
            tweet_id=tweet_id,
            created_at=created_at,
            username=username,
            display_name=display_name,
            content=str(tweet.get("text", "")),
            url=f"https://x.com/{username}/status/{tweet_id}" if username != "unknown" else "",
            like_count=int(metrics.get("like_count", 0) or 0),
            retweet_count=int(metrics.get("retweet_count", 0) or 0),
            lang=str(tweet.get("lang", "unknown") or "unknown"),
            raw_payload=raw_payload,
        )
