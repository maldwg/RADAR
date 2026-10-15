"""Bounded retries and pacing for provider JSON requests."""
from __future__ import annotations

import time
import math
from email.utils import parsedate_to_datetime
from typing import Any

import requests


class JsonTransport:
    """Share HTTP mechanics while adapters own endpoints and response semantics."""
    def __init__(self, *, headers: dict[str, str] | None = None, timeout: float = 30,
                 pause: float = 1, retries: int = 3, max_wait: float = 60):
        """Configure bounded requests and retries without constructing a connection."""
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 for value in (timeout, pause, max_wait)) or timeout == 0:
            raise ValueError("HTTP timeout must be positive; pause and max_wait must be finite and nonnegative")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("HTTP retries must be a nonnegative integer")
        self.headers = headers or {}
        self.timeout, self.pause, self.retries, self.max_wait = timeout, pause, retries, max_wait
        self.last_request = 0.0
        self.session: requests.Session | None = None

    def retry_delay(self, response: requests.Response, attempt: int) -> float:
        """Interpret standard Retry-After; providers can override nonstandard headers."""
        value = response.headers.get("Retry-After", "")
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                delay = 2 ** attempt
        return min(self.max_wait, max(0.0, delay))

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Retry transient responses and reject malformed JSON objects."""
        for attempt in range(self.retries + 1):
            wait = self.pause - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            try:
                get = self.session.get if self.session is not None else requests.get
                response = get(url, params=params, headers=self.headers, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.retries:
                    raise
                time.sleep(min(self.max_wait, 2 ** attempt))
                continue
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                delay = self.retry_delay(response, attempt)
                response.close()
                time.sleep(delay)
                continue
            try:
                response.raise_for_status()
                payload = response.json()
            finally:
                response.close()
            if not isinstance(payload, dict):
                raise ValueError("provider response must be a JSON object")
            return payload
        raise RuntimeError("JSON request exhausted retries")
