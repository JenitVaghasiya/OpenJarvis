"""Exponential backoff + jitter for the cloud calls a rollout makes.

The judge already had this (``evals/core/scorer.py``) — it was added after a run
where 93/100 GAIA judge calls 429'd and zeroed the whole bench. The rollout's own
two cloud paths did NOT:

* the ORCHESTRATOR call (``toolorchestra/unified.py``), and
* the EXPERT dispatch (``expert_registry.py``).

Without backoff a single transient 429 makes the expert return an empty/error
observation, the clean gate then rejects the whole trajectory ("error or empty
tool observation"), and the rollout is silently wasted. That's tolerable at
concurrency 12; at 100 it would shred the batch. Same policy as the judge:
retry only on transient errors, exponential + jittered, then give up and let the
caller record the failure.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

LOGGER = logging.getLogger(__name__)

MAX_RETRIES = 6
BASE_DELAY_S = 2.0
MAX_DELAY_S = 60.0

# Transient / server-side failures. A 400 (bad request) or 401 (bad key) is NOT
# here on purpose: retrying those just burns time on a deterministic failure.
RETRYABLE_MARKERS = (
    "429",
    "rate_limit",
    "rate limit",
    "overloaded",
    "timeout",
    "timed out",
    "503",
    "502",
    "500",
    "connection",
    "temporarily unavailable",
    "internalservererror",
    "apiconnectionerror",
    # An expert that returns 200-OK with an EMPTY body. The HTTP call "succeeds",
    # so nothing raises and no retry fires — the rollout just gets an empty
    # observation and the clean gate then bins the whole trajectory. Seen from the
    # OpenRouter-hosted Qwen 122B/397B (audit 2026-07-13: 6 of 72 rollouts). We
    # raise EmptyExpertResponse for it so it retries like any other transient.
    "empty expert response",
)


class EmptyExpertResponse(RuntimeError):
    """An expert returned 200-OK with no content — transient, worth retrying."""

    def __init__(self, model: str) -> None:
        super().__init__(f"empty expert response from {model}")

T = TypeVar("T")


def is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in RETRYABLE_MARKERS)


def with_backoff(fn: Callable[[], T], *, what: str = "cloud call") -> T:
    """Run ``fn``, retrying transient failures with exponential backoff + jitter.

    Re-raises the last exception on a non-retryable error or once the retry
    budget is exhausted, so the caller still sees the failure.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below
            last_exc = exc
            if attempt == MAX_RETRIES - 1 or not is_retryable(exc):
                raise
            delay = min(BASE_DELAY_S * (2**attempt), MAX_DELAY_S)
            delay += random.uniform(0.0, delay * 0.25)  # jitter: de-sync the herd
            LOGGER.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what,
                attempt + 1,
                MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


__all__ = ["with_backoff", "is_retryable", "MAX_RETRIES"]
