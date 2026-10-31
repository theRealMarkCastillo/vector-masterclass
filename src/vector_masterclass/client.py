"""Bounded retry client for idempotent retrieval requests only."""

import math
import random
import time

import httpx


def search_with_retry(
    client: httpx.Client, payload: dict, *, token: str, attempts=3, budget_seconds=5.0
):
    if not 1 <= attempts <= 3 or not 0 < budget_seconds <= 30:
        raise ValueError("Retries require 1–3 attempts and a total budget in (0,30] seconds")
    deadline = time.monotonic() + budget_seconds
    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Client retry budget exhausted")
        response = client.post(
            "/search", json=payload, headers={"Authorization": f"Bearer {token}"}, timeout=remaining
        )
        if time.monotonic() > deadline:
            raise TimeoutError("Client retry budget exhausted during request")
        if response.status_code not in {429, 503, 504} or attempt == attempts - 1:
            response.raise_for_status()
            return response.json()
        try:
            delay = max(0, float(response.headers.get("Retry-After", "0")))
        except ValueError as error:
            # HTTP-date Retry-After is outside this local fixture's numeric contract.
            raise ValueError("Local retry client requires numeric Retry-After") from error
        if not math.isfinite(delay):
            raise ValueError("Retry-After must be finite")
        delay = max(delay, min(0.1 * 2**attempt, 1) + random.uniform(0, 0.05))
        if delay >= deadline - time.monotonic():
            raise TimeoutError("Retry-After exceeds remaining client budget")
        time.sleep(delay)
    raise AssertionError("unreachable")
