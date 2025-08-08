"""Rate limiting functionality for API requests."""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter with per-provider limits and adaptive throttling."""

    def __init__(self):
        self._limits: Dict[str, Dict[str, float]] = {}
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._last_failure: Dict[str, float] = {}
        self._burst_capacity: Dict[str, int] = {}
        self._burst_tokens: Dict[str, int] = defaultdict(int)

    def set_limit(
        self,
        provider: str,
        requests_per_minute: int,
        requests_per_second: Optional[int] = None,
        burst_capacity: Optional[int] = None,
    ) -> None:
        """
        Set rate limits for a provider.

        Args:
            provider: Provider name
            requests_per_minute: Maximum requests per minute
            requests_per_second: Maximum requests per second (optional)
            burst_capacity: Maximum burst requests allowed (optional)
        """
        self._limits[provider] = {
            "per_minute": requests_per_minute,
            "per_second": requests_per_second or requests_per_minute // 60,
        }
        if burst_capacity:
            self._burst_capacity[provider] = burst_capacity
            self._burst_tokens[provider] = burst_capacity

    async def acquire(self, provider: str) -> None:
        """
        Acquire permission to make a request.

        Args:
            provider: Provider name

        Blocks until request can be made within rate limits.
        """
        if provider not in self._limits:
            return  # No limits set, allow immediately

        async with self._locks[provider]:
            # Check for exponential backoff due to previous failures
            await self._apply_exponential_backoff(provider)
            await self._wait_for_capacity(provider)
            self._record_request(provider)

    async def _wait_for_capacity(self, provider: str) -> None:
        """Wait until there's capacity for another request."""
        limits = self._limits[provider]
        window = self._windows[provider]
        current_time = time.time()

        # Clean old requests from window (older than 1 minute)
        while window and current_time - window[0] > 60:
            window.popleft()

        # Check per-minute limit
        if len(window) >= limits["per_minute"]:
            # Wait until oldest request falls outside window
            sleep_time = 60 - (current_time - window[0]) + 0.1
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                await self._wait_for_capacity(provider)
                return

        # Check per-second limit if set
        if limits["per_second"]:
            recent_requests = [
                req_time for req_time in window if current_time - req_time < 1
            ]
            if len(recent_requests) >= limits["per_second"]:
                # Wait until we're under the per-second limit
                sleep_time = 1 - (current_time - recent_requests[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    await self._wait_for_capacity(provider)
                    return

    async def _apply_exponential_backoff(self, provider: str) -> None:
        """Apply exponential backoff if there have been recent failures."""
        current_time = time.time()
        failure_count = self._failure_counts[provider]

        if failure_count > 0:
            last_failure = self._last_failure.get(provider, 0)
            # Reset failure count if it's been more than 5 minutes since last failure
            if current_time - last_failure > 300:
                self._failure_counts[provider] = 0
                return

            # Calculate backoff: 2^failures seconds with jitter, max 60 seconds
            backoff_seconds = min(2**failure_count, 60)
            jitter = random.uniform(0.1, 0.3) * backoff_seconds
            total_backoff = backoff_seconds + jitter

            time_since_failure = current_time - last_failure
            if time_since_failure < total_backoff:
                remaining_backoff = total_backoff - time_since_failure
                logger.info(
                    f"Applying exponential backoff for {provider}: "
                    f"{remaining_backoff:.1f}s (failure count: {failure_count})"
                )
                await asyncio.sleep(remaining_backoff)

    def _record_request(self, provider: str) -> None:
        """Record a request timestamp."""
        self._windows[provider].append(time.time())
        # Refill burst tokens if applicable
        if provider in self._burst_capacity:
            current_time = time.time()
            # Refill one token per second, up to capacity
            if hasattr(self, "_last_refill"):
                elapsed = current_time - self._last_refill.get(provider, current_time)
                tokens_to_add = int(elapsed)
                if tokens_to_add > 0:
                    self._burst_tokens[provider] = min(
                        self._burst_capacity[provider],
                        self._burst_tokens[provider] + tokens_to_add,
                    )
                    if not hasattr(self, "_last_refill"):
                        self._last_refill = {}
                    self._last_refill[provider] = current_time

    def get_remaining_quota(self, provider: str) -> Dict[str, int]:
        """
        Get remaining quota for a provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary with remaining requests per minute/second
        """
        if provider not in self._limits:
            return {"per_minute": float("inf"), "per_second": float("inf")}

        limits = self._limits[provider]
        window = self._windows[provider]
        current_time = time.time()

        # Clean old requests
        while window and current_time - window[0] > 60:
            window.popleft()

        # Calculate remaining quota
        remaining_per_minute = max(0, limits["per_minute"] - len(window))

        recent_requests = [
            req_time for req_time in window if current_time - req_time < 1
        ]
        remaining_per_second = max(
            0, limits.get("per_second", float("inf")) - len(recent_requests)
        )

        return {
            "per_minute": remaining_per_minute,
            "per_second": remaining_per_second,
        }

    def record_failure(self, provider: str) -> None:
        """Record a failure for exponential backoff calculation."""
        current_time = time.time()
        self._failure_counts[provider] += 1
        self._last_failure[provider] = current_time
        logger.warning(
            f"Recorded failure for {provider} (count: {self._failure_counts[provider]})"
        )

    def record_success(self, provider: str) -> None:
        """Record a success, potentially reducing failure count."""
        if self._failure_counts[provider] > 0:
            # Reduce failure count on success, but don't go below 0
            self._failure_counts[provider] = max(0, self._failure_counts[provider] - 1)
            if self._failure_counts[provider] == 0:
                logger.info(f"Cleared failure backoff for {provider}")

    def reset_quota(self, provider: str) -> None:
        """Reset quota for a provider (for testing or manual override)."""
        if provider in self._windows:
            self._windows[provider].clear()
        # Also reset failure state
        self._failure_counts[provider] = 0
        if provider in self._last_failure:
            del self._last_failure[provider]

    def get_stats(self, provider: str) -> Dict[str, any]:
        """
        Get statistics for a provider.

        Args:
            provider: Provider name

        Returns:
            Statistics dictionary
        """
        if provider not in self._limits:
            return {"configured": False}

        limits = self._limits[provider]
        window = self._windows[provider]
        current_time = time.time()

        # Clean old requests
        while window and current_time - window[0] > 60:
            window.popleft()

        recent_requests = [
            req_time for req_time in window if current_time - req_time < 1
        ]

        return {
            "configured": True,
            "limits": limits,
            "requests_last_minute": len(window),
            "requests_last_second": len(recent_requests),
            "remaining_quota": self.get_remaining_quota(provider),
            "oldest_request_age": (current_time - window[0] if window else None),
        }
