"""Rate limiting functionality for API requests."""

import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, Optional


class RateLimiter:
    """Rate limiter with per-provider limits and adaptive throttling."""

    def __init__(self):
        self._limits: Dict[str, Dict[str, float]] = {}
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def set_limit(
        self,
        provider: str,
        requests_per_minute: int,
        requests_per_second: Optional[int] = None,
    ) -> None:
        """
        Set rate limits for a provider.

        Args:
            provider: Provider name
            requests_per_minute: Maximum requests per minute
            requests_per_second: Maximum requests per second (optional)
        """
        self._limits[provider] = {
            "per_minute": requests_per_minute,
            "per_second": requests_per_second or requests_per_minute // 60,
        }

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

    def _record_request(self, provider: str) -> None:
        """Record a request timestamp."""
        self._windows[provider].append(time.time())

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

    def reset_quota(self, provider: str) -> None:
        """Reset quota for a provider (for testing or manual override)."""
        if provider in self._windows:
            self._windows[provider].clear()

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
