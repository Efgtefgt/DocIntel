"""Wall-clock budget for processing one document."""

from __future__ import annotations

import time

from docintel.core.exceptions import ProcessingTimeoutError


class Deadline:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def check(self, what: str = "processing") -> None:
        if self.expired():
            raise ProcessingTimeoutError(f"The {self._seconds:.0f}s document time limit was reached during {what}.")

    def cap(self, seconds: float) -> float:
        """Return ``seconds`` limited to the remaining budget (minimum 1 second)."""
        return max(1.0, min(seconds, self.remaining()))
