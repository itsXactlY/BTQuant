from __future__ import annotations

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(self, transition) -> None:
        self.buffer.append(transition)

    def sample(self, n: int):
        n = min(n, len(self.buffer))
        return list(self.buffer)[:n]

    def __len__(self) -> int:
        return len(self.buffer)
