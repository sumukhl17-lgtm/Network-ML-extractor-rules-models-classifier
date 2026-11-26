import asyncio


class DecodeCapacity:
    def __init__(self, maximum: int) -> None:
        self._maximum = maximum
        self._active = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active >= self._maximum:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._active -= 1
