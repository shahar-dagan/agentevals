import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC


@dataclass
class BufferedLogRecord:
    timestamp: str
    level: str
    logger_name: str
    message: str
    exc_text: str | None = None


class RingBufferLogHandler(logging.Handler):
    def __init__(self, capacity: int = 1000):
        super().__init__()
        self._buffer: deque[BufferedLogRecord] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        from datetime import datetime

        self._buffer.append(
            BufferedLogRecord(
                timestamp=datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
                exc_text=record.exc_text,
            )
        )

    def get_text(self) -> str:
        lines = []
        for r in self._buffer:
            lines.append(f"[{r.timestamp}] {r.level} {r.logger_name}: {r.message}")
            if r.exc_text:
                lines.append(r.exc_text)
        return "\n".join(lines)


log_buffer = RingBufferLogHandler(capacity=1000)
