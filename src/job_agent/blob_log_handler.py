"""Azure Blob Storage logging handler.

Buffers log records in memory and flushes them to an **append blob** on a
configurable interval (default 30 s) or when the buffer exceeds a size
threshold.  Each day gets its own blob so logs are easy to browse in Storage
Explorer:

    <container>/
        job_agent_2026-03-03.log
        job_agent_2026-03-04.log
        ...

Enable by setting these environment variables:
    AZURE_BLOB_LOG_CONNECTION_STRING  – Storage account connection string
    AZURE_BLOB_LOG_CONTAINER          – Container name (default: "app-logs")
    AZURE_BLOB_LOG_PREFIX             – Blob name prefix (default: "job_agent")
    AZURE_BLOB_LOG_FLUSH_INTERVAL     – Seconds between flushes (default: 30)
"""

from __future__ import annotations

import atexit
import datetime
import logging
import threading
from io import BytesIO
from typing import Optional


logger = logging.getLogger(__name__)


class AzureBlobLogHandler(logging.Handler):
    """Buffered logging handler that writes to Azure Blob Storage append blobs."""

    def __init__(
        self,
        connection_string: str,
        container_name: str = "app-logs",
        blob_prefix: str = "job_agent",
        flush_interval: float = 30.0,
        level: int = logging.DEBUG,
    ) -> None:
        super().__init__(level=level)
        self._connection_string = connection_string
        self._container_name = container_name
        self._blob_prefix = blob_prefix
        self._flush_interval = flush_interval

        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._closed = False

        # Lazy-import so the handler module can be imported even when the
        # azure-storage-blob package is missing (it just won't work at
        # runtime).
        from azure.storage.blob import BlobServiceClient

        self._blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self._ensure_container()
        self._start_flush_timer()
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Container / blob helpers
    # ------------------------------------------------------------------

    def _ensure_container(self) -> None:
        """Create the blob container if it doesn't already exist."""
        try:
            container_client = self._blob_service_client.get_container_client(
                self._container_name
            )
            if not container_client.exists():
                self._blob_service_client.create_container(self._container_name)
                logger.info(
                    "Created blob log container: %s", self._container_name
                )
        except Exception:
            logger.exception("Failed to ensure blob container %s", self._container_name)

    def _blob_name(self) -> str:
        """Return the blob name for today, e.g. ``job_agent_2026-03-03.log``."""
        today = datetime.date.today().isoformat()
        return f"{self._blob_prefix}_{today}.log"

    def _get_append_blob_client(self):
        """Return an AppendBlobClient for today's blob, creating it if needed."""
        from azure.storage.blob import BlobType

        blob_client = self._blob_service_client.get_blob_client(
            container=self._container_name,
            blob=self._blob_name(),
        )
        try:
            blob_client.get_blob_properties()
        except Exception:
            # Blob doesn't exist yet — create a new append blob.
            blob_client.create_append_blob()
        return blob_client

    # ------------------------------------------------------------------
    # Flush mechanics
    # ------------------------------------------------------------------

    def _start_flush_timer(self) -> None:
        if self._closed:
            return
        self._timer = threading.Timer(self._flush_interval, self._flush_on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _flush_on_timer(self) -> None:
        """Called by the background timer thread."""
        try:
            self.flush()
        finally:
            self._start_flush_timer()

    def flush(self) -> None:  # noqa: D401
        """Write buffered records to the append blob."""
        with self._lock:
            if not self._buffer:
                return
            payload = "".join(self._buffer)
            self._buffer.clear()

        try:
            blob_client = self._get_append_blob_client()
            blob_client.append_block(
                BytesIO(payload.encode("utf-8")),
                length=len(payload.encode("utf-8")),
            )
        except Exception:
            # Avoid infinite recursion — don't use logger here.
            import sys
            print(
                f"[AzureBlobLogHandler] Failed to flush logs to blob: {self._blob_name()}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # logging.Handler interface
    # ------------------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            with self._lock:
                self._buffer.append(msg)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        self.flush()
        super().close()
