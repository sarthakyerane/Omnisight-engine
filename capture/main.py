"""
Capture Daemon — main entry point.

Responsibilities:
  1. Capture screenshots at a configurable interval.
  2. Deduplicate frames using perceptual hashing (pHash).
  3. Enforce privacy: skip capture for apps/windows on the denylist.
  4. Pause capture when the user has been idle beyond IDLE_TIMEOUT_SECONDS.
  5. Persist novel frames to disk + SQLite, then publish an event to Redis
     so the AI worker can process them asynchronously.
  6. Provide a system-tray icon with Pause/Resume/Quit controls.
"""
import json
import os
import sys
import threading
import queue
from datetime import datetime, timezone
from pathlib import Path

import mss
import mss.tools
import redis
from loguru import logger
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

from config import settings
from storage.database import SessionLocal, engine
from storage.models import Base, Frame
from capture.change_detection import compute_phash, is_novel_frame
from capture.tray_app import TrayApp

# ── Platform-specific idle / window helpers ────────────────────────────────────
if sys.platform == "win32":
    from capture.platform_win import get_idle_duration_seconds, get_active_window_info
elif sys.platform == "darwin":
    from capture.platform_macos import get_idle_duration_seconds, get_active_window_info
else:
    from capture.platform_linux import get_idle_duration_seconds, get_active_window_info


def _build_frame_dir() -> Path:
    """Return the directory for today's frames, creating it if necessary."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = Path(settings.DATA_DIR) / "frames" / date_str
    path.mkdir(parents=True, exist_ok=True)
    return path


class CaptureDaemon:
    """
    Orchestrates screenshot capture, deduplication, privacy filtering,
    and hand-off to the background save/publish worker.
    """

    def __init__(self) -> None:
        self.tray = TrayApp()
        self._last_hashes: dict[str, str] = {}  # monitor_key -> phash
        self._last_frame_date: str = ""          # "YYYY-MM-DD" — cache frame dir
        self._frame_dir: Path = Path()

        # Bounded queue prevents memory explosion when the save worker stalls.
        self.save_queue: queue.Queue = queue.Queue(maxsize=100)

        self._redis: redis.Redis | None = None

    # ── Startup / Shutdown ─────────────────────────────────────────────────────

    def startup(self) -> None:
        """Initialise DB schema and external connections. Called once before run()."""
        Base.metadata.create_all(bind=engine)
        self._connect_redis()
        self._frame_dir = _build_frame_dir()
        self._last_frame_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _connect_redis(self) -> None:
        try:
            client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            client.ping()
            self._redis = client
            logger.info("Connected to Redis successfully.")
        except redis.ConnectionError:
            logger.warning(
                "Could not connect to Redis. Frame events will not be queued. "
                "Start Redis or set REDIS_URL to enable the AI worker pipeline."
            )
            self._redis = None

    # ── Privacy ────────────────────────────────────────────────────────────────

    def _is_app_denied(self, app_name: str, window_title: str) -> bool:
        """Return True if the active app or window matches the privacy denylist."""
        app_lower = app_name.lower()
        title_lower = window_title.lower()
        return any(
            keyword in app_lower or keyword in title_lower
            for keyword in settings.DENYLIST_APPS
        )

    # ── Frame directory caching ────────────────────────────────────────────────

    def _get_frame_dir(self) -> Path:
        """Return today's frame directory, refreshing the cache at midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_frame_date:
            self._frame_dir = _build_frame_dir()
            self._last_frame_date = today
        return self._frame_dir

    # ── Capture loop ──────────────────────────────────────────────────────────

    def capture_loop(self) -> None:
        """Main capture loop — runs in a dedicated daemon thread."""
        logger.info("Capture loop started.")
        with mss.mss() as sct:
            while not self.tray.stop_event.is_set():
                if self.tray.is_paused:
                    self.tray.stop_event.wait(timeout=1.0)
                    continue

                idle_seconds = get_idle_duration_seconds()
                if idle_seconds > settings.IDLE_TIMEOUT_SECONDS:
                    self.tray.stop_event.wait(timeout=1.0)
                    continue

                app_name, window_title = get_active_window_info()

                # Privacy check — log app name only, never the window title
                if self._is_app_denied(app_name, window_title):
                    logger.debug(f"Privacy denylist matched for app: {app_name!r}. Skipping.")
                    self.tray.stop_event.wait(timeout=1.0)
                    continue

                timestamp = datetime.now(timezone.utc)
                frame_dir = self._get_frame_dir()

                for i, monitor in enumerate(sct.monitors[1:], start=1):
                    sct_img = sct.grab(monitor)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

                    phash = compute_phash(img)
                    monitor_key = f"monitor_{i}"
                    prior_hash = self._last_hashes.get(monitor_key)

                    if is_novel_frame(phash, prior_hash, settings.PHASH_THRESHOLD):
                        logger.debug(f"Novel frame on {monitor_key} (app={app_name!r})")
                        self._last_hashes[monitor_key] = phash

                        file_name = (
                            f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{monitor_key}.jpg"
                        )
                        file_path = frame_dir / file_name

                        try:
                            self.save_queue.put_nowait(
                                (img, timestamp, i, app_name, window_title, file_path, phash)
                            )
                        except queue.Full:
                            logger.error(
                                "Save queue full — dropping frame to prevent memory growth. "
                                f"File would have been: {file_path}"
                            )
                            img.close()
                    else:
                        img.close()

                self.tray.stop_event.wait(timeout=settings.CAPTURE_INTERVAL_SECONDS)

        logger.info("Capture loop exited.")

    # ── Save worker ───────────────────────────────────────────────────────────

    def save_worker(self) -> None:
        """
        Background thread: dequeues frames and performs disk I/O + DB inserts.

        On shutdown, drains the remaining queue so no frames are silently lost.
        """
        logger.info("Save worker started.")

        while not self.tray.stop_event.is_set():
            try:
                item = self.save_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._process_save_item(item)

        # Drain any remaining items after stop signal
        logger.info(f"Draining save queue ({self.save_queue.qsize()} items remaining)...")
        while not self.save_queue.empty():
            try:
                item = self.save_queue.get_nowait()
                self._process_save_item(item)
            except queue.Empty:
                break

        logger.info("Save worker stopped cleanly.")

    def _process_save_item(self, item: tuple) -> None:
        """Persist one frame: disk → DB → Redis."""
        img, ts, monitor_id, app_name, window_title, file_path, phash = item

        # Store relative path in DB so the database is portable
        relative_path = str(Path(file_path).relative_to(settings.DATA_DIR))

        with SessionLocal() as db:
            try:
                # 1. Disk I/O
                img.save(file_path, "JPEG", quality=85)
                img.close()

                # 2. Database insert
                new_frame = Frame(
                    ts=ts,
                    monitor_id=monitor_id,
                    app_name=app_name,
                    window_title=window_title,
                    path=relative_path,
                    phash=phash,
                )
                db.add(new_frame)
                db.commit()
                db.refresh(new_frame)
                logger.debug(f"Frame {new_frame.id} saved → {relative_path}")

                # 3. Publish to Redis (only after DB succeeds)
                if self._redis:
                    try:
                        event = {
                            "frame_id": new_frame.id,
                            "path": relative_path,
                            "app_name": app_name,
                            "ts": ts.isoformat(),
                        }
                        self._redis.lpush("frame_queue", json.dumps(event))
                    except redis.ConnectionError:
                        logger.warning(
                            f"Redis unavailable — frame {new_frame.id} saved to DB "
                            "but not queued for AI processing."
                        )

            except SQLAlchemyError as exc:
                logger.error(f"DB error saving frame: {exc}")
                db.rollback()
                img.close()
            except OSError as exc:
                logger.error(f"Disk I/O error saving frame to {file_path}: {exc}")
                img.close()
            except Exception as exc:
                logger.error(f"Unexpected error in save worker: {exc}")
                img.close()
            finally:
                self.save_queue.task_done()

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start all threads, block on the tray icon, then shut down cleanly."""
        logger.info("Initialising Capture Daemon...")
        self.startup()

        capture_thread = threading.Thread(
            target=self.capture_loop, name="CaptureLoop", daemon=True
        )
        save_thread = threading.Thread(
            target=self.save_worker, name="SaveWorker", daemon=True
        )

        capture_thread.start()
        save_thread.start()

        # TrayApp.run() blocks the main thread until the user clicks Quit
        self.tray.run()

        # After tray exits: wait for threads to finish cleanly
        logger.info("Shutting down — waiting for threads to finish...")
        capture_thread.join(timeout=5)
        self.save_queue.join()   # Ensure every queued frame is processed
        save_thread.join(timeout=10)

        logger.info("Capture Daemon stopped cleanly.")


def main() -> None:
    daemon = CaptureDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
