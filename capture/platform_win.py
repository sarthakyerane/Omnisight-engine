"""
Windows-specific helpers for idle detection and active window inspection.

Uses GetTickCount64 (not GetTickCount) to avoid the 49-day 32-bit wrap-around.
"""
import ctypes

import psutil
import win32gui
import win32process
from loguru import logger


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def get_idle_duration_seconds() -> float:
    """
    Returns the number of seconds since the last mouse or keyboard input.

    Uses GetTickCount64 to avoid the 49-day wrap-around present in GetTickCount.
    Returns 0.0 (not idle) on API failure so capture is never silently suppressed.
    """
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)

    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
        logger.warning("GetLastInputInfo failed — assuming user is active.")
        return 0.0

    # GetTickCount64 returns milliseconds since system boot; never wraps on 64-bit.
    tick_now: int = ctypes.windll.kernel32.GetTickCount64()
    millis_idle: int = tick_now - lii.dwTime
    return max(millis_idle, 0) / 1000.0


def get_active_window_info() -> tuple[str, str]:
    """
    Returns (app_name, window_title) for the foreground window.

    Falls back to ("Unknown", "") if the window handle is invalid or
    the process cannot be inspected (e.g. elevated process).
    """
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return "Unknown", ""

    title: str = win32gui.GetWindowText(hwnd) or ""

    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        app_name: str = process.name()
    except Exception as exc:
        logger.debug(f"Could not get process name for hwnd {hwnd}: {exc}")
        app_name = "Unknown"

    return app_name, title
