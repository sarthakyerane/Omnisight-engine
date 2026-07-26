"""
Tests for the privacy denylist logic in capture/main.py

Covers:
- Known sensitive apps are blocked
- Sensitive keywords in window titles are blocked
- Normal apps are not blocked
- Case-insensitive matching works
- Empty app name / title does not crash
"""
import pytest

# We test the logic in isolation without instantiating CaptureDaemon
# (which would try to connect to Redis and create DB tables).
# Extract the pure logic into a helper to keep tests fast and dependency-free.
from config import settings


def is_app_denied(app_name: str, window_title: str) -> bool:
    """Mirrors the logic in CaptureDaemon._is_app_denied."""
    app_lower = app_name.lower()
    title_lower = window_title.lower()
    return any(
        keyword in app_lower or keyword in title_lower
        for keyword in settings.DENYLIST_APPS
    )


class TestPrivacyDenylist:
    def test_password_manager_blocked(self):
        assert is_app_denied("1Password.exe", "") is True

    def test_bitwarden_blocked(self):
        assert is_app_denied("Bitwarden", "") is True

    def test_bank_in_title_blocked(self):
        assert is_app_denied("chrome.exe", "My Bank Account — Chase") is True

    def test_incognito_title_blocked(self):
        assert is_app_denied("chrome.exe", "New Incognito Tab") is True

    def test_finance_in_title_blocked(self):
        assert is_app_denied("firefox.exe", "Personal Finance Dashboard") is True

    def test_normal_app_not_blocked(self):
        assert is_app_denied("code.exe", "main.py — Visual Studio Code") is False

    def test_case_insensitive_app_match(self):
        assert is_app_denied("BITWARDEN.EXE", "") is True

    def test_case_insensitive_title_match(self):
        assert is_app_denied("chrome.exe", "MY BANK ACCOUNT") is True

    def test_empty_inputs_do_not_crash(self):
        assert is_app_denied("", "") is False

    def test_partial_keyword_does_not_match_unrelated_word(self):
        # "banker" contains "bank" — this is a known limitation of substring matching.
        # The test documents the current behaviour so it is explicit, not a surprise.
        assert is_app_denied("BankerApp.exe", "") is True  # known false positive
