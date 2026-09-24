"""Live, stateful WebArena Reddit edit-task primitives.

This adapter uses the deployed Postmill site and its normal authenticated HTML
forms.  It intentionally keeps the mutation workflow separate from the
single-step Shopping benchmark.
"""

from __future__ import annotations

import html
import re
from typing import Any

import requests


def choose_edit_target(task: dict[str, Any], delivered_state: dict[str, Any]) -> tuple[str, int, str]:
    """Use the state delivered to the Editor; mismatches remain observable."""
    return (
        str(delivered_state.get("forum", task["forum"])),
        int(delivered_state.get("submission_id", task["submission_id"])),
        str(delivered_state.get("slug", task["slug"])),
    )


def build_edit_request(task: dict[str, Any], delivered_state: dict[str, Any]) -> dict[str, Any]:
    body = str(delivered_state.get("body", "")).rstrip()
    append = str(task["required_append"])
    return {
        "submission_id": choose_edit_target(task, delivered_state)[1],
        "body": body if append in body else f"{body}\n\n{append}".lstrip(),
    }


class RedditHTTPExecutor:
    """Authenticated reader for the locally deployed WebArena Postmill site."""

    def __init__(self, base_url: str = "http://localhost:7771") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def login(self, username: str = "MarvelsGrantMan136", password: str = "test1234") -> None:
        page = self.session.get(f"{self.base_url}/login", timeout=20)
        page.raise_for_status()
        token = self._first(r'name="_csrf_token" value="([^"]+)"', page.text)
        if not token:
            raise RuntimeError("Reddit login form did not expose a CSRF token")
        response = self.session.post(
            f"{self.base_url}/login_check",
            data={"_csrf_token": token, "_username": username, "_password": password, "_remember_me": "on"},
            allow_redirects=True,
            timeout=20,
        )
        response.raise_for_status()
        if "Log out" not in response.text:
            raise RuntimeError("Reddit login did not establish an authenticated session")

    def read_submission(self, forum: str, submission_id: int, slug: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/f/{forum}/{submission_id}/{slug}", timeout=20)
        response.raise_for_status()
        body_html = self._first(r'<div class="submission__body[^>]*>(.*?)</div>', response.text)
        title_html = self._first(r'<h1[^>]*class="submission__title[^>]*>(.*?)</h1>', response.text)
        return {
            "forum": forum,
            "submission_id": submission_id,
            "slug": slug,
            "title": self._text(title_html),
            "body": self._text(body_html),
            "url": response.url,
        }

    def update_submission(self, forum: str, submission_id: int, slug: str, body: str) -> None:
        edit_url = f"{self.base_url}/f/{forum}/{submission_id}/-/edit"
        page = self.session.get(edit_url, timeout=20)
        page.raise_for_status()
        token = self._first(r'name="submission\[_token\]"[^>]*value="([^"]+)"', page.text)
        title = self._first(r'<textarea[^>]*name="submission\[title\]"[^>]*>(.*?)</textarea>', page.text)
        if not token or not title:
            raise RuntimeError("Reddit edit form did not expose required submission fields")
        response = self.session.post(
            edit_url,
            data={
                "submission[title]": self._text(title),
                "submission[body]": body,
                "submission[userFlag]": "none",
                "submission[email]": "",
                "submission[_token]": token,
            },
            allow_redirects=True,
            timeout=20,
        )
        response.raise_for_status()

    @staticmethod
    def _first(pattern: str, value: str) -> str:
        match = re.search(pattern, value, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _text(value: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()
