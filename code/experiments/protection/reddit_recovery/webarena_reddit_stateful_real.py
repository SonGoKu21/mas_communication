"""Live, stateful WebArena Reddit edit-task primitives.

This adapter uses the deployed Postmill site and its normal authenticated HTML
forms.  It intentionally keeps the mutation workflow separate from the
single-step Shopping benchmark.
"""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup


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

    def __init__(
        self,
        base_url: str = "http://localhost:7771",
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def close(self) -> None:
        self.session.close()

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

    def resolve_forum(self, requested_forum: str) -> str:
        direct = self.session.get(
            f"{self.base_url}/f/{quote(requested_forum, safe='')}", timeout=20
        )
        if direct.status_code == 200:
            return requested_forum
        search = self.session.get(
            f"{self.base_url}/search", params={"q": requested_forum}, timeout=20
        )
        search.raise_for_status()
        soup = BeautifulSoup(search.text, "html.parser")
        candidates: list[tuple[str, str]] = []
        for anchor in soup.find_all("a", href=True):
            match = re.fullmatch(r"/f/([^/]+)", str(anchor["href"]))
            if match:
                candidates.append((anchor.get_text(" ", strip=True), match.group(1)))
        if not candidates:
            raise RuntimeError(f"Reddit forum was not found: {requested_forum}")
        requested = requested_forum.casefold()
        candidates.sort(
            key=lambda item: (
                item[0].casefold() != requested,
                not item[0].casefold().startswith(requested),
                len(item[0]),
            )
        )
        return candidates[0][1]

    def read_latest_submission(self, forum: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/f/{quote(forum, safe='')}/new", timeout=20)
        response.raise_for_status()
        submissions = parse_submission_entries(response.text, self.base_url, limit=1)
        if not submissions:
            raise RuntimeError(f"Reddit forum has no visible submissions: {forum}")
        return {"forum": forum, **submissions[0]}

    def read_user_comments(self, username: str) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{self.base_url}/user/{quote(username, safe='')}/comments", timeout=20
        )
        response.raise_for_status()
        return parse_comment_entries(response.text, self.base_url)

    def read_user_submissions(self, username: str) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{self.base_url}/user/{quote(username, safe='')}/submissions", timeout=20
        )
        response.raise_for_status()
        return parse_submission_entries(response.text, self.base_url, limit=100)

    def read_forum_submissions(self, forum: str, limit: int) -> list[dict[str, Any]]:
        response = self.session.get(f"{self.base_url}/f/{quote(forum, safe='')}", timeout=20)
        response.raise_for_status()
        submissions = parse_submission_entries(response.text, self.base_url, limit=limit)
        for submission in submissions:
            if submission.get("body") or not submission.get("url"):
                continue
            detail = self.session.get(str(submission["url"]), timeout=20)
            detail.raise_for_status()
            soup = BeautifulSoup(detail.text, "html.parser")
            body = soup.select_one(".submission__body")
            submission["body"] = body.get_text(" ", strip=True) if body else ""
        return submissions

    @staticmethod
    def _first(pattern: str, value: str) -> str:
        match = re.search(pattern, value, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _text(value: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _signed_integer(value: str) -> int:
    normalized = value.replace("−", "-")
    match = re.search(r"-?\d+", normalized)
    return int(match.group(0)) if match else 0


def parse_comment_entries(page_html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    comments: list[dict[str, Any]] = []
    for comment in soup.select(".comment"):
        score = comment.select_one(".vote__net-score")
        permalink = comment.select_one(".comment__permalink")
        body = comment.select_one(".comment__body")
        comments.append(
            {
                "comment_id": str(comment.get("id", "")).removeprefix("comment_"),
                "score": _signed_integer(score.get_text(" ", strip=True) if score else "0"),
                "text": body.get_text(" ", strip=True) if body else "",
                "url": urljoin(base_url.rstrip("/") + "/", permalink.get("href", ""))
                if permalink
                else "",
            }
        )
    return comments


def parse_submission_entries(
    page_html: str, base_url: str, *, limit: int
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    submissions: list[dict[str, Any]] = []
    for rank, submission in enumerate(soup.select(".submission")[:limit], start=1):
        title_anchor = submission.select_one(".submission__title a, a.submission__title")
        author_anchor = submission.select_one(".submission__submitter")
        body = submission.select_one(".submission__body")
        score = submission.select_one(".vote__net-score")
        internal_path = ""
        for anchor in submission.find_all("a", href=True):
            candidate = str(anchor["href"])
            if re.match(r"^/f/[^/]+/\d+/[^/]+$", candidate):
                internal_path = candidate
                break
        title_href = str(title_anchor.get("href", "")) if title_anchor else ""
        post_path = title_href if title_href.startswith("/f/") else internal_path
        post_match = re.search(r"/f/([^/]+)/(\d+)/", post_path)
        post_url = urljoin(base_url.rstrip("/") + "/", post_path) if post_path else ""
        submissions.append(
            {
                "rank": rank,
                "submission_id": int(post_match.group(2)) if post_match else None,
                "forum": post_match.group(1) if post_match else "",
                "title": title_anchor.get_text(" ", strip=True) if title_anchor else "",
                "author": author_anchor.get_text(" ", strip=True) if author_anchor else "",
                "body": body.get_text(" ", strip=True) if body else "",
                "url": post_url,
                "canonical_url": f"http://www.reddit.com{post_path}"
                if post_path
                else "",
                "external_url": urljoin(base_url.rstrip("/") + "/", title_href)
                if title_href
                else "",
                "score": _signed_integer(score.get_text(" ", strip=True) if score else "0"),
            }
        )
    return submissions


class RedditEvidenceWorker:
    """Collect task-relevant evidence from the deployed Postmill service."""

    def __init__(self, executor: Any) -> None:
        self.executor = executor

    def collect(self, task: dict[str, Any]) -> dict[str, Any]:
        parameters = task.get("instantiation_dict")
        parameters = parameters if isinstance(parameters, dict) else {}
        requested_forum = str(parameters.get("forum") or parameters.get("subreddit") or "")
        resolved_forum = self.executor.resolve_forum(requested_forum)
        task_stratum = str(task.get("task_stratum", ""))
        common = {
            "task_id": task["task_id"],
            "query_type": task_stratum,
            "requested_forum": requested_forum,
            "resolved_forum": resolved_forum,
            "forum_binding_exact": requested_forum.casefold()
            == resolved_forum.casefold(),
        }
        if task_stratum == "comment_state_query":
            latest = self.executor.read_latest_submission(resolved_forum)
            comments = self.executor.read_user_comments(str(latest["author"]))
            downvoted = [comment for comment in comments if int(comment["score"]) < 0]
            return {
                **common,
                "latest_submission": latest,
                "comments_total": len(comments),
                "downvoted_comment_count": len(downvoted),
                "downvoted_comments": downvoted,
            }
        if task_stratum == "top_post_semantic_query":
            requested_count = int(parameters.get("number", 10))
            return {
                **common,
                "requested_count": requested_count,
                "posts": self.executor.read_forum_submissions(
                    resolved_forum, requested_count
                ),
            }
        if task_stratum == "no_op_user_lookup":
            username = str(parameters.get("user", ""))
            submissions = self.executor.read_user_submissions(username)
            matching = [
                submission
                for submission in submissions
                if str(submission.get("forum", "")).casefold()
                == requested_forum.casefold()
            ]
            return {
                **common,
                "user": username,
                "user_submission_count": len(submissions),
                "matching_submission_count": len(matching),
                "matching_submissions": matching,
                "safe_no_op": not matching,
            }
        raise ValueError(f"unsupported Reddit evidence task stratum: {task_stratum}")
