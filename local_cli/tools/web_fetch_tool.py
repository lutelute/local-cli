"""Web fetch tool for retrieving URL content.

Uses ``urllib.request`` to fetch the content of a URL and returns the
text with HTML tags stripped.  Handles redirects, timeouts, SSL errors,
and non-text content types gracefully.
"""

import re
import ssl
import urllib.error
import urllib.request

from local_cli.tools.base import Tool

# Default maximum content length in characters.
_DEFAULT_MAX_LENGTH = 50000

# Request timeout in seconds.
_REQUEST_TIMEOUT = 30


class WebFetchTool(Tool):
    """Fetch the text content of a URL."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch the content of a URL and return it as plain text. "
            "HTML tags are stripped automatically. Useful for reading "
            "web pages, documentation, or API responses."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "max_length": {
                    "type": "integer",
                    "description": (
                        "Maximum number of characters to return. "
                        f"Defaults to {_DEFAULT_MAX_LENGTH}."
                    ),
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: object) -> str:
        """Fetch URL content and return it as plain text.

        Args:
            **kwargs: Must include ``url`` (str).  May include
                ``max_length`` (int, default 50000).

        Returns:
            The plain-text content of the page, or an error message
            if the fetch fails.
        """
        url = kwargs.get("url")
        if not isinstance(url, str) or not url.strip():
            return "Error: 'url' parameter is required and must be a non-empty string."

        max_length = kwargs.get("max_length", _DEFAULT_MAX_LENGTH)
        if not isinstance(max_length, (int, float)):
            max_length = _DEFAULT_MAX_LENGTH
        max_length = max(1, int(max_length))

        # Build request with a reasonable User-Agent.
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "local-cli/1.0"},
        )

        try:
            response = urllib.request.urlopen(
                request, timeout=_REQUEST_TIMEOUT
            )
        except urllib.error.HTTPError as exc:
            return f"Error: HTTP {exc.code} {exc.reason} for URL: {url}"
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            if isinstance(exc.reason, ssl.SSLError):
                return f"Error: SSL error fetching URL: {reason}"
            return f"Error: could not fetch URL: {reason}"
        except ValueError:
            return f"Error: invalid URL: {url}"
        except OSError as exc:
            return f"Error: network error: {exc}"

        # Reject non-text content types.
        content_type = response.headers.get("Content-Type", "")
        if content_type and not any(
            t in content_type.lower()
            for t in ("text/", "application/json", "application/xml")
        ):
            return (
                f"Error: non-text content type '{content_type}' "
                f"for URL: {url}"
            )

        # Read and decode response body.
        try:
            raw_bytes = response.read()
        except OSError as exc:
            return f"Error: failed to read response: {exc}"

        # Determine encoding from Content-Type header, default to utf-8.
        encoding = "utf-8"
        if "charset=" in content_type.lower():
            charset_part = content_type.lower().split("charset=")[-1]
            encoding = charset_part.split(";")[0].strip()

        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = raw_bytes.decode("utf-8", errors="replace")

        # Strip HTML tags with a simple regex.
        text = re.sub(r"<[^>]+>", "", text)

        # Collapse excessive whitespace.
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # Truncate to max_length.
        if len(text) > max_length:
            text = text[:max_length] + "\n... [content truncated]"

        return text
