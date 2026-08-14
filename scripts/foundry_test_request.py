"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Sanitized test script to diagnose connectivity to a Foundry endpoint.

Usage:
  - Set environment variables `FOUNDRY_ENDPOINT` and `FOUNDRY_TOKEN` (or pass them inline).
  - Run: `python scripts/foundry_test_request.py`

The script performs:
  1. Basic validation of the endpoint URL.
  2. DNS lookup and TCP connect to the host:port.
  3. A minimal POST request with `requests` while enabling urllib3 debug logging.

The token is never printed; any header output will have the token value replaced with "<REDACTED>".

Do NOT paste real tokens when sharing the output; the script already redacts them.

Also useful curl check (sanitized):
  curl -v -X POST "https://YOUR_ENDPOINT_URL/" \
    -H "Authorization: Bearer <REDACTED>" \
    -H "Content-Type: application/json" \
    -d '{"input_data":{"data":"{}"}}'

"""

import logging
import os
import socket
import sys
from urllib.parse import urlparse

import requests
from requests.exceptions import (
    ConnectionError as ReqConnectionError,
)
from requests.exceptions import (
    RequestException,
    Timeout,
)

LOG = logging.getLogger("foundry_test")


def mask_token(headers: dict) -> dict:
    h = dict(headers)
    if "Authorization" in h:
        h["Authorization"] = "Bearer <REDACTED>"
    return h


def dns_and_tcp_check(hostname: str, port: int, timeout: float = 5.0) -> None:
    try:
        ip = socket.gethostbyname(hostname)
        print(f"DNS OK: {hostname} -> {ip}")
    except Exception as exc:
        print(f"DNS lookup failed for {hostname}: {exc}")
        raise

    try:
        sock = socket.create_connection((hostname, port), timeout=timeout)
        sock.close()
        print(f"TCP OK: {hostname}:{port}")
    except Exception as exc:
        print(f"TCP connect failed to {hostname}:{port}: {exc}")
        raise


def run_minimal_post(endpoint: str, token: str, timeout: float = 10.0) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Enable urllib3 debug logging to show lower-level network behavior.
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)

    payload = {"input_data": {"data": json_encode_safe(None)}}

    print("Making POST request (response and any connection errors will be shown).\n")
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        print("Request headers sent (token redacted):")
        print(mask_token(headers))
        print(f"Response status: {resp.status_code}")
        print("Response text (first 1000 chars):")
        print(resp.text[:1000])
    except ReqConnectionError as exc:
        print(f"Connection error: {exc}")
        # Provide a bit more info for common socket reasons (sanitized).
        print("Hint: Connection reset by peer / Connection aborted often means the server closed the connection.")
        raise
    except Timeout as exc:
        print(f"Request timed out: {exc}")
        raise
    except RequestException as exc:
        print(f"Request failed: {exc}")
        raise


def json_encode_safe(obj):
    # Minimal JSON-safe placeholder for an empty payload.
    # Represented as an empty JSON object so the endpoint receives valid JSON.
    return {}


def main():
    endpoint = os.environ.get("FOUNDRY_ENDPOINT")
    token = os.environ.get("FOUNDRY_TOKEN")

    if not endpoint:
        print("Environment variable FOUNDRY_ENDPOINT is not set. Exiting.")
        sys.exit(2)
    if not token:
        print("Environment variable FOUNDRY_TOKEN is not set. Exiting.")
        sys.exit(2)

    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        print(f"FOUNDRY_ENDPOINT looks invalid: {endpoint}")
        sys.exit(2)

    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        dns_and_tcp_check(parsed.hostname, port)
    except Exception:
        print("Network preflight checks failed. Please verify endpoint and network connectivity.")
        sys.exit(1)

    # Attempt the POST. Exceptions will print details and re-raise.
    try:
        run_minimal_post(endpoint, token)
    except Exception:
        print("POST attempt failed. See above output for details.")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
