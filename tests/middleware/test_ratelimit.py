from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.ratelimit import RateLimitMiddleware, rate_limit
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from tests.types import TestClientFactory


def test_rate_limit_decorator_within_limit(
    test_client_factory: TestClientFactory,
) -> None:
    @rate_limit(max_requests=2, window_seconds=60)
    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Homepage", status_code=200)

    app = Starlette(
        routes=[Route("/", endpoint=homepage)],
        middleware=[Middleware(RateLimitMiddleware)],
    )

    client = test_client_factory(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "1"

    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limit_decorator_exceeds_limit(
    test_client_factory: TestClientFactory,
) -> None:
    @rate_limit(max_requests=2, window_seconds=60)
    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Homepage", status_code=200)

    app = Starlette(
        routes=[Route("/", endpoint=homepage)],
        middleware=[Middleware(RateLimitMiddleware)],
    )

    client = test_client_factory(app)

    client.get("/")
    client.get("/")

    response = client.get("/")
    assert response.status_code == 429
    assert response.text == "Too Many Requests"


def test_rate_limit_without_decorator_no_restriction(
    test_client_factory: TestClientFactory,
) -> None:
    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Homepage", status_code=200)

    app = Starlette(
        routes=[Route("/", endpoint=homepage)],
        middleware=[Middleware(RateLimitMiddleware)],
    )

    client = test_client_factory(app)

    for _ in range(10):
        response = client.get("/")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" not in response.headers


def test_rate_limit_default_config(
    test_client_factory: TestClientFactory,
) -> None:
    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Homepage", status_code=200)

    app = Starlette(
        routes=[Route("/", endpoint=homepage)],
        middleware=[Middleware(RateLimitMiddleware, default_max_requests=2, default_window_seconds=60)],
    )

    client = test_client_factory(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "1"

    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "0"

    response = client.get("/")
    assert response.status_code == 429


def test_rate_limit_decorator_overrides_default(
    test_client_factory: TestClientFactory,
) -> None:
    @rate_limit(max_requests=3, window_seconds=60)
    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Homepage", status_code=200)

    def other(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Other", status_code=200)

    app = Starlette(
        routes=[
            Route("/", endpoint=homepage),
            Route("/other", endpoint=other),
        ],
        middleware=[Middleware(RateLimitMiddleware, default_max_requests=2, default_window_seconds=60)],
    )

    client = test_client_factory(app)

    for _ in range(3):
        response = client.get("/")
        assert response.status_code == 200

    response = client.get("/")
    assert response.status_code == 429

    for _ in range(2):
        response = client.get("/other")
        assert response.status_code == 200

    response = client.get("/other")
    assert response.status_code == 429


def test_rate_limit_custom_identifier(
    test_client_factory: TestClientFactory,
) -> None:
    def custom_identifier(request: Request) -> str:
        return request.headers.get("X-User-Id", "anonymous")

    @rate_limit(max_requests=2, window_seconds=60, identifier=custom_identifier)
    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Homepage", status_code=200)

    app = Starlette(
        routes=[Route("/", endpoint=homepage)],
        middleware=[Middleware(RateLimitMiddleware)],
    )

    client = test_client_factory(app)

    response = client.get("/", headers={"X-User-Id": "user1"})
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Remaining"] == "1"

    response = client.get("/", headers={"X-User-Id": "user2"})
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Remaining"] == "1"

    response = client.get("/", headers={"X-User-Id": "user1"})
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Remaining"] == "0"

    response = client.get("/", headers={"X-User-Id": "user1"})
    assert response.status_code == 429

    response = client.get("/", headers={"X-User-Id": "user2"})
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limit_window_reset(
    test_client_factory: TestClientFactory,
) -> None:
    @rate_limit(max_requests=2, window_seconds=1)
    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("Homepage", status_code=200)

    app = Starlette(
        routes=[Route("/", endpoint=homepage)],
        middleware=[Middleware(RateLimitMiddleware)],
    )

    client = test_client_factory(app)

    base_time = time.time()

    with patch("time.time", return_value=base_time):
        client.get("/")
        client.get("/")
        response = client.get("/")
        assert response.status_code == 429

    with patch("time.time", return_value=base_time + 2):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["X-RateLimit-Remaining"] == "1"


def test_rate_limit_with_class_based_view(
    test_client_factory: TestClientFactory,
) -> None:
    from starlette.endpoints import HTTPEndpoint

    class Homepage(HTTPEndpoint):
        @rate_limit(max_requests=2, window_seconds=60)
        def get(self, request: Request) -> PlainTextResponse:
            return PlainTextResponse("Homepage", status_code=200)

    app = Starlette(
        routes=[Route("/", endpoint=Homepage)],
        middleware=[Middleware(RateLimitMiddleware)],
    )

    client = test_client_factory(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "1"

    client.get("/")

    response = client.get("/")
    assert response.status_code == 429
