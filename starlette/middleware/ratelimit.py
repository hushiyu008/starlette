from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass
class RateLimitConfig:
    max_requests: int
    window_seconds: int
    identifier: Callable[[Request], str] | None = None


@dataclass
class RateLimitStore:
    requests: list[float] = field(default_factory=list)


def rate_limit(
    max_requests: int,
    window_seconds: int = 60,
    identifier: Callable[[Request], str] | None = None,
) -> Callable[[Any], Any]:
    def decorator(endpoint: Any) -> Any:
        config = RateLimitConfig(
            max_requests=max_requests,
            window_seconds=window_seconds,
            identifier=identifier,
        )

        if inspect.iscoroutinefunction(endpoint):

            @functools.wraps(endpoint)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Response:
                request = _extract_request(args)
                return await _check_rate_limit(request, endpoint, config, args, kwargs)

            async_wrapper._rate_limit_config = config
            return async_wrapper
        else:

            @functools.wraps(endpoint)
            async def sync_wrapper(*args: Any, **kwargs: Any) -> Response:
                request = _extract_request(args)
                return await _check_rate_limit(request, endpoint, config, args, kwargs)

            sync_wrapper._rate_limit_config = config
            return sync_wrapper

    return decorator


def _extract_request(args: tuple[Any, ...]) -> Request:
    for arg in args:
        if isinstance(arg, Request):
            return arg
    if len(args) >= 2 and isinstance(args[1], Request):
        return args[1]
    if len(args) >= 1 and isinstance(args[0], Request):
        return args[0]
    raise ValueError("No Request object found in arguments")


async def _check_rate_limit(
    request: Request,
    endpoint: Callable[..., Any],
    config: RateLimitConfig,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Response:
    stores = request.scope["state"].get("_rate_limit_stores", {})

    if config.identifier is not None:
        identifier = config.identifier(request)
    elif request.client is not None:
        identifier = request.client.host
    else:
        identifier = "unknown"

    route_key = _get_route_key(request.scope)
    store_key = f"{route_key}:{identifier}"

    current_time = time.time()
    window_start = current_time - config.window_seconds

    if store_key not in stores:
        stores[store_key] = RateLimitStore()

    store = stores[store_key]
    store.requests = [t for t in store.requests if t > window_start]

    if len(store.requests) >= config.max_requests:
        return PlainTextResponse("Too Many Requests", status_code=429)

    store.requests.append(current_time)
    remaining = config.max_requests - len(store.requests)

    if inspect.iscoroutinefunction(endpoint):
        response = await endpoint(*args, **kwargs)
    else:
        response = await run_in_threadpool(endpoint, *args, **kwargs)

    if isinstance(response, Response):
        response.headers["X-RateLimit-Limit"] = str(config.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

    return response


def _get_route_key(scope: Scope) -> str:
    path = scope.get("path", "/")
    method = scope.get("method", "GET")
    return f"{method}:{path}"


class RateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        default_max_requests: int | None = None,
        default_window_seconds: int = 60,
        default_identifier: Callable[[Request], str] | None = None,
    ) -> None:
        self.app = app
        self.default_config: RateLimitConfig | None = None
        if default_max_requests is not None:
            self.default_config = RateLimitConfig(
                max_requests=default_max_requests,
                window_seconds=default_window_seconds,
                identifier=default_identifier,
            )
        self._stores: dict[str, RateLimitStore] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope.setdefault("state", {})
        scope["state"]["_rate_limit_stores"] = self._stores

        if self.default_config is None:
            await self.app(scope, receive, send)
            return

        endpoint = self._get_endpoint(scope)
        has_decorator_config = False
        if endpoint is not None:
            has_decorator_config = hasattr(endpoint, "_rate_limit_config")

        if has_decorator_config:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive, send)
        identifier = self._get_identifier(request, self.default_config)
        route_key = _get_route_key(scope)
        store_key = f"{route_key}:{identifier}"

        is_allowed, remaining = self._is_allowed(store_key, self.default_config)

        if not is_allowed:
            response = PlainTextResponse("Too Many Requests", status_code=429)
            await response(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-ratelimit-limit", str(self.default_config.max_requests).encode()))
                headers.append((b"x-ratelimit-remaining", str(remaining).encode()))
                message = dict(message)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)

    def _get_endpoint(self, scope: Scope) -> Any:
        app = scope.get("app")
        if app is None:
            return None

        router = getattr(app, "router", None)
        if router is None:
            return None

        routes = getattr(router, "routes", [])

        for route in routes:
            if not hasattr(route, "matches"):
                continue

            match_result = route.matches(scope)
            if not isinstance(match_result, tuple) or len(match_result) < 2:
                continue

            match, child_scope = match_result

            if hasattr(match, "value"):
                match_value = match.value
            else:
                match_value = match

            if match_value == 2:
                return child_scope.get("endpoint")

        return None

    def _get_identifier(self, request: Request, config: RateLimitConfig) -> str:
        if config.identifier is not None:
            return config.identifier(request)
        if request.client is not None:
            return request.client.host
        return "unknown"

    def _is_allowed(self, store_key: str, config: RateLimitConfig) -> tuple[bool, int]:
        current_time = time.time()
        window_start = current_time - config.window_seconds

        if store_key not in self._stores:
            self._stores[store_key] = RateLimitStore()

        store = self._stores[store_key]
        store.requests = [t for t in store.requests if t > window_start]

        remaining = config.max_requests - len(store.requests)

        if len(store.requests) >= config.max_requests:
            return False, 0

        store.requests.append(current_time)
        return True, remaining - 1
