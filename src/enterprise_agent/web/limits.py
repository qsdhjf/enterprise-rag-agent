"""Bound request bodies before JSON decoding, including chunked requests."""

import asyncio
import re
from threading import BoundedSemaphore

from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app, maximum: int = 16384, *, upload_slots=2, body_slots=16, read_timeout=15):
        self.app = app
        self.maximum = maximum
        self.upload_slots = BoundedSemaphore(upload_slots)
        self.body_slots = BoundedSemaphore(body_slots)
        self.read_timeout = read_timeout

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        upload = re.fullmatch(r"/api/knowledge/libraries/[a-f0-9]{32}/documents(?:/[a-f0-9]{32})?", scope["path"])
        maximum = 8 * 1024 * 1024 if upload and scope["method"] in {"POST", "PUT"} else self.maximum
        headers = dict(scope.get("headers", []))
        if b"content-length" in headers:
            try:
                length = int(headers[b"content-length"])
                if length < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
            if length > maximum:
                return await JSONResponse({"detail": f"Request body exceeds {maximum} bytes"}, status_code=413)(scope, receive, send)
        slots = self.upload_slots if upload else self.body_slots
        if not slots.acquire(blocking=False):
            return await JSONResponse({"detail": "上传或请求并发已满，请稍后重试"}, status_code=429,
                                      headers={"Retry-After": "2"})(scope, receive, send)
        try:
            body = bytearray()
            try:
                # Overall read deadline, not a timer reset by every tiny chunk.
                async with asyncio.timeout(self.read_timeout):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        part = message.get("body", b"")
                        if len(body) + len(part) > maximum:
                            return await JSONResponse({"detail": f"Request body exceeds {maximum} bytes"}, status_code=413)(scope, receive, send)
                        body.extend(part)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await JSONResponse({"detail": "请求体读取超时，请检查网络后重试"}, status_code=408)(scope, receive, send)
            sent = False

            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            # Keep upload slots until the handler persisted/rejected the body.
            await self.app(scope, replay, send)
        finally:
            slots.release()
