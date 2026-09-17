"""ASGI slow/chunked/concurrent request tests; no external server required."""

import asyncio
import json

import pytest
from starlette.responses import JSONResponse

from enterprise_agent.web.limits import BodyLimitMiddleware

UPLOAD="/api/knowledge/libraries/" + "a"*32 + "/documents"


def scope(path=UPLOAD, headers=()):
    return {"type":"http", "method":"POST", "path":path, "headers":list(headers), "http_version":"1.1"}


async def app(scope, receive, send):
    request=await receive()
    await JSONResponse({"received":len(request["body"])})(scope,receive,send)


async def call(middleware, receiver, context=None):
    messages=[]
    async def send(message):
        messages.append(message)
    await middleware(context or scope(),receiver,send)
    status=next((m["status"] for m in messages if m["type"]=="http.response.start"),None)
    return status,messages


async def tiny():
    return {"type":"http.request","body":b"ok","more_body":False}


def test_content_length_rejected_without_reading_body():
    async def forbidden():
        raise AssertionError("Must reject before reading body")
    status,_=asyncio.run(call(BodyLimitMiddleware(app),forbidden,scope(headers=[(b"content-length",b"99999999")])))
    assert status==413


@pytest.mark.parametrize("length", [b"-1",b"invalid"])
def test_invalid_length(length):
    assert asyncio.run(call(BodyLimitMiddleware(app),tiny,scope(headers=[(b"content-length",length)])))[0]==400


def test_slow_body_deadline_releases_slot():
    async def run():
        middleware=BodyLimitMiddleware(app,upload_slots=1,read_timeout=.025)
        async def slow():
            await asyncio.sleep(.1)
            return await tiny()
        assert (await call(middleware,slow))[0]==408
        assert (await call(middleware,tiny))[0]==200
    asyncio.run(run())


def test_trickling_chunks_cannot_reset_overall_deadline():
    async def run():
        middleware=BodyLimitMiddleware(app,read_timeout=.035)
        async def trickle():
            await asyncio.sleep(.01)
            return {"type":"http.request","body":b"x","more_body":True}
        assert (await call(middleware,trickle))[0]==408
    asyncio.run(run())


def test_upload_concurrency_is_bounded_and_slot_released():
    async def run():
        gate,entered=asyncio.Event(),asyncio.Event()
        middleware=BodyLimitMiddleware(app,upload_slots=1)
        async def slow():
            entered.set()
            await gate.wait()
            return await tiny()
        running=asyncio.create_task(call(middleware,slow))
        await entered.wait()
        status,messages=await call(middleware,tiny)
        assert status==429
        assert (b"retry-after",b"2") in messages[0]["headers"]
        # Small JSON requests have a separate bounded pool.
        assert (await call(middleware,tiny,scope("/ask")))[0]==200
        gate.set()
        assert (await running)[0]==200
        assert (await call(middleware,tiny))[0]==200
    asyncio.run(run())


def test_chunked_limit_even_if_declared_length_is_small():
    async def run():
        chunks=iter([{"type":"http.request","body":b"a"*10,"more_body":True},
                     {"type":"http.request","body":b"b"*10,"more_body":False}])
        async def receive():
            return next(chunks)
        middleware=BodyLimitMiddleware(app,maximum=16)
        assert (await call(middleware,receive,scope("/ask",[(b"content-length",b"1")])))[0]==413
        assert (await call(middleware,tiny,scope("/ask")))[0]==200
    asyncio.run(run())


def test_slot_held_while_handler_owns_upload_buffer():
    async def run():
        entered,release=asyncio.Event(),asyncio.Event()
        async def processing(scope,receive,send):
            await receive()
            entered.set()
            await release.wait()
            await JSONResponse({"ok":True})(scope,receive,send)
        middleware=BodyLimitMiddleware(processing,upload_slots=1)
        first=asyncio.create_task(call(middleware,tiny))
        await entered.wait()
        assert (await call(middleware,tiny))[0]==429
        release.set()
        assert (await first)[0]==200
    asyncio.run(run())


def test_disconnect_and_exception_release_slot():
    async def run():
        middleware=BodyLimitMiddleware(app,upload_slots=1)
        async def disconnect():
            return {"type":"http.disconnect"}
        assert (await call(middleware,disconnect))[0] is None
        assert (await call(middleware,tiny))[0]==200
        async def broken():
            raise RuntimeError("broken transport")
        with pytest.raises(RuntimeError):
            await call(middleware,broken)
        assert (await call(middleware,tiny))[0]==200
    asyncio.run(run())
