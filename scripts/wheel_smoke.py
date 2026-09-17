"""Run with the interpreter of a NEW environment where the built wheel is installed."""

import os
import tempfile
import time
from importlib.metadata import version
from importlib.resources import files

from fastapi.testclient import TestClient
from enterprise_agent.web.api import app


def main():
    package = str(files("enterprise_agent"))
    assert "site-packages" in package, f"Expected installed wheel, got editable source: {package}"
    assert version("enterprise-agent") == "0.5.0"
    os.environ["AGENT_MODE"] = "demo"
    os.environ["AGENT_KNOWLEDGE_MODEL_ENABLED"] = "0"
    with tempfile.TemporaryDirectory(prefix="ea-wheel-check-") as directory:
        os.environ["AGENT_KNOWLEDGE_DIR"] = directory
        check_http()
    print("Fresh wheel installation passed: version, packaged UI, sample ask, upload and actual parser subprocess.")
    print(package)


def check_http():
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        response = client.post("/ask", json={"question": "Atlas archive"})
        assert response.status_code == 200
        result = response.json()
        assert result["mode"] == "demo" and result["evidence"]
        assert result["model_token_count"] == 0
        assert client.get("/static/knowledge.js").status_code == 200
        assert client.get("/static/setup.js").status_code == 200
        assert files("enterprise_agent.knowledge").joinpath("eval_fixture.json").is_file()
        assert files("enterprise_agent.knowledge").joinpath("resources.py").is_file()
        assert client.get("/api/knowledge/session").status_code == 200
        lib = client.post("/api/knowledge/libraries", json={"name": "Wheel test"}).json()["id"]
        base = f"/api/knowledge/libraries/{lib}"
        response = client.post(base + "/documents", params={"filename": "a.txt"}, content=b"Orchid launch is Friday")
        assert response.status_code == 202
        doc = response.json()["id"]
        for _ in range(100):
            parsed = client.get(base + f"/documents/{doc}").json()
            if parsed["status"] in {"ready", "failed"}:
                break
            time.sleep(.1)
        assert parsed["status"] == "ready", parsed
        result = client.post(base + "/ask", json={"question": "Orchid launch"}).json()
        assert "Friday" in result["answer"]
        assert client.get("/api/knowledge/storage").json()["documents"] == 1
        unchanged = client.put(base + f"/documents/{doc}", params={"filename":"a.txt"}, content=b"Orchid launch is Friday").json()
        assert unchanged["unchanged"] and unchanged["revision"] == 1


if __name__ == "__main__":
    main()
