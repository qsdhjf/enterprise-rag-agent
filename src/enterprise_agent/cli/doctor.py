"""Safe diagnostics: no secrets in output; no paid model calls."""

import argparse
import json
import os
import sys
from importlib.resources import files


def main() -> None:
    parser = argparse.ArgumentParser(description="Check installation/configuration without sending data to models")
    parser.add_argument("--mode", choices=("demo", "live"), default="demo")
    parser.add_argument("--identity-file")
    args = parser.parse_args()
    checks = [{"check": "python_3_12_or_newer", "ok": sys.version_info >= (3, 12)}]
    checks.append({"check": "web_assets", "ok": files("enterprise_agent.web").joinpath("static/index.html").is_file()})
    checks.append({"check": "demo_corpus", "ok": files("enterprise_agent.demo").joinpath("corpus.json").is_file()})
    if args.mode == "live":
        from enterprise_agent.infrastructure.config import load_embedding_settings, load_llm_settings, load_reranker_settings
        from enterprise_agent.infrastructure.opensearch_client import load_opensearch_settings
        from enterprise_agent.web.security import load_identities
        for name, loader in (("llm_config", load_llm_settings), ("embedding_config", load_embedding_settings),
                             ("reranker_config", load_reranker_settings), ("opensearch_config", load_opensearch_settings)):
            try:
                loader()
                checks.append({"check": name, "ok": True})
            except Exception:
                checks.append({"check": name, "ok": False, "help": "Check the documented environment variables; values are not logged"})
        try:
            load_identities(args.identity_file or os.environ.get("AGENT_IDENTITY_FILE", ""))
            checks.append({"check": "identity_file", "ok": True})
        except Exception:
            checks.append({"check": "identity_file", "ok": False, "help": "Create a scoped identity; see docs/DEPLOYMENT.md"})
    print(json.dumps({"mode": args.mode, "checks": checks,
                      "note": "Configuration only; does not prove network/model readiness or send paid model calls."}, indent=2))
    if not all(item["ok"] for item in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
