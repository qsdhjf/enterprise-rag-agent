"""Start the product UI and API with explicit demo/live modes."""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Agent: evidence-first RAG workspace")
    parser.add_argument("--mode", choices=("demo", "workspace", "live"), default="demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--identity-file", help="Live mode: server-owned hashed bearer identities JSON")
    parser.add_argument("--knowledge-dir", help="Persistent private personal-document directory (single worker)")
    parser.add_argument("--enable-knowledge-model", action="store_true", help="Allow opt-in document excerpts to be sent to configured LLM")
    parser.add_argument("--enable-setup", action="store_true", help="Enable provider setup on loopback only; keys stay on the server")
    args = parser.parse_args()
    if args.mode in {"live", "workspace"} and not (args.identity_file or os.getenv("AGENT_IDENTITY_FILE")):
        parser.error("live/workspace mode requires --identity-file or AGENT_IDENTITY_FILE; see docs/DEPLOYMENT.md")
    os.environ["AGENT_MODE"] = args.mode
    if args.identity_file:
        os.environ["AGENT_IDENTITY_FILE"] = args.identity_file
    if args.knowledge_dir:
        os.environ["AGENT_KNOWLEDGE_DIR"] = args.knowledge_dir
    if args.enable_knowledge_model:
        os.environ["AGENT_KNOWLEDGE_MODEL_ENABLED"] = "1"
    if args.enable_setup:
        os.environ["AGENT_SETUP_ENABLED"] = "1"
    import uvicorn
    # Keep local-mode peer checks tied to the real connection, not forwarded headers.
    uvicorn.run("enterprise_agent.api:app", host=args.host, port=args.port, access_log=False, proxy_headers=False)


if __name__ == "__main__":
    main()
