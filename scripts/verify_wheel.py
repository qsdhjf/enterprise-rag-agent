"""Verify distributable archives contain product assets and no private environment."""

import argparse
import tarfile
import zipfile
from pathlib import Path


def safe_members(names):
    for name in names:
        parts = Path(name).parts
        assert not any(part in {".env", ".local", ".git", ".venv", "node_modules"} for part in parts), name
        assert not any(part.startswith("identities") for part in parts), name
        assert "provider-settings.private.json" not in parts, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheels = list(args.directory.glob("*.whl"))
    assert wheels, "No wheel found"
    required = {"enterprise_agent/web/static/index.html", "enterprise_agent/web/static/app.js",
                "enterprise_agent/web/static/style.css", "enterprise_agent/demo/corpus.json",
                "enterprise_agent/demo/cases.jsonl", "enterprise_agent/cli/serve.py"}
    required |= {"enterprise_agent/web/static/knowledge.js", "enterprise_agent/web/static/knowledge.css",
                 "enterprise_agent/knowledge/parser.py", "enterprise_agent/knowledge/store.py"}
    required |= {"enterprise_agent/web/static/setup.js", "enterprise_agent/knowledge/agent.py",
                 "enterprise_agent/knowledge/eval_fixture.json", "enterprise_agent/cli/evaluate_knowledge.py"}
    required.add("enterprise_agent/knowledge/resources.py")
    for path in wheels:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            safe_members(names)
            assert required <= set(names), required - set(names)
        print(f"Wheel verified: {path.name} ({len(names)} files)")
    for path in args.directory.glob("*.tar.gz"):
        with tarfile.open(path) as archive:
            names = archive.getnames()
            safe_members(names)
        print(f"Source distribution verified: {path.name} ({len(names)} files)")


if __name__ == "__main__":
    main()
