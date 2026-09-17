"""Create an allowlisted source release, with content hashes and secret-pattern checks."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {
    "README.md", "README.zh-CN.md", "LICENSE", "NOTICE", "CONTRIBUTING.md", "SECURITY.md",
    "CODE_OF_CONDUCT.md", "CHANGELOG.md", "ROADMAP.md", "pyproject.toml", "uv.lock",
    "Dockerfile", "compose.demo.yaml", "compose.workspace.yaml", ".env.example", ".gitignore", ".dockerignore",
    "environment.yml", "requirements-conda.lock", "package.json", "package-lock.json",
}
DIRECTORIES = {"src", "tests", "docs", ".github", "scripts", "infra", "eval"}
SUFFIXES = {".py", ".ps1", ".md", ".json", ".jsonl", ".toml", ".yml", ".yaml", ".js", ".cjs", ".css", ".html", ".png", ".svg", ".txt"}
SECRET_PATTERNS = [re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"), re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
                   re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"), re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]


def collect_files():
    result = []
    candidates = [ROOT / name for name in TOP_FILES]
    for directory in DIRECTORIES:
        candidates.extend((ROOT / directory).rglob("*"))
    for path in sorted(candidates):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if any(part in {"__pycache__", ".venv", ".local", "node_modules", ".git", "artifacts", ".pytest_cache"} for part in relative.parts):
            continue
        included = (relative.as_posix() in TOP_FILES or
                    (relative.parts[0] in DIRECTORIES and path.suffix in SUFFIXES))
        if not included:
            continue
        if any(part.startswith(".env") for part in relative.parts) and relative.as_posix() != ".env.example":
            raise ValueError("Private environment file in release allowlist")
        if "identities" in path.name.lower():
            raise ValueError("Identity file in release allowlist")
        if path.name == "provider-settings.private.json":
            raise ValueError("Private provider configuration in release allowlist")
        if path.suffix != ".png":
            content = path.read_text(encoding="utf-8")
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                raise ValueError(f"Potential credential detected in {relative}; inspect privately (value withheld)")
        result.append(path)
    required = TOP_FILES - {"requirements-conda.lock", "environment.yml"}
    absent = required - {path.relative_to(ROOT).as_posix() for path in result}
    if absent:
        raise ValueError(f"Missing release files: {sorted(absent)}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    digest_path = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or digest_path.exists():
        parser.error("Release or digest already exists; choose a new destination")
    files = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            manifest[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr("enterprise-agent/" + relative, data)
        archive.writestr("enterprise-agent/release-manifest.json", json.dumps(manifest, indent=2))
    with output.open("rb") as archive:
        digest = hashlib.file_digest(archive, "sha256").hexdigest()
    with digest_path.open("x", encoding="utf-8") as target:
        target.write(f"{digest}  {output.name}\n")
    print(f"Release verified: {len(files)} allowlisted files. SHA-256: {digest}")
    print(output)
    print("Pattern scanning does not replace manual security/license review. Nothing was published online.")


if __name__ == "__main__":
    main()
