"""Online SQLite backup. Contains private files, extracted text and session hashes."""

import argparse
import os
import sqlite3
from pathlib import Path


def backup(source: Path, destination: Path):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Backup must not replace the active database")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the exact new path with restrictive POSIX permissions; never overwrite.
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        reader = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        writer = sqlite3.connect(destination)
        try:
            if reader.execute("PRAGMA user_version").fetchone()[0] not in {1, 2, 3}:
                raise ValueError("Not a compatible personal-knowledge database")
            reader.backup(writer, pages=128, sleep=.05)
            if writer.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Backup integrity check failed")
        finally:
            writer.close()
            reader.close()
    except BaseException:
        destination.unlink(missing_ok=True)  # Only our newly reserved failed backup.
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".local/knowledge"))
    parser.add_argument("--output", type=Path, required=True, help="NEW private backup file; never put in a public release")
    args = parser.parse_args()
    result = backup(args.data_dir / "knowledge.sqlite3", args.output)
    print(f"Backup verified: {result}")
    print("Contains private documents/session hashes. Encrypt and restrict access; see docs/KNOWLEDGE.md.")


if __name__ == "__main__":
    main()
