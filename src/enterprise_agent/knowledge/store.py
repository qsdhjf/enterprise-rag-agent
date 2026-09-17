"""Durable, owner-scoped SQLite store. Raw files are BLOBs, never user-named paths.

Each mutation uses a short transaction. Parsing and model calls never hold a DB
write lock. Revision checks keep deleted/replaced documents from reappearing.
"""

import hashlib
import json
import os
import secrets
import sqlite3
import shutil
from contextlib import contextmanager
from pathlib import Path
from time import time
from uuid import uuid4

from enterprise_agent.knowledge.parser import MAX_BYTES, validate_filename
from enterprise_agent.knowledge.resources import KnowledgeLimits, vector_reservation


class KnowledgeError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class KnowledgeStore:
    def __init__(self, path: Path, *, limits=None):
        self.path = path
        self.limits = limits or KnowledgeLimits.from_environment()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 3:
                raise RuntimeError("知识库由更新版本创建，请使用匹配的软件版本")
            if version in {1, 2}:
                self._check_disk(path.stat().st_size)
                backup_path = path.with_name("before-v3-" + uuid4().hex + ".sqlite3")
                descriptor = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
                backup = sqlite3.connect(backup_path)
                try:
                    db.backup(backup)
                finally:
                    backup.close()
                os.chmod(backup_path, 0o600)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS libraries(id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS library_owner ON libraries(owner);
                CREATE TABLE IF NOT EXISTS documents(
                    id TEXT PRIMARY KEY, library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, suffix TEXT NOT NULL, sha256 TEXT NOT NULL, raw BLOB NOT NULL,
                    size INTEGER NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL,
                    error TEXT, warnings TEXT NOT NULL DEFAULT '[]', created REAL NOT NULL, updated REAL NOT NULL,
                    UNIQUE(library_id, name));
                CREATE TABLE IF NOT EXISTS chunks(
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, location TEXT NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY(document_id, ordinal));
                CREATE TABLE IF NOT EXISTS vector_jobs(
                    library_id TEXT PRIMARY KEY REFERENCES libraries(id) ON DELETE CASCADE,
                    profile_hash TEXT NOT NULL, corpus_hash TEXT NOT NULL,
                    status TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL,
                    error TEXT, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS vectors(
                    document_id TEXT NOT NULL, ordinal INTEGER NOT NULL, revision INTEGER NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY(document_id,ordinal),
                    FOREIGN KEY(document_id,ordinal) REFERENCES chunks(document_id,ordinal) ON DELETE CASCADE);
            """)
            if version < 3:
                # Keep the v2 table shapes above so migration is one transaction.
                # Column checks also support a pre-release DB with partial DDL.
                additions = {"vectors": {"profile_hash": "TEXT NOT NULL DEFAULT ''",
                    "input_hash": "TEXT NOT NULL DEFAULT ''", "dimensions": "INTEGER NOT NULL DEFAULT 0"},
                    "vector_jobs": {"job_id": "TEXT NOT NULL DEFAULT ''", "dimensions": "INTEGER NOT NULL DEFAULT 4096",
                    "reused": "INTEGER NOT NULL DEFAULT 0", "embedded": "INTEGER NOT NULL DEFAULT 0",
                    "reserved_bytes": "INTEGER NOT NULL DEFAULT 0"}}
                for table, columns in additions.items():
                    present = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
                    for column, definition in columns.items():
                        if column not in present:
                            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                from enterprise_agent.knowledge.retrieval import embedding_hash, normalized
                old = list(db.execute("""SELECT v.*,c.text,d.name,j.profile_hash AS old_profile FROM vectors v
                    JOIN chunks c ON c.document_id=v.document_id AND c.ordinal=v.ordinal
                    JOIN documents d ON d.id=v.document_id JOIN vector_jobs j ON j.library_id=d.library_id
                    WHERE j.status='ready' AND d.status='ready' AND d.revision=v.revision"""))
                for row in old:
                    try:
                        vector = json.loads(row["vector"])
                        vector_reservation(len(vector))
                        normalized(vector, len(vector))
                    except (ValueError, TypeError, KnowledgeError):
                        continue  # Never trust unidentifiable/corrupt legacy vectors.
                    db.execute("UPDATE vectors SET profile_hash=?,input_hash=?,dimensions=? WHERE document_id=? AND ordinal=?",
                        (row["old_profile"], embedding_hash(row), len(vector), row["document_id"], row["ordinal"]))
            db.execute("PRAGMA user_version=3")

    @contextmanager
    def db(self, write=False):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def session(self, token: str | None, create=False):
        with self.db(True) as db:
            if token and len(token) < 200:
                row = db.execute("SELECT owner FROM sessions WHERE token_hash=? AND expires>?",
                                 (hashlib.sha256(token.encode()).hexdigest(), time())).fetchone()
                if row:
                    return row["owner"], None
            if not create:
                raise KnowledgeError(401, "请先打开知识库页面建立本机会话")
            token, owner = secrets.token_urlsafe(32), "local:" + uuid4().hex
            db.execute("DELETE FROM sessions WHERE expires<?", (time(),))
            db.execute("INSERT INTO sessions VALUES(?,?,?)",
                       (hashlib.sha256(token.encode()).hexdigest(), owner, time() + 365 * 86400))
            return owner, token

    @staticmethod
    def library(db, owner, library_id):
        row = db.execute("SELECT * FROM libraries WHERE id=? AND owner=?", (library_id, owner)).fetchone()
        if row is None:
            raise KnowledgeError(404, "知识库不存在或无权访问")
        return dict(row)

    def libraries(self, owner):
        with self.db() as db:
            return [dict(row) for row in db.execute("""SELECT l.id,l.name,l.created,
                COUNT(d.id) AS document_count, COALESCE(SUM(d.size),0) AS bytes
                FROM libraries l LEFT JOIN documents d ON d.library_id=l.id
                WHERE l.owner=? GROUP BY l.id ORDER BY l.created""", (owner,))]

    def create_library(self, owner, name):
        with self.db(True) as db:
            if db.execute("SELECT COUNT(*) FROM libraries").fetchone()[0] >= 200:
                raise KnowledgeError(409, "本服务已达到 200 个知识库上限；请联系管理员")
            if db.execute("SELECT COUNT(*) FROM libraries WHERE owner=?", (owner,)).fetchone()[0] >= 20:
                raise KnowledgeError(409, "每个身份最多创建 20 个知识库")
            item = {"id": uuid4().hex, "name": name.strip(), "created": time()}
            db.execute("INSERT INTO libraries VALUES(?,?,?,?)", (item["id"], owner, item["name"], item["created"]))
            return item

    def delete_library(self, owner, library_id):
        with self.db(True) as db:
            self.library(db, owner, library_id)
            db.execute("DELETE FROM libraries WHERE id=?", (library_id,))

    @staticmethod
    def public_document(row):
        item = dict(row)
        item.pop("raw", None)
        item["warnings"] = json.loads(item["warnings"])
        return item

    def documents(self, owner, library_id):
        with self.db() as db:
            self.library(db, owner, library_id)
            return [self.public_document(row) for row in db.execute("""SELECT d.id,d.library_id,d.name,d.suffix,
                d.sha256,d.size,d.revision,d.status,d.error,d.warnings,d.created,d.updated,
                (SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.id) AS chunk_count
                FROM documents d WHERE library_id=? ORDER BY created""", (library_id,))]

    def put_document(self, owner, library_id, name, raw, document_id=None):
        try:
            suffix = validate_filename(name)
        except ValueError as error:
            raise KnowledgeError(422, str(error)) from None
        if not raw or len(raw) > MAX_BYTES:
            raise KnowledgeError(413 if raw else 422, "文件不能为空，且不能超过 8 MiB")
        self._check_disk(len(raw))
        digest = hashlib.sha256(raw).hexdigest()
        with self.db(True) as db:
            self.library(db, owner, library_id)
            old = None
            if document_id:
                old = db.execute("SELECT * FROM documents WHERE id=? AND library_id=?", (document_id, library_id)).fetchone()
                if old is None:
                    raise KnowledgeError(404, "文档不存在")
            collision = db.execute("SELECT id FROM documents WHERE library_id=? AND name=?", (library_id, name)).fetchone()
            if collision and collision["id"] != document_id:
                raise KnowledgeError(409, "存在同名文档，请使用该文档的“替换文件”操作")
            if old and old["status"] == "ready" and old["sha256"] == digest and old["name"] == name:
                return {"id": document_id, "revision": old["revision"], "status": "ready", "name": name, "unchanged": True}
            count, size = db.execute("""SELECT COUNT(*),COALESCE(SUM(d.size),0) FROM documents d
                JOIN libraries l ON l.id=d.library_id WHERE l.owner=?""", (owner,)).fetchone()
            if (not old and count >= 200) or size - (old["size"] if old else 0) + len(raw) > 128 * 1024**2:
                raise KnowledgeError(413, "每个身份最多 200 个文件、总计 128 MiB；请先删除不需要的文件")
            service_bytes = db.execute("SELECT COALESCE(SUM(size),0) FROM documents").fetchone()[0]
            if service_bytes - (old["size"] if old else 0) + len(raw) > 512 * 1024**2:
                raise KnowledgeError(413, "本服务原文件总量达到 512 MiB 上限；请联系管理员")
            identity = document_id or uuid4().hex
            revision = old["revision"] + 1 if old else 1
            if old:
                db.execute("DELETE FROM chunks WHERE document_id=?", (identity,))
                db.execute("""UPDATE documents SET name=?,suffix=?,sha256=?,raw=?,size=?,revision=?,status='queued',
                    error=NULL,warnings='[]',updated=? WHERE id=?""", (name, suffix, digest, raw, len(raw), revision, time(), identity))
            else:
                db.execute("""INSERT INTO documents(id,library_id,name,suffix,sha256,raw,size,revision,status,created,updated)
                    VALUES(?,?,?,?,?,?,?,?,'queued',?,?)""", (identity, library_id, name, suffix, digest, raw, len(raw), revision, time(), time()))
            return {"id": identity, "revision": revision, "status": "queued", "name": name}

    def mutate_document(self, owner, library_id, document_id, action):
        with self.db(True) as db:
            self.library(db, owner, library_id)
            row = db.execute("SELECT status FROM documents WHERE id=? AND library_id=?", (document_id, library_id)).fetchone()
            if row is None:
                raise KnowledgeError(404, "文档不存在")
            if action == "delete":
                db.execute("DELETE FROM documents WHERE id=?", (document_id,))
            elif action == "retry":
                if row["status"] != "failed":
                    raise KnowledgeError(409, "只有失败的文档可以重试；更新内容请替换文件")
                db.execute("UPDATE documents SET status='queued',error=NULL,revision=revision+1,updated=? WHERE id=?", (time(), document_id))

    def preview(self, owner, library_id, document_id, offset=0, limit=50):
        with self.db() as db:
            self.library(db, owner, library_id)
            row = db.execute("SELECT * FROM documents WHERE id=? AND library_id=?", (document_id, library_id)).fetchone()
            if row is None:
                raise KnowledgeError(404, "文档不存在")
            result = self.public_document(row)
            result["chunks"] = [dict(c) for c in db.execute("SELECT ordinal,location,text FROM chunks WHERE document_id=? ORDER BY ordinal LIMIT ? OFFSET ?", (document_id, limit, offset))]
            result["chunk_count"] = db.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (document_id,)).fetchone()[0]
            return result

    def recover(self):
        with self.db(True) as db:
            db.execute("UPDATE documents SET status='queued',error=NULL WHERE status IN ('parsing','indexing')")
            # Network jobs interrupted mid-run require renewed explicit consent;
            # replaying them automatically could incur duplicate provider charges.
            db.execute("UPDATE vector_jobs SET status='failed',reserved_bytes=0,error='服务重启中断了向量任务；已保存批次可复用，请确认后继续构建' WHERE status IN ('queued','building')")

    def claim(self):
        with self.db(True) as db:
            row = db.execute("SELECT * FROM documents WHERE status='queued' ORDER BY updated LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE documents SET status='parsing',updated=? WHERE id=?", (time(), row["id"]))
            return dict(row)

    def active(self, document_id, revision):
        with self.db() as db:
            return db.execute("SELECT 1 FROM documents WHERE id=? AND revision=? AND status IN ('parsing','indexing')", (document_id, revision)).fetchone() is not None

    def phase(self, document_id, revision, status):
        with self.db(True) as db:
            db.execute("UPDATE documents SET status=?,updated=? WHERE id=? AND revision=?", (status, time(), document_id, revision))

    def finish(self, document_id, revision, result):
        if "error" not in result:
            self._check_disk(sum(len(chunk["text"].encode()) for chunk in result["chunks"]))
        with self.db(True) as db:
            if not db.execute("SELECT 1 FROM documents WHERE id=? AND revision=?", (document_id, revision)).fetchone():
                return  # Deleted/replaced while parser was running: never resurrect it.
            if "error" in result:
                db.execute("UPDATE documents SET status='failed',error=?,updated=? WHERE id=?", (result["error"], time(), document_id))
                return
            owner = db.execute("SELECT l.owner FROM libraries l JOIN documents d ON d.library_id=l.id WHERE d.id=?", (document_id,)).fetchone()[0]
            count = db.execute("SELECT COUNT(*) FROM chunks c JOIN documents d ON c.document_id=d.id JOIN libraries l ON d.library_id=l.id WHERE l.owner=?", (owner,)).fetchone()[0]
            if count + len(result["chunks"]) > 10000:
                db.execute("UPDATE documents SET status='failed',error='身份切块总量超过 10000；请减少文件',updated=? WHERE id=?", (time(), document_id))
                return
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            db.executemany("INSERT INTO chunks VALUES(?,?,?,?)", [(document_id, c["ordinal"], c["location"], c["text"]) for c in result["chunks"]])
            db.execute("UPDATE documents SET status='ready',error=NULL,warnings=?,updated=? WHERE id=?", (json.dumps(result["warnings"], ensure_ascii=False), time(), document_id))

    def corpus(self, owner, library_id):
        with self.db() as db:
            self.library(db, owner, library_id)
            return [dict(r) for r in db.execute("""SELECT c.*,d.name,d.revision,d.sha256 FROM chunks c
                JOIN documents d ON d.id=c.document_id WHERE d.library_id=? AND d.status='ready'""", (library_id,))]

    def verify_evidence(self, owner, library_id, evidence):
        with self.db() as db:
            self.library(db, owner, library_id)
            for item in evidence:
                row = db.execute("SELECT revision,status FROM documents WHERE id=? AND library_id=?", (item["document_id"], library_id)).fetchone()
                if row is None or row["revision"] != item["revision"] or row["status"] != "ready":
                    raise KnowledgeError(409, "问答期间资料已更新或删除；请重新提问")

    def vector_status(self, owner, library_id, profile_hash):
        from enterprise_agent.knowledge.retrieval import corpus_hash
        rows = self.corpus(owner, library_id)
        with self.db() as db:
            row = db.execute("SELECT * FROM vector_jobs WHERE library_id=?", (library_id,)).fetchone()
        if row is None:
            return {"status": "not_built", "done": 0, "total": len(rows), "error": None, "reused": 0, "embedded": 0}
        result = dict(row)
        result["stale"] = result["profile_hash"] != profile_hash or result["corpus_hash"] != corpus_hash(rows)
        if result["stale"] and result["status"] == "ready":
            result["status"] = "stale"
        return result

    @staticmethod
    def _corpus_rows(db, library_id):
        return [dict(r) for r in db.execute("""SELECT c.*,d.name,d.revision FROM chunks c
            JOIN documents d ON d.id=c.document_id WHERE d.library_id=? AND d.status='ready'""", (library_id,))]

    @staticmethod
    def _reusable(db, library_id, rows, profile_hash, dimensions):
        from enterprise_agent.knowledge.retrieval import embedding_hash, row_key
        metadata = {(r["document_id"], r["ordinal"]): r for r in db.execute("""SELECT v.document_id,v.ordinal,
            v.revision,v.input_hash FROM vectors v JOIN documents d ON d.id=v.document_id
            WHERE d.library_id=? AND v.profile_hash=? AND v.dimensions=?""", (library_id, profile_hash, dimensions))}
        return {row_key(row) for row in rows if (cached := metadata.get((row["document_id"], row["ordinal"])))
                and cached["revision"] == row["revision"] and cached["input_hash"] == embedding_hash(row)}

    def _vector_usage(self, db, owner=None):
        clause, params = ("WHERE l.owner=?", (owner,)) if owner else ("", ())
        used = db.execute(f"""SELECT COALESCE(SUM(length(CAST(v.vector AS BLOB))),0) FROM vectors v
            JOIN documents d ON d.id=v.document_id JOIN libraries l ON l.id=d.library_id {clause}""", params).fetchone()[0]
        reserved = db.execute(f"""SELECT COALESCE(SUM(j.reserved_bytes),0) FROM vector_jobs j
            JOIN libraries l ON l.id=j.library_id {clause}""", params).fetchone()[0]
        return used, reserved

    def storage_usage(self, owner):
        with self.db() as db:
            raw, documents = db.execute("""SELECT COALESCE(SUM(d.size),0),COUNT(*) FROM documents d
                JOIN libraries l ON l.id=d.library_id WHERE l.owner=?""", (owner,)).fetchone()
            used, reserved = self._vector_usage(db, owner)
        return {"raw_bytes": raw, "documents": documents, "raw_limit": 128 * 1024**2,
                "vector_bytes": used, "vector_reserved_bytes": reserved, "vector_limit": self.limits.owner_vector_bytes,
                "note": "向量额度按逻辑数据计算，不含 SQLite 页、WAL、备份或解析临时文件"}

    def _check_capacity(self, db, owner, reservation):
        global_used, global_reserved = self._vector_usage(db)
        used, reserved = self._vector_usage(db, owner)
        if global_used + global_reserved + reservation > self.limits.vector_bytes:
            raise KnowledgeError(413, "服务向量容量不足；请清理不需要的语义索引或联系管理员")
        if used + reserved + reservation > self.limits.owner_vector_bytes:
            raise KnowledgeError(413, "当前身份的向量额度不足；请清理不需要的语义索引")
        self._check_disk(global_reserved + reservation)

    def _check_disk(self, reservation=0):
        # Leave headroom for the SQLite WAL. This is a guard, not a hard disk quota.
        if shutil.disk_usage(self.path.parent).free < self.limits.minimum_free_bytes + reservation * 2:
            raise KnowledgeError(507, "磁盘可用空间低于安全余量；未继续写入，请联系管理员释放空间")

    def _vector_plan(self, db, owner, library_id, profile_hash, dimensions, force=False):
        from enterprise_agent.knowledge.retrieval import corpus_hash, row_key, embedding_text
        self.library(db, owner, library_id)
        if db.execute("SELECT 1 FROM documents WHERE library_id=? AND status!='ready'", (library_id,)).fetchone():
            raise KnowledgeError(409, "请先处理所有文件的解析失败/排队状态，再构建语义索引")
        rows = self._corpus_rows(db, library_id)
        if not rows or len(rows) > 2000:
            raise KnowledgeError(409, "语义索引需要 1–2000 个就绪切块；请拆分知识库")
        reused = set() if force else self._reusable(db, library_id, rows, profile_hash, dimensions)
        pending = [r for r in rows if row_key(r) not in reused]
        fingerprint = corpus_hash(rows)
        plan_id = hashlib.sha256(json.dumps([owner, library_id, profile_hash, dimensions, fingerprint,
            sorted(reused), bool(force)]).encode()).hexdigest()
        return {"plan_id": plan_id, "profile_hash": profile_hash, "corpus_hash": fingerprint,
                "total": len(rows), "reused": len(reused), "pending": len(pending), "dimensions": dimensions,
                "characters_to_send": sum(len(embedding_text(r)) for r in pending),
                "estimated_batches": (len(pending)+15)//16,
                "reservation_bytes": len(pending) * vector_reservation(dimensions), "force": bool(force)}

    def vector_plan(self, owner, library_id, profile_hash, dimensions, force=False):
        with self.db() as db:
            return self._vector_plan(db, owner, library_id, profile_hash, dimensions, force)

    def queue_vectors(self, owner, library_id, profile_hash, dimensions=4096, *, plan_id=None, force=False):
        with self.db(True) as db:
            plan = self._vector_plan(db, owner, library_id, profile_hash, dimensions, force)
            if plan_id is not None and plan_id != plan["plan_id"]:
                raise KnowledgeError(409, "文件、模型或已完成批次已变化；请重新查看预估并确认")
            old = db.execute("SELECT status FROM vector_jobs WHERE library_id=?", (library_id,)).fetchone()
            if old and old["status"] in {"queued", "building"}:
                raise KnowledgeError(409, "该库正在构建，请等待或取消当前任务")
            self._check_capacity(db, owner, plan["reservation_bytes"])
            job_id = uuid4().hex
            if force:
                # Keep the old payload until overwritten, but invalidate reuse metadata.
                db.execute("UPDATE vectors SET profile_hash='' WHERE document_id IN (SELECT id FROM documents WHERE library_id=?)", (library_id,))
            db.execute("""INSERT INTO vector_jobs(library_id,profile_hash,corpus_hash,status,done,total,error,updated,
                job_id,dimensions,reused,embedded,reserved_bytes) VALUES(?,?,?,'queued',?,?,NULL,?,?,?,?,0,?)
                ON CONFLICT(library_id) DO UPDATE SET profile_hash=excluded.profile_hash,
                corpus_hash=excluded.corpus_hash,status='queued',done=excluded.done,total=excluded.total,error=NULL,
                updated=excluded.updated,job_id=excluded.job_id,dimensions=excluded.dimensions,
                reused=excluded.reused,embedded=0,reserved_bytes=excluded.reserved_bytes""",
                (library_id, profile_hash, plan["corpus_hash"], plan["reused"], plan["total"], time(),
                 job_id, dimensions, plan["reused"], plan["reservation_bytes"]))
        return dict(plan, status="queued", job_id=job_id)

    def claim_vectors(self):
        with self.db(True) as db:
            row = db.execute("SELECT j.*,l.owner FROM vector_jobs j JOIN libraries l ON l.id=j.library_id WHERE j.status='queued' ORDER BY j.updated LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE vector_jobs SET status='building' WHERE library_id=?", (row["library_id"],))
            return dict(row)

    def vector_progress(self, job, done):
        with self.db(True) as db:
            cursor = db.execute("""UPDATE vector_jobs SET done=? WHERE library_id=? AND status='building'
                AND profile_hash=? AND corpus_hash=? AND job_id=?""", (done, job["library_id"], job["profile_hash"], job["corpus_hash"], job["job_id"]))
            if cursor.rowcount != 1:
                raise KnowledgeError(409, "向量构建已取消或知识库已删除")

    def _active_vector_job(self, db, job):
        from enterprise_agent.knowledge.retrieval import corpus_hash
        self.library(db, job["owner"], job["library_id"])
        active = db.execute("SELECT * FROM vector_jobs WHERE library_id=? AND status='building' AND job_id=?",
                            (job["library_id"], job["job_id"])).fetchone()
        if active is None:
            raise KnowledgeError(409, "向量任务已取消、重建或删除")
        rows = self._corpus_rows(db, job["library_id"])
        if corpus_hash(rows) != job["corpus_hash"]:
            raise KnowledgeError(409, "构建期间资料已更新，请重新确认索引构建")
        return rows, active

    def pending_vectors(self, job):
        from enterprise_agent.knowledge.retrieval import row_key
        with self.db() as db:
            rows, _ = self._active_vector_job(db, job)
            reused = self._reusable(db, job["library_id"], rows, job["profile_hash"], job["dimensions"])
            return [r for r in rows if row_key(r) not in reused]

    def save_vector_batch(self, job, batch, vectors):
        from enterprise_agent.knowledge.retrieval import row_key, embedding_hash, normalized
        if not batch or len(batch) > 16 or len(batch) != len(vectors):
            raise KnowledgeError(502, "向量批次长度无效")
        encoded = [json.dumps(normalized(v, job["dimensions"]), separators=(",", ":"), allow_nan=False) for v in vectors]
        unit = vector_reservation(job["dimensions"])
        if any(len(v.encode()) > unit for v in encoded):
            raise KnowledgeError(502, "向量数据超过预留空间")
        with self.db(True) as db:
            rows, active = self._active_vector_job(db, job)
            current = {row_key(r): r for r in rows}
            keys = [row_key(r) for r in batch]
            reusable = self._reusable(db, job["library_id"], rows, job["profile_hash"], job["dimensions"])
            if (len(set(keys)) != len(keys) or set(keys) & reusable or any(
                    key not in current or embedding_hash(current[key]) != embedding_hash(row)
                    for key,row in zip(keys,batch,strict=True))):
                raise KnowledgeError(409, "批次已完成或原文变化，拒绝重复写入")
            self._check_disk(sum(len(v) for v in encoded))
            reservation = active["reserved_bytes"] - len(batch)*unit
            if reservation < 0:
                raise KnowledgeError(409, "任务空间预留不匹配，请重新确认构建")
            db.executemany("""INSERT INTO vectors(document_id,ordinal,revision,vector,profile_hash,input_hash,dimensions)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(document_id,ordinal) DO UPDATE SET revision=excluded.revision,
                vector=excluded.vector,profile_hash=excluded.profile_hash,input_hash=excluded.input_hash,dimensions=excluded.dimensions""",
                [(r["document_id"],r["ordinal"],r["revision"],v,job["profile_hash"],embedding_hash(r),job["dimensions"])
                 for r,v in zip(batch,encoded,strict=True)])
            db.execute("""UPDATE vector_jobs SET done=done+?,embedded=embedded+?,reserved_bytes=?
                WHERE library_id=? AND job_id=?""", (len(batch),len(batch),reservation,job["library_id"],job["job_id"]))

    def finish_vectors(self, job):
        with self.db(True) as db:
            rows, _ = self._active_vector_job(db, job)
            if len(self._reusable(db, job["library_id"], rows, job["profile_hash"], job["dimensions"])) != len(rows):
                raise KnowledgeError(409, "向量不完整；不发布部分索引")
            db.execute("UPDATE vector_jobs SET status='ready',done=total,error=NULL,reserved_bytes=0 WHERE library_id=? AND job_id=?",
                       (job["library_id"], job["job_id"]))

    def fail_vectors(self, job, message):
        with self.db(True) as db:
            db.execute("UPDATE vector_jobs SET status='failed',error=?,reserved_bytes=0 WHERE library_id=? AND job_id=? AND status='building'", (message, job["library_id"], job["job_id"]))

    def cancel_vectors(self, owner, library_id):
        with self.db(True) as db:
            self.library(db, owner, library_id)
            db.execute("UPDATE vector_jobs SET status='failed',reserved_bytes=0,error='用户取消；已保存批次可复用，已发出的请求可能仍计费' WHERE library_id=? AND status IN ('queued','building')", (library_id,))

    def clear_vectors(self, owner, library_id):
        with self.db(True) as db:
            self.library(db, owner, library_id)
            db.execute("DELETE FROM vector_jobs WHERE library_id=?", (library_id,))
            db.execute("DELETE FROM vectors WHERE document_id IN (SELECT id FROM documents WHERE library_id=?)", (library_id,))

    def read_vectors(self, owner, library_id, profile_hash, fingerprint):
        with self.db() as db:
            self.library(db, owner, library_id)
            job = db.execute("SELECT * FROM vector_jobs WHERE library_id=?", (library_id,)).fetchone()
            if not job or job["status"] != "ready" or job["profile_hash"] != profile_hash or job["corpus_hash"] != fingerprint:
                raise KnowledgeError(409, "语义索引未就绪或已过期；请构建/重建，或切换关键词检索")
            return {f'{r["document_id"]}:{r["revision"]}:{r["ordinal"]}': json.loads(r["vector"]) for r in db.execute(
                "SELECT v.* FROM vectors v JOIN documents d ON d.id=v.document_id WHERE d.library_id=?", (library_id,))}
