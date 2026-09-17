"""Bounded text extraction worker. Never render HTML, execute macros or fetch URLs.

Run in a disposable subprocess, supervised by KnowledgeService. Locations refer
to the extracted text (PDF pages, DOCX paragraphs, text line ranges), not pixels.
"""

import json
import re
import sys
import zipfile
from pathlib import Path

from defusedxml import ElementTree

MAX_BYTES = 8 * 1024 * 1024
MAX_TEXT = 500_000
MAX_CHUNKS = 1200
MAX_PAGES = 300
SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def validate_filename(name: str) -> str:
    if (not name or len(name) > 180 or name != name.strip()
            or any(c in name for c in '/\\:<>|?*')
            or any(ord(c) < 32 for c in name) or name in {".", ".."}):
        raise ValueError("文件名无效；请使用不含路径的普通文件名")
    suffix = Path(name).suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError("仅支持 UTF-8 TXT、Markdown、文本型 PDF 和 DOCX")
    return suffix


def extract(path: Path, suffix: str) -> dict:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("文件超过 8 MiB")
    sections = []
    warnings = []
    total = 0

    def add(location, text):
        nonlocal total
        text = text.replace("\x00", "").strip()
        total += len(text)
        if total > MAX_TEXT:
            raise ValueError("提取文本超过 50 万字符；请拆分文件")
        if text:
            sections.append((location, text))

    if suffix in {".txt", ".md"}:
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ValueError("检测到二进制或非 UTF-8 文本；请转换为 UTF-8")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("文本不是 UTF-8 编码；请转换后重试") from None
        # Small line groups retain a meaningful source location without a giant section.
        lines = text.splitlines()
        for start in range(0, len(lines), 12):
            add(f"第 {start + 1}–{min(start + 12, len(lines))} 行", "\n".join(lines[start:start + 12]))
    elif suffix == ".docx":
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if (len(entries) > 2000 or sum(e.file_size for e in entries) > 32 * 1024 * 1024
                        or any(e.file_size / max(1, e.compress_size) > 200 for e in entries)):
                    raise ValueError("DOCX 解压体积或压缩比超过安全限制")
                if len({e.filename for e in entries}) != len(entries):
                    raise ValueError("DOCX 包含重复条目")
                document = archive.getinfo("word/document.xml")
                if document.file_size > 8 * 1024 * 1024:
                    raise ValueError("DOCX 正文过大")
                root = ElementTree.fromstring(archive.read(document))
                namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                for number, paragraph in enumerate(root.findall(".//w:p", namespace), 1):
                    add(f"第 {number} 段", "".join(t.text or "" for t in paragraph.findall(".//w:t", namespace)))
                warnings.append("DOCX 提取正文及表格单元格文字，不保留表格布局、图片、页眉和批注。")
        except (zipfile.BadZipFile, KeyError):
            raise ValueError("不是有效的 DOCX 文件") from None
    elif suffix == ".pdf":
        from pypdf import PdfReader
        if not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError("文件内容不是 PDF")
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            raise ValueError("暂不支持加密 PDF；请先解密")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError("PDF 超过 300 页；请拆分文件")
        empty_pages = []
        for number, page in enumerate(reader.pages, 1):
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 8 * 1024 * 1024:
                raise ValueError("PDF 页面内容流过大")
            text = page.extract_text() or ""
            if not text.strip():
                empty_pages.append(number)
            add(f"第 {number} 页", text)
        if empty_pages:
            warnings.append(f"{len(empty_pages)} 页没有可提取文字（可能为扫描页）；未做 OCR，这些页不参与检索。")
        warnings.append("PDF 只提取文字；多栏、表格顺序和图片内容需要人工核对。")
    else:
        raise ValueError("不支持的文件格式")
    if not sections:
        raise ValueError("没有可提取文字；空文件或扫描件请先做 OCR 后重试")
    chunks = []
    for location, text in sections:
        for start in range(0, len(text), 700):
            piece = text[start:start + 900].strip()
            if piece:
                chunks.append({"ordinal": len(chunks) + 1, "location": location, "text": piece})
            if len(chunks) > MAX_CHUNKS:
                raise ValueError("切块超过 1200 个；请拆分文件")
            if start + 900 >= len(text):
                break
    return {"chunks": chunks, "warnings": warnings}


def main():
    try:
        # POSIX also has a kernel virtual-memory cap. Windows is monitored by parent RSS.
        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        result = extract(Path(sys.argv[1]), sys.argv[3])
    except ValueError as error:
        result = {"error": str(error)[:300]}
    except Exception:
        result = {"error": "文件解析失败；文件可能损坏、不受支持或超过资源限制"}
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
