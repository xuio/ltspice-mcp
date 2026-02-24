from __future__ import annotations

from pathlib import Path


def _decode_utf16_without_bom(blob: bytes) -> str:
    # LTspice on macOS commonly writes UTF-16LE without BOM.
    text_le = blob.decode("utf-16le", errors="replace")
    text_be = blob.decode("utf-16be", errors="replace")

    def score(text: str) -> int:
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
        replacement = text.count("\ufffd")
        return printable - replacement * 10

    return text_le if score(text_le) >= score(text_be) else text_be


def decode_text_bytes(blob: bytes) -> str:
    if blob.startswith(b"\xef\xbb\xbf"):
        return blob.decode("utf-8-sig", errors="replace")
    if blob.startswith(b"\xff\xfe"):
        return blob.decode("utf-16le", errors="replace")
    if blob.startswith(b"\xfe\xff"):
        return blob.decode("utf-16be", errors="replace")

    null_ratio = blob.count(b"\x00") / max(len(blob), 1)
    if null_ratio > 0.10:
        return _decode_utf16_without_bom(blob)
    return blob.decode("utf-8", errors="replace")


def read_text_auto(path: str | Path) -> str:
    file_path = Path(path)
    blob = file_path.read_bytes()
    return decode_text_bytes(blob).replace("\x00", "")
