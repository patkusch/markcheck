"""Work out what kind of file we are looking at, from its first bytes and its name."""
from __future__ import annotations

from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".heic", ".heif", ".avif", ".bmp", ".svg", ".dng"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
TEXT_EXT = {".txt", ".md", ".html", ".htm", ".json", ".csv", ".rtf"}
DOC_EXT = {".pdf", ".docx", ".pptx", ".xlsx"}


def sniff(path: Path, head: bytes) -> str:
    if head.startswith(b"\xff\xd8\xff") or head.startswith(b"\x89PNG") or head[:4] == b"GIF8":
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"mif1", b"msf1", b"avif"):
            return "image"
        if brand in (b"M4A ",):
            return "audio"
        return "video"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio"
    if head[:3] == b"ID3" or head[:4] == b"fLaC" or head[:4] == b"OggS":
        return "audio"
    if head[:4] == b"%PDF":
        return "pdf"
    ext = path.suffix.lower()
    for kind, exts in (("image", IMAGE_EXT), ("video", VIDEO_EXT), ("audio", AUDIO_EXT), ("pdf", {".pdf"}), ("document", DOC_EXT), ("text", TEXT_EXT)):
        if ext in exts:
            return kind
    return "other"
