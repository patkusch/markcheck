"""The "only the maker can check this" section.

makers_for(kind, findings) returns a list of {"who", "where", "note"} dicts
(plus "checked", the date and way we confirmed the address).

These are hidden marks that only the company that added them can read.
Addresses were checked on 2026-10-31 as noted per entry; see docs/research.md.
"""
from __future__ import annotations

SYNTHID_SHARED = "Several companies use SynthID, so a hit does not say which one made the file."

_SYNTHID_MEDIA = {
    "who": "Google SynthID (also used by OpenAI, ElevenLabs and others)",
    "where": "synthid.com (SynthID Detector), or ask the Gemini app",
    "note": "Checks images, audio and video for Google's hidden mark; " + SYNTHID_SHARED[0].lower() + SYNTHID_SHARED[1:],
    "checked": "2026-10-31, deepmind.google/models/synthid",
}
_OPENAI_IMAGE = {
    "who": "OpenAI (images)",
    "where": "openai.com/verify",
    "note": "OpenAI's own checker for images its tools made (address from TechCrunch, 2026-05-19; the page refused our automated visit).",
    "checked": "unverified: page returned 403 to our fetch",
}
_ANTHROPIC_FILES = {
    "who": "Anthropic (files Claude makes)",
    "where": "claude.com/check-content",
    "note": "Looks for Claude's signed label in images, video and audio up to 100 MB; it does not check text.",
    "checked": "2026-10-31, page read",
}
_MICROSOFT = {
    "who": "Microsoft (Foundry and Microsoft AI)",
    "where": "ai.azure.com/nextgen/validate (Content Provenance Detection)",
    "note": "Microsoft's free site checks images, audio and video for its signed labels and hidden marks.",
    "checked": "2026-10-31, learn.microsoft.com provenance-disclosure page (updated 2026-10-21)",
}
_AMAZON = {
    "who": "Amazon (Titan and Nova Canvas images)",
    "where": "Amazon Bedrock watermark detection (in the Bedrock console or API; needs an AWS account)",
    "note": "Amazon's own check for the hidden mark its image models add.",
    "checked": "2026-10-31, AWS Nova Canvas service card",
}
_ELEVENLABS = {
    "who": "ElevenLabs (voice audio)",
    "where": "elevenlabs.io/ai-speech-classifier",
    "note": "Gives a likelihood that a clip was made with ElevenLabs; it reads only the first minute (address found by search, page not opened by us).",
    "checked": "partly: found via web search 2026-10-31, not opened",
}
_GOOGLE_TEXT = {
    "who": "Google SynthID (text from Gemini; Anthropic's text mark builds on it)",
    "where": "no public text checker found",
    "note": "Google marks Gemini text but we found no public way to check text; " + SYNTHID_SHARED[0].lower() + SYNTHID_SHARED[1:],
    "checked": "2026-10-31, deepmind.google/models/synthid lists no text checker",
}
_ANTHROPIC_TEXT = {
    "who": "Anthropic (Claude text)",
    "where": "Detection API, private preview for approved organisations",
    "note": "Claude's hidden text mark can only be checked through Anthropic's private-preview service.",
    "checked": "2026-10-31, claude.com/check-content page",
}
_OPENAI_TEXT = {
    "who": "OpenAI (ChatGPT and Codex text in the EU)",
    "where": "approved researchers only",
    "note": "OpenAI's text mark has no public checker (TechCrunch, 2026-10-05).",
    "checked": "per docs/research.md; not re-checked",
}

BY_KIND = {
    "image": [_SYNTHID_MEDIA, _OPENAI_IMAGE, _MICROSOFT, _AMAZON, _ANTHROPIC_FILES],
    "audio": [_SYNTHID_MEDIA, _ELEVENLABS, _MICROSOFT, _ANTHROPIC_FILES],
    "video": [_SYNTHID_MEDIA, _MICROSOFT, _ANTHROPIC_FILES],
    "text": [_GOOGLE_TEXT, _ANTHROPIC_TEXT, _OPENAI_TEXT],
    "pdf": [_GOOGLE_TEXT, _ANTHROPIC_TEXT, _OPENAI_TEXT],
    "document": [_GOOGLE_TEXT, _ANTHROPIC_TEXT, _OPENAI_TEXT],
}


def makers_for(kind: str, findings) -> list[dict]:
    """Return the maker-only checkers relevant to this kind of file."""
    return [dict(e) for e in BY_KIND.get(kind, [])]
