"""Download model files once, so markcheck can run offline afterwards.

    python -m markcheck.fetch_models          # Adobe TrustMark files (label check)
    python -m markcheck.fetch_models --text   # two small language models (text estimate)

markcheck itself never goes online. This helper is the only part that does,
and only when you run it. It asks the trustmark library to fetch the files
for the variant markcheck reads (Q, the library's default) from Adobe's
server. The library checks each file against a fixed checksum.

With --text it fetches the two language models named in
data/text_calibration.json from Hugging Face: only their settings,
tokenizer and safetensors weight files, nothing else.
"""
from __future__ import annotations

import argparse
import sys

from .detectors import trustmark_check


# Only the files the text check needs. Top-level files only ("*/*" skips other formats in sub-folders).
TEXT_ALLOW = ["*.json", "*.safetensors", "tokenizer*", "*.txt"]
TEXT_IGNORE = ["*/*"]


def fetch_text_models() -> int:
    from .textcheck import load_calibration

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("The text check's libraries are not installed. Install them with: pip install 'markcheck[text]'")
        return 2
    names = load_calibration()["models"]
    print(f"Fetching the language models for the text check from Hugging Face: {', '.join(names)}.")
    print("This is about 0.7 GB per model and happens once.")
    failed = []
    for name in names:
        try:
            path = snapshot_download(repo_id=name, allow_patterns=TEXT_ALLOW, ignore_patterns=TEXT_IGNORE)
        except Exception as exc:  # network, disk or permission problems
            print(f"Could not fetch {name}: {exc}")
            failed.append(name)
            continue
        print(f"  {name}: saved in {path}")
    if failed:
        print(f"Some models could not be fetched: {', '.join(failed)}")
        return 2
    print("Done. markcheck text can now run without going online.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markcheck.fetch_models",
                                 description="Download model files once, so markcheck can run offline.")
    ap.add_argument("--text", action="store_true",
                    help="fetch the two language models for 'markcheck text' instead of the TrustMark files")
    args = ap.parse_args(argv)
    if args.text:
        return fetch_text_models()
    try:
        from trustmark import TrustMark
    except ImportError:
        print("The trustmark library is not installed. Install it with: pip install 'markcheck[trustmark]'")
        return 2
    variant = trustmark_check.VARIANT
    missing = trustmark_check.missing_files(variant)
    if not missing:
        print(f"TrustMark model files (variant {variant}) are already on this computer. Nothing to do.")
        return 0
    print(f"Fetching TrustMark model files (variant {variant}) from Adobe: {', '.join(missing)}")
    TrustMark(verbose=False, model_type=variant, device="cpu", loadRemover=False)
    still = trustmark_check.missing_files(variant)
    if still:
        print(f"Some files could not be fetched: {', '.join(still)}")
        return 2
    print("Done. markcheck can now check for TrustMark without going online.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
