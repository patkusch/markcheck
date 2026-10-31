"""Download the Adobe TrustMark model files once, so markcheck can check for TrustMark offline.

    python -m markcheck.fetch_models

markcheck itself never goes online. This helper is the only part that does,
and only when you run it. It asks the trustmark library to fetch the files
for the variant markcheck reads (Q, the library's default) from Adobe's
server. The library checks each file against a fixed checksum.
"""
from __future__ import annotations

import sys

from .detectors import trustmark_check


def main() -> int:
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
