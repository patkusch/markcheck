# Build rounds

Each round: build, test, look for weak spots, improve. Numbers come from `scripts/measure.py`.

## Round 1 — first working version

- **Built:** signed-label check (C2PA) with the official trust list bundled, metadata tags (IPTC and China's AIGC field), Stable Diffusion's hidden mark, Adobe TrustMark, and the "only the maker can check this" list.
- **What failed on the way:**
  - The SDXL mark read back as noise (0 of 48 bits on one picture). diffusers writes it from a float picture with colours swapped, so the checker now reads both ways.
  - Plain test pictures with large smooth areas would not hold the Stable Diffusion mark at all. Fixture pictures now have texture.
  - With default settings the C2PA library contacted a server named in the file. Remote fetching and certificate-status lookups are now switched off, and a test proves no request is made.
- **Numbers (before: nothing built):**
  - Hits on labelled files: C2PA 4/4, metadata 7/7, SD mark 3/3, TrustMark 1/1.
  - False alarms on 12 unlabelled controls: 0.
  - Survival (still found / total):

| Label | original | JPEG q75 | 50% resize | screenshot-like |
|---|---|---|---|---|
| c2pa | 4/4 | 0/4 | 0/4 | 0/4 |
| metadata | 7/7 | 0/7 | 7/7 | 0/7 |
| sd_watermark | 3/3 | 0/3 | 0/3 | 3/3 |
| trustmark | 1/1 | 1/1 | 1/1 | 1/1 |

- **Tests:** 125 passed.

## Round 2 — more test files, honest survival table (test data and measurement)

- **Built:** 84 test files instead of 31. Stable Diffusion mark on 7 pictures (v1, v2, SDXL; three kinds of scene; 512 to 1024; four saved as quality-95 JPEGs). TrustMark on 5 pictures (256 to 1024, PNG and JPEG). C2PA labels on WebP, GIF, TIFF, WAV, MP4 and PDF too, plus the C2PA public test files from Adobe. 51 unlabelled controls: photo-like textures at several JPEG qualities, flat graphics, a chart, screenshots with text, grayscale, see-through, tiny, GIF, very wide and very tall, pure noise, and every unmarked picture the mark makers started from. Every fixture script uses fixed seeds; pictures come out the same on every run.
- **What failed on the way:**
  - **TrustMark false alarm (real bug).** One ordinary control said "TrustMark found". The library calls a mark present whenever its error check passes, and its weakest schema (BCH_3) passes by luck often: 14 of 350 unmarked pictures we tried (about 1 in 25). markcheck now counts how many bits the error check had to fix and only says "found" when a random picture would do that well less than once in a million times, the same bar as the Stable Diffusion check. All real marks we made still read with 0 bits fixed, before and after JPEG and resizing. A real BCH_3 mark that needs any fixing is now reported as not found; that is the price of not crying wolf.
  - Quality-95 JPEG keeps the Stable Diffusion mark only just (SD v2 at 30-32 of 32 bits with 30 needed; SDXL 40-48 of 48 with 42 needed).
  - The old 50% resize kept metadata while the JPEG copy dropped it, so the table mixed two things. Each action is now measured twice: metadata kept and metadata dropped.
- **Numbers (before → after):** labelled files C2PA 4 → 14, metadata 7 → 7, SD mark 3 → 7, TrustMark 1 → 5; controls 12 → 51; all found, 0 false alarms.

| Label | original | JPEG q75, metadata kept | JPEG q75, metadata dropped | 50% resize, metadata kept | 50% resize, metadata dropped | screenshot-like |
|---|---|---|---|---|---|---|
| c2pa | 7/11 (4 say changed) | 0/9 (9 say changed) | 0/11 | 0/9 (9 say changed) | 0/11 | 0/11 |
| metadata | 7/7 | 5/7 | 0/7 | 7/7 | 0/7 | 0/7 |
| sd_watermark | 7/7 | 0/7 | 0/7 | 0/7 | 0/7 | 7/7 |
| trustmark | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

  "Say changed" means the signed label is still in the file but reports the file changed after signing. That is the label working as designed, not surviving. The two metadata losses with metadata kept are PNG text tags, which have no place in a JPEG.

## Round 2, part two — output red-team (CLI and wording)

- **What failed:** 19 new tests failed before the fixes. The main ones:
  - The word "real" slipped into two explanations ("not that what it shows is real"). A test now scans every fixture's text and JSON output for banned judgement words.
  - An unreadable subfolder or a broken link inside a folder was skipped silently and the run exited 0. It now reports the problem and exits 2 (or 1 if a label was found elsewhere).
  - Any target under a parent folder named "venv" was skipped entirely.
  - Text output showed raw Python dicts, raw TrustMark bits and raw C2PA validation codes.
  - A signed label that does not mention AI was still called "AI-related".
  - The "only the maker can check this" list repeated for every file. It now prints once per file kind at the end (JSON keeps it per file).
  - A 200 MB non-media file was read whole. Non-media files are now read up to 16 MB.
- **What changed:** jargon (C2PA, trust list, certificate, XMP, IPTC, AIGC, hidden mark, payload) is explained in brackets the first time it appears in a run. `--json` now has `exit_code`, `summary` and `problems`, and stays valid JSON on errors. A test blocks all network sockets and runs every fixture: it still works.
- **Tests:** 125 → 244 passed (both round-2 agents combined).
