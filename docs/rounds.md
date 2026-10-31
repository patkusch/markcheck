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
