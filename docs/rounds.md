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

## Round 3 — real AI files and an honesty audit

- **Built:** 7 files from real AI products in `tests/fixtures/realworld/` (OpenAI gpt-image from Wikimedia Commons, Adobe Photoshop + Firefly, an Adobe label kept online only, three Adobe TrustMark samples, an SDXL picture from AUTOMATIC1111), each with source and licence in `SOURCES.md`. A separate audit checked every sentence markcheck prints against what the code actually checked.
- **What failed on real files:**
  - OpenAI's real label read as "signature does not check out". Its certificate uses C2PA's own signing purpose, which the library rejects unless told to accept it. It now reads as signed and unchanged.
  - Adobe's real certificate was called one "anyone could have made". It is on the older interim list, which C2PA froze on 1 January 2026. That list is now bundled and reported on its own line.
  - A hidden mark named inside a signed label was dropped from the output. The SDXL model name in AUTOMATIC1111 settings was not shown. Both are shown now.
  - Adobe's TrustMark variant P samples are missed. markcheck reads only variant Q and now says which variants it did not read.
- **What failed in the audit:**
  - "Unchanged since signing" printed whenever no error came back. It now needs a confirmed signature and a confirmed file match.
  - A trusted signer on a changed file was called "not on the trust list". It now reads the library's own trust result.
  - TrustMark said "read back without errors" even when its error check fixed bits. It now says how many bits were fixed.
  - A comment gave a false-alarm figure the tests did not back up. A test now checks the comment's numbers.
  - A file where a check did not run (for example TrustMark without its model files) just said "No label found". It now adds "Some checks did not run on this file".
  - The wording test skipped several JSON fields. It now scans everything except file paths.
  - The lead found one more: a signer reported "not on the trust list" was also told "could not check against the trust list". Fixed, with a test.
- **Numbers (before → after):** labelled files C2PA 14 → 18, metadata 7 → 8, SD mark 7 → 7, TrustMark 5 → 6, all found; controls 51 → 52, 0 false alarms. Real files only: C2PA 4/4, metadata 1/1, TrustMark 1/1 found.

| Label | original | JPEG q75, metadata kept | JPEG q75, metadata dropped | 50% resize, metadata kept | 50% resize, metadata dropped | screenshot-like |
|---|---|---|---|---|---|---|
| c2pa | 10/14 (4 say changed) | 0/12 (12 say changed) | 0/14 | 0/12 (12 say changed) | 0/14 | 0/14 |
| metadata | 8/8 | 5/8 | 0/8 | 8/8 | 0/8 | 0/8 |
| sd_watermark | 7/7 | 0/7 | 0/7 | 0/7 | 0/7 | 7/7 |
| trustmark | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |

- **Tests:** 244 → 276 passed.

# Text check (`markcheck text`)

Numbers come from `scripts/measure_text.py`, on texts never used for fitting.
The data is fetched once and never committed: HC3 (CC-BY-SA-4.0), MAGE's out-of-domain GPT-4 files (Apache-2.0 / CC BY 4.0) and the 91 TOEFL essays from Liang et al. 2023 (no licence file, so used only for measuring).
How to read the numbers:

- **AUROC**: the chance that a random AI text scores higher than a random person's text. 0.5 is a coin toss; 1.0 is perfect.
- **Caught at 1 in 100**: the share of AI texts caught when the line is set so that only 1 in 100 people's texts is wrongly flagged.
- **Calibration error**: the average gap between the stated chance and how often it came true. 0 is perfect; 0.10 means the stated chances were off by about 10 points.
- **Above / below the band**: the share of texts put above or below the "can't tell" band (35-65%).

## Text round 1 — first working version

- **Built:** Binoculars score from SmolLM2-360M and SmolLM2-360M-Instruct (Apache-2.0, about 1.4 GB together, about 0.3-0.8 seconds per text on this laptop's CPU), plus two cheap measures: how often the next word was in the model's top 10 guesses, and how even the sentence lengths are. A logistic curve turns them into a chance. Fitted on 907 HC3 texts (half people, half ChatGPT), split by question so no question is in both halves.
- **What failed on the way:**
  - HC3 answers by people have stray spaces before full stops and commas. A detector could learn that instead of the writing. We tidy spacing on every text, AI and people's, before scoring.
  - Our first wording described sentence evenness as "unusually even" whatever the fitted model did with it. Reasons now say which way each measure moved this estimate.
  - Loading the models printed progress bars. They are now switched off.
- **Numbers (texts of 100 words or more):**

| Held-out set | Texts | AUROC | Caught at 1 in 100 | Calibration error | People above band | AI below band |
|---|---|---|---|---|---|---|
| HC3, same topics as training | 914 | 0.998 | 99% | 0.006 | 0.9% | 0.5% |
| MAGE, new topics, GPT-4 | 1,428 | 0.938 | 50% | 0.228 | 0.7% | 44.5% |
| MAGE reworded by another AI | 2,235 | 0.558 | 0.4% | 0.662 | 0.7% | 99.1% |
| TOEFL essays by non-native students | 63 | - | - | - | 4.8% | - |

- **By kind of writing (MAGE):** news (CNN) AUROC 0.81 with 75% of AI texts below the band; film reviews 0.996 but 60% of AI texts below the band; dialogue summaries 0.98; medical abstracts 0.996.
- **By length (HC3 + MAGE held-out, before the minimum):** 30-59 words AUROC 0.92 and only 30% caught at 1 in 100; 60-99 words 0.96 and 50%; 100-149 words 0.97 and 82%; 150-249 words 0.98 and 90%; 250 words or more 0.94 and 48% (longer MAGE news texts, scored on their first 512 word pieces).
- **Honest reading:** on text like its training data it is almost never wrong. On newer AI text about other topics it ranks well (AUROC 0.94) but its chances are far too sure of themselves: almost half of the GPT-4 texts were put below the band. Reworded AI text was missed almost completely. The fit learned what ChatGPT answers in 2023 look like, not AI text in general.
- **Tests:** 329 passed, plus 3 slow tests with the real models.

## Text round 2 — fit on two sources, wider "can't tell" band

- **Built:**
  - The fit now also uses half of MAGE (split by text, so no text is in both halves). HC3 and MAGE count equally, so the bigger one does not set the curve alone.
  - Three more measures from the same model pass: the Fast-DetectGPT curvature, the performer's average surprise, and the text's length. They only stay because they helped on held-out texts (see below).
  - The "can't tell" band is now set from the data: its top is where at most 1 in 100 people's training texts reach, and never below 65%. Its bottom is where at most 1 in 100 AI training texts fall, and never above 35%. That gave 7-83%.
  - The overlapping predictability measures are reported as one reason, because their separate weights mean little when they move together.
  - The answer now also says how much AI text is caught, not only how often it is missed.
- **What we tried and dropped:** the round-1 three measures fitted on both sources caught 61% of GPT-4 texts at 1 in 100 and put 8.8% of TOEFL essays above a 35-65% band. The six measures caught 78% and put 3.3% above the band, so we kept six.
- **Numbers (texts of 100 words or more; MAGE is now only the held-out half):**

| Held-out set | Texts | AUROC | Caught at 1 in 100 | Calibration error | People above band | AI above band (caught) | AI below band |
|---|---|---|---|---|---|---|---|
| HC3, same topics | 914 | 0.998 | 99.5% | 0.018 | 0.3% | 99.0% | 0.2% |
| MAGE, GPT-4 | 721 | 0.964 | 77.5% | 0.041 | 0.8% | 73.4% | 2.2% |
| MAGE reworded by another AI | 1,856 | 0.665 | 3.5% | 0.635 | 0.8% | 2.0% | 44.5% |
| TOEFL essays (all 91, any length) | 91 | - | - | - | 3.3% | - | - |

- **Writing it has never seen** (refit with one kind of MAGE writing left out, tested on that kind):

| Left out | AUROC | People above band | AI above band |
|---|---|---|---|
| News (CNN) | 0.72 | **16%** | 57% |
| Dialogue summaries | 0.99 | 2.4% | 88% |
| Film reviews | 0.999 | 0% | 78% |
| Medical abstracts | 0.98 | 1.0% | 82% |

- **By length (held-out HC3 + MAGE):** 30-59 words AUROC 0.92, 3.2% of people's texts above the band; 60-99 words 0.99, 2.5%; 100-149 words 0.98, 0.7%; 150-249 words 0.996, 0%; 250+ words 0.98, 0.9%. We keep the minimum at 100 words: below it, people's texts were flagged three to four times as often. Between 100 and 150 words the answer carries a note that it is less accurate.
- **Honest reading:** calibration on GPT-4 text went from badly over-sure (error 0.228) to close (0.041), and missed AI text below the band fell from 44.5% to 2.2%. The price is a wider band: 32% of people's GPT-4-set texts and 60% of the TOEFL essays now land in "can't tell". That is the honest answer for them. The big failure is writing the fit has never seen: with no news in training, 16 in 100 people's news articles were put above the band. Reworded AI text is still missed almost entirely.

## Text round 3 — choosing the models, edge cases, fairness checks

- **Built:**
  - Compared three model pairs on the same 2,131 texts (every third text), with the round-2 recipe. SmolLM2-360M stayed: it was the most accurate and the fastest.

| Pair | HC3 AUROC | GPT-4 AUROC | GPT-4 caught at 1 in 100 | People above band (HC3 / GPT-4 set) | Speed |
|---|---|---|---|---|---|
| SmolLM2-360M base + instruct | 0.995 | 0.980 | 83% | 0.0% / 0.8% | about 0.3-0.8 s per text |
| GPT-2 + GPT-2 medium | 0.994 | 0.969 | 77% | 1.8% / 1.7% | similar |
| Qwen2.5-0.5B base + instruct | 0.990 | 0.960 | 71% | 1.8% / 0.0% | about 3 times slower |

  - A separate review found and fixed 22 edge cases, each with a test: short forms such as "Dr." and "U.S." no longer end a sentence; list items and headings are their own sentences; words in other alphabets and numbers such as "3.5" count as one word; the first word of each sentence now counts towards that sentence; the "only part was scored" note was shown when nothing was cut; very large or binary input is handled; every error gives JSON when asked.
  - The cheap measures are now recomputed with the current rules every time we measure. 50 texts whose tidy-up changed were scored again.
  - Sentences are listed only when they differ clearly from the whole text.
  - Added essays and abstracts by US school and college students from the same study as the TOEFL essays, to compare fairly.
  - The answer now always says how it did on writing it was never fitted on.
- **Numbers (texts of 100 words or more):**

| Held-out set | Texts | AUROC | Caught at 1 in 100 | Calibration error | People above band | People in band | AI above band (caught) | AI below band |
|---|---|---|---|---|---|---|---|---|
| HC3, same topics | 914 | 0.998 | 99.7% | 0.018 | 0.3% | 15% | 98.8% | 0.2% |
| MAGE, GPT-4 | 717 | 0.969 | 79.9% | 0.039 | 0.6% | 32% | 74.5% | 2.5% |
| MAGE reworded by another AI | 1,852 | 0.669 | 4.1% | 0.636 | 0.6% | 32% | 2.1% | 45.7% |
| TOEFL essays, non-native students (all 91) | 91 | - | - | - | 3.3% | 56% | - | - |
| US eighth-grade essays | 88 | - | - | - | 0% | 4.5% | - | - |
| US college essays | 70 | - | - | - | 0% | 5.7% | - | - |
| Stanford course abstracts | 145 | - | - | - | 0% | 5.5% | - | - |

- **Writing it has never seen:** news left out of fitting, then tested: AUROC 0.74, 14% of people's news put above the band. The other three kinds stayed between 0% and 1%.
- **Honest reading:**
  - **Bias against non-native writers is real here too.** The typical TOEFL essay got 9%; the typical US student essay got under 1%. 3 in 100 TOEFL essays were put above the band, against none of 303 texts by US students. The wide band catches most of the rest as "can't tell", but that is still unequal treatment. This is why the caveat is printed every time.
  - Reworded AI text is still missed. We did not try to fix this: the research says no detector can do so reliably.
  - Kinds of writing the fit has not seen, such as news, can be flagged wrongly far more often.
- **Tests:** 365 fast tests pass, plus 3 slow tests with the real models.

## Text round 4 (stopped early, 2026-11-01)

What landed:
- Scripts that download RAID samples and newer AI models' writing (download once; nothing gets committed).
- Scripts that gather more essays by non-native English writers.
- Measuring that splits results by AI model and by kind of writing.
- The README accuracy table is now written straight from the stored results, so the two cannot drift apart.

Not done yet:
- The check has not been refitted on the new data. All numbers shown are still round 3's.
- There are no leave-one-model-out or leave-one-kind-out results yet.
- Reworded AI text has not been measured on RAID's reworded set yet. It is still caught only 2 times in 100.
- Flag rates for non-native writers, with confidence ranges, have not been run on the larger set yet.
- A test that the "How often" list names AI models the check never saw was written, but it was held back because it fails until the refit is done.
