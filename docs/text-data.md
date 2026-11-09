# Data for the text check

None of this data is in the repository.
Each script downloads it once into `~/.cache/markcheck/textdata/`.
`scripts/text_data.py` then joins it into one file, `splits.jsonl`.
Run the download scripts with `python -I`.

Counts are the texts in `splits.jsonl` after short texts (under 30 words) were dropped.
The check only gives a number for texts of 100 words or more, so that count is shown too.
Checked on 2026-11-01.

## Where each set comes from

| Set | What it is | Licence | Downloaded | Texts by people (100+ words) | AI texts (100+ words) |
|---|---|---|---|---|---|
| HC3 | Questions answered by people and by ChatGPT | CC BY-SA 4.0 | 73 MB | 1,200 (664) | 1,200 (1,158) |
| MAGE, other-topics test file | News, reviews, dialogue summaries, medical abstracts; GPT-4 | Apache 2.0 | 6 MB | 761 (735) | 779 (692) |
| MAGE, reworded copy | The same AI texts, reworded by another AI | Apache 2.0 | (same) | the same 761 | 1,599 (1,499) |
| RAID sample | 8 kinds of writing, 11 AI models from 2019-2023 | MIT | 140 MB read from an 11.8 GB file | 418 (418) | 4,686 (4,302) |
| RAID, reworded sample | RAID texts reworded by another AI | MIT | (same) | 197 (193) | 1,304 (1,072) |
| DetectRL-X, English part | People's texts and texts by GPT-4o, Gemini 2.5 Flash, DeepSeek-V3 and Qwen-Max | MIT | 80 MB read from an 879 MB file | 720 (720) | 720 (720) |
| Beemo | Answers by ten open chat models, and by people to the same prompts | MIT | 8 MB | 264 (138) | 300 (201) |
| PERSUADE 2.0 | US school essays: 400 by English learners, 400 by matched non-learners | CC BY 4.0 | 617 MB | 800 (800) | - |
| ELLIPSE | Essays by English learners | CC BY 4.0 | 10 MB | 300 (298) | - |
| TOEFL essays (Liang et al. 2023) | Essays by non-native English students | No licence file | under 1 MB | 91 (63) | - |
| US student essays (same study) | US eighth-grade and college essays, Stanford course abstracts | No licence file | under 1 MB | 303 (290) | - |

Licences were read from each dataset's own page or README on 2026-11-01.
The Liang et al. sets have no licence file, so we only use them for measuring, on our own machine.

## How each set is used

- HC3, MAGE and RAID can be used for fitting. Each is split in half by source document, so a person's text and the AI texts made from it always land on the same side.
- DetectRL-X and Beemo are never used for fitting. They test AI models the check never saw.
- The reworded sets are kept aside for testing, unless a round says otherwise.
- PERSUADE, ELLIPSE, TOEFL and the US student sets hold only people's writing. They measure how often the check wrongly flags people, by group.

## How to rebuild

```
python -I scripts/text_fetch_raid.py
python -I scripts/text_fetch_newmodels.py
python -I scripts/text_fetch_fairness.py
python scripts/text_data.py
python scripts/text_features.py HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-360M-Instruct
```

The last step scores every text with the two small models, about half a second per text on a laptop.
The scores are kept in the same folder, outside git.

## How to measure and ship a new fit

```
python scripts/measure_text.py HuggingFaceTB/SmolLM2-360M-Instruct --train-on hc3+mage+raid \
    --compare-with old_calibration.json --json-out results.json
python scripts/measure_text.py HuggingFaceTB/SmolLM2-360M-Instruct --train-on hc3+mage+raid --write
python scripts/text_fairness.py --merge
python -m markcheck.textcheck.accuracy --write-readme
```

- The first step fits on the train side and measures on the texts kept aside. It also refits once per AI model and once per kind of writing, each time leaving that one out.
- `--compare-with` measures the old fit on the same texts kept aside and applies the ship rule. A new fit ships only if at most 1 in 100 people's texts from HC3, MAGE and RAID are flagged, and no kind of writing has more people's texts flagged than before. It must also catch nearly as much AI text as before (no more than 2 in 100 fewer), and must not put many more people's texts in the "can't tell" band (no more than 5 in 100 more). Without those two checks, a fit could pass just by refusing to answer.
- `--write` saves the new fit, and the numbers the command line shows, to `src/markcheck/data/text_calibration.json`.
- `--merge` adds the fairness numbers, worked out with that same fit.
- The last step copies the numbers into the README, so the two cannot drift apart.
- Use `--baseline` instead of a refit to measure the shipped fit as it is.

## Where past results are kept

Each measuring run worth keeping is saved in `docs/results/`, as a JSON file for scripts and a text file for people. They hold numbers only, never the texts. Save new runs there with `--json-out` and say in `docs/rounds.md` whether the fit shipped.

## Comparing a bigger model pair

`scripts/text_compare_pairs.py` scores a fixed sample of held-out texts with a second model pair. It then compares how well each pair tells AI text from people's.

```
python scripts/text_compare_pairs.py ~/.cache/markcheck/models/qwen1.5b ~/.cache/markcheck/models/qwen1.5b-instruct --name qwen1.5b --limit 300
python scripts/text_compare_pairs.py ~/.cache/markcheck/models/qwen1.5b ~/.cache/markcheck/models/qwen1.5b-instruct --name qwen1.5b --limit 300 --report
```

- To download a bigger pair that can resume if the line is slow, run `bash scripts/text_fetch_pair.sh Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct`. It saves each model under `~/.cache/markcheck/models/`.
- The observer and performer can be Hugging Face names or local folders. Local folders help when a download needs to resume.
- `--name` keeps the cached scores file name short.
- `--half` halves the memory a pair needs. It gives nearly the same scores but runs much slower on a laptop processor.
