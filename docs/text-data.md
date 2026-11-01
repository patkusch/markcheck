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
