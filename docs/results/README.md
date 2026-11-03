# Stored results

Each file here is one measuring run of the text check. Every number comes from texts that were kept aside and never used for fitting.

- `round7-refit-hc3-mage-raid.json` holds every number from the round 7 refit, for scripts to read.
- `round7-refit-hc3-mage-raid.txt` is the same run, printed as a person would read it.

These files hold numbers only. They never hold the texts themselves, because the data sets have their own licences. The texts are downloaded once into `~/.cache/markcheck/textdata` (see `docs/text-data.md`).

The fit that ships is in `src/markcheck/data/text_calibration.json`. A result stored here does not mean it shipped. `docs/rounds.md` says which ones did.
- `round7b-refit-weighted-by-model.*` is the second round 7 try. Each AI model gets equal weight in fitting. It skips the leave-one-out refits.
- `round8-refit-more-people-news.*` is the round 8 refit, with 108 more news articles by people added from RAID. The JSON comes from a quick rerun without the leave-one-out refits; the text file has both.
- `round8-shipped-fit-baseline.*` measures the fit that ships today on the same held-out texts as round 8, so the two can be compared fairly.
- `round8b-refit-strict-band.*` is the round 8 refit with a stricter top of the "can't tell" band (96% instead of 87%).
