# markcheck

## What it does

markcheck looks inside your files for labels that say a file was made or changed with AI.
You point it at a file or a folder.
It tells you in plain English which labels it found.
It also tells you what each label shows, and what it does not show.

It runs on your own computer.
It never goes online, and nothing is uploaded.

It looks for four kinds of label:

- **Signed labels (C2PA, also called Content Credentials).** A note attached to the file that says which tool made it. The note is signed, so any later change to the file shows up. OpenAI, Adobe, Microsoft, Google, Amazon and Anthropic use these.
- **Metadata tags.** Metadata means notes stored inside a file about the file. markcheck reads the news-industry (IPTC) tag that says a picture was made by AI, and the "AIGC" field that China requires on AI-made files.
- **Stable Diffusion's hidden mark.** A hidden mark is a pattern woven into the pixels that people cannot see. Stable Diffusion tools can add an open one with a fixed message.
- **Adobe TrustMark.** An open hidden mark made by Adobe.

For each file it also lists the company checkers that only the maker can run, such as Google's SynthID Detector.

## Why it matters

New laws ask AI makers to label what their tools make.
The EU AI Act asks for this from August 2026.
California and China have similar rules.

But labels are easy to misread.
A missing label is often taken to mean a person made the file.
That is wrong: labels are often lost when files are shared.
A valid signature is often taken to mean the picture is true.
That is also wrong: it only shows who signed it and that the file has not changed since.

markcheck says plainly which of these it can and cannot tell you.

## What it can and can't tell you

| What markcheck finds | What it shows | What it does not show |
|---|---|---|
| A signed label (C2PA) that checks out | Who signed it, which tool or AI model the label names, and that the file is unchanged since signing | That the content is accurate. Who really made it, if the signer is not on the official C2PA trust list (the list of signers whose identity has been checked) |
| A signed label that says the file changed | The file once had a label, and it was edited after signing | What was changed, or whether the change matters. Even re-saving can do this |
| A metadata tag saying "made by AI" | The file says so about itself | Who wrote the tag. Tags are plain text that anyone can add or remove |
| Stable Diffusion's hidden mark | The picture passed through a Stable Diffusion tool that adds the mark | Which person or service made it. The mark is public, so any tool can add it |
| Adobe TrustMark | The picture carries a TrustMark and its short message | Who made it. The message is usually a pointer to a signed label kept elsewhere |
| No label at all | Nothing | That a person made it. **No label found does not prove a file was made by a person.** |
| Hidden marks only the maker can read (Google SynthID, OpenAI, Microsoft, Amazon, Anthropic's text mark) | markcheck cannot read these | It points you to each company's own checker instead |

markcheck never says a file is "human-made" or "authentic".
It says "no label found" or "signed by X; unchanged since signing".

## Do labels survive normal sharing?

We took every labelled test picture and did what people do every day: save as a JPEG, shrink it, or take a screenshot.
Then we checked again.
"Kept" means the program carried the file's notes along, as photo editors often do.
"Dropped" means only the pixels were saved, as most upload sites do.

| Label | Original | JPEG at quality 75, notes kept | JPEG at quality 75, notes dropped | Half size, notes kept | Half size, notes dropped | Screenshot-like copy |
|---|---|---|---|---|---|---|
| Signed label (C2PA) | 10/14 | 0/12 (12 say changed) | 0/14 | 0/12 (12 say changed) | 0/14 | 0/14 |
| Metadata tag | 8/8 | 5/8 | 0/8 | 8/8 | 0/8 | 0/8 |
| Stable Diffusion mark | 7/7 | 0/7 | 0/7 | 0/7 | 0/7 | 7/7 |
| Adobe TrustMark | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |

What this means in plain words:

- **Signed labels break on any change.** That is by design. When the notes are kept, the label stays but says the file changed. When the notes are dropped, the label is gone. Four "originals" are test files that were changed on purpose.
- **Metadata tags last only as long as the notes are kept.** A screenshot or an upload site usually loses them.
- **Stable Diffusion's mark is fragile.** A screenshot keeps it, because the pixels are unchanged. Re-saving as a JPEG or shrinking the picture loses it.
- **TrustMark survived every everyday change we tried.** It was built for this.

We also tested files made by real AI products, from public sources with open licences: an OpenAI picture, an Adobe Firefly picture, a Stable Diffusion XL picture and Adobe's TrustMark samples.
markcheck found the label in all of them.
The results after everyday changes matched the table above.
The sources and licences are in `tests/fixtures/realworld/SOURCES.md`.

On 52 files with no label, markcheck raised 0 false alarms.

The full numbers for each round of building are in `docs/rounds.md`.
Run `python scripts/measure.py` to measure them yourself.

## Install

You need Python 3.11, 3.12 or 3.13.

```
git clone https://github.com/patkusch/markcheck
cd markcheck
python -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
```

`[all]` adds the two hidden-mark checks.
They need the PyTorch library, which is a large download.
Without `[all]`, markcheck still checks signed labels and metadata tags.

The TrustMark check needs Adobe's model files.
markcheck never downloads them by itself.
Fetch them once with:

```
python -m markcheck.fetch_models
```

## Use

```
markcheck photo.jpg
markcheck my-folder/
markcheck photo.jpg --json
```

The answer it gives back to scripts (the exit code):

- `0`: no label found. This does not prove a file was made by a person.
- `1`: a label was found in at least one file.
- `2`: something could not be checked.

Hidden files, and tool folders such as `.git`, are skipped.

## What markcheck will not do

markcheck only finds and explains labels.
It does not remove, change or add labels.

## Limits

- It reads only the TrustMark variant most people use (Q). Other variants are reported as not checked.
- China's AIGC field layout follows secondary sources. We could not read the official standard.
- It cannot read text marks such as SynthID Text, OpenAI's or Anthropic's. No one outside those companies can yet.
- The C2PA trust list is bundled as it stood on the date in `src/markcheck/data/README.md`. It is not updated automatically.

## Licence

MIT. Test files from other sources keep their own licences, listed next to them.
