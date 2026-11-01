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

`[all]` adds the two hidden-mark checks and the text check.
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

## Text check: an estimate, not a finding

`markcheck text` looks at a piece of writing and estimates the chance that it was AI-written.
It is a separate check from the label checks above.
It does not look for a label.
It asks two small open language models how predictable the words are.
A language model is a program that guesses the next word; AI text tends to use the words such a program expects.

```
markcheck text "paste the text here"
markcheck text --file essay.txt
cat essay.txt | markcheck text -
markcheck text --file essay.txt --json
```

It gives:

- an estimated chance, such as "Estimated chance this was AI-written: 72% (uncertain band 7%-84% means we can't tell)";
- the reasons, worked out from this text's own numbers, such as how predictable the words were and how even the sentence lengths were;
- the sentences that moved the estimate most;
- how often the check was right and wrong on texts it had never seen;
- always, one line: this is an estimate, it is wrong sometimes, it can be unfair to people writing in a language that is not their first, and it should not be the only basis for any decision about a person.

It needs at least 100 words.
Below that, our tests showed people's texts were wrongly flagged three to four times as often, so it gives no number and says why.

### How accurate it is

<!-- text-accuracy:start -->
Measured on texts kept aside, never used for fitting. The "can't tell" band is 7%-84%. Texts under 100 words get no number. Full numbers are in `docs/rounds.md`.

| What we measured | How often |
|---|---|
| people's texts put above the band (similar topics) | fewer than 1 in 100 |
| AI texts caught above the band (similar topics) | 99 in 100 |
| AI texts put below the band (similar topics) | fewer than 1 in 100 |
| people's texts put above the band (news, reviews, summaries, abstracts; GPT-4) | 1 in 100 |
| AI texts caught above the band (news, reviews, summaries, abstracts; GPT-4) | 74 in 100 |
| AI texts put below the band (news, reviews, summaries, abstracts; GPT-4) | 2 in 100 |
| AI texts reworded by another AI, caught above the band | 2 in 100 |
| essays by non-native English students put above the band | 3 in 100 |
| essays by non-native English students put inside the band | 57 in 100 |
| essays by US school and college students put inside or above the band | 5 in 100 |
| people's news put above the band when that kind of writing was left out of fitting (the worst case we tried) | 14 in 100 |
<!-- text-accuracy:end -->

What this means in plain words:

- **It is good at text like its examples, and much worse at anything else.**
- **Reworded AI text gets past it almost every time.**
- **It is unfair to non-native English writers.** Their essays got higher estimates than US students' essays, as other detectors do.
- **The percentage depends on the mix.** It was fitted on half AI and half people's text. Where AI text is rarer, the chance is lower than shown.

### What it must not be used for

- It must not be the only basis for any decision about a person: a grade, a job, an accusation or a penalty.
- It cannot show who wrote something.
- It is not a check for copying, and it does not say whether anyone broke a rule.
- markcheck gives no tips for changing a text so it scores differently, and never will.

The research behind these choices is in `docs/text-research.md`.

### Install the text check

The text check needs extra libraries and two small open models (SmolLM2-360M and SmolLM2-360M-Instruct, Apache-2.0, about 1.4 GB together):

```
pip install -e '.[text]'
python -m markcheck.fetch_models --text
```

After that it runs on your computer and never goes online.
On a laptop without a graphics card, one text takes about 3 seconds, most of it loading the models.

## What markcheck will not do

markcheck only finds and explains labels, and estimates the chance a text was AI-written.
It does not remove, change or add labels.
It does not help anyone change a text to get a different estimate.

## Limits

- It reads only the TrustMark variant most people use (Q). Other variants are reported as not checked.
- China's AIGC field layout follows secondary sources. We could not read the official standard.
- It cannot read text marks such as SynthID Text, OpenAI's or Anthropic's. No one outside those companies can yet.
- The C2PA trust list is bundled as it stood on the date in `src/markcheck/data/README.md`. It is not updated automatically.

## Licence

MIT. Test files from other sources keep their own licences, listed next to them.
