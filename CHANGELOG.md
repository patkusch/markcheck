# Changes

## Unreleased

- Shows words hidden in invisible tag characters, and leaves flag emoji alone.
- `markcheck text --labels-only` checks for hidden labels and characters without loading any models.
- Reads text files saved as UTF-16 or UTF-32, as well as UTF-8.
- Finds every hidden label in a text, says how many there are, and leaves them all out of the words that get scored.
- A GitHub job, started by hand, compares a bigger model pair with the shipped one, so no laptop has to do the work.
- Finds signed labels (C2PA) hidden as invisible characters after the words of a text, in `markcheck text` and in text files. It reads who signed them and whether the signature holds, but not yet whether the words changed since.
- Counts other invisible characters in a text, such as zero-width spaces, and says they prove nothing on their own.
- `markcheck text` estimates the chance a text was AI-written, with reasons, and says how often it is wrong.
- The ship rule for a new text fit now also checks that it keeps catching AI text and does not just say "can't tell" more often.
- Every measuring run is kept in `docs/results/`.
- The text scorer can use half precision, so bigger model pairs fit in a laptop's memory.

## 0.1.0

- Reads signed labels (C2PA), "made by AI" metadata tags, Stable Diffusion's hidden mark and Adobe TrustMark.
- Points to each company's own checker for marks only they can read.
