# What AI labels exist, and which ones anyone can check

Research snapshot, 2026-10-31. Four researchers each covered one area. A lead then checked the most surprising claims against their sources. Anything we could not confirm is marked **unverified**.

Two kinds of label come up throughout:

- **Signed label (C2PA, "Content Credentials").** A note attached to the file saying who made it, signed so that any later change shows. Anyone can read it. Saving a screenshot or re-uploading the file usually throws it away.
- **Hidden mark (watermark).** A pattern woven into the pixels, sound or word choices. It survives more handling. Usually only the company that added it can check for it.

## Who labels what

| Lab | What gets labelled | Kind of label | Who can check it |
|---|---|---|---|
| Google (SynthID) | text, images, audio, video | hidden mark | Google only (SynthID Detector portal, Gemini app) |
| Google (Pixel 10, Photos, Vertex) | photos, edits, images | signed label | anyone |
| OpenAI images | images | signed label + SynthID | label: anyone. Mark: openai.com/verify |
| OpenAI text ("textGrain", announced 2026-10-05) | text | hidden mark; required in the EU for ChatGPT and Codex | approved researchers only |
| Anthropic Claude | text | hidden mark, based on SynthID-Text | detection service in private preview |
| Anthropic Claude | files Claude makes (.png, .jpg, .svg) | signed label | anyone; claude.com/check-content |
| Meta | images | visible "Imagined with AI" + metadata tag + hidden mark | tag: anyone. Mark: no public checker |
| Microsoft Foundry | images, audio, video | signed label + hidden mark | label: anyone. Mark: Microsoft's free site |
| Adobe Firefly | images | signed label (TrustMark mark is available) | anyone |
| Amazon Titan / Nova | images, video | signed label + hidden mark | label: anyone. Mark: Bedrock service |
| Stability AI | images | open hidden mark with a fixed message | anyone, but only on untouched files |
| ElevenLabs | audio | SynthID + their own classifier | ElevenLabs only |
| Midjourney | images | metadata tag (pledged 2024) | anyone. 2026 status **unverified** |
| Mistral | — | no source found | **unverified** |

Several companies now share Google's SynthID. OpenAI, ElevenLabs, NVIDIA and Kakao use it, and Anthropic's text mark builds on it. So a SynthID hit does not say which company made something.

## Laws that make labelling compulsory

- **EU AI Act, Article 50.** AI makers must mark synthetic content so machines can detect it. This applies from 2 August 2026. Systems already on sale before that date get until 2 December 2026 (secondary sources). The final Code of Practice of 10 June 2026 expects both a signed label and a hidden mark.
- **California SB 942 / AB 853.** Large AI makers must offer a free public checker. This applies from 2 August 2026, for platforms from 2027 and for cameras from 2028.
- **China, standard GB 45438-2025.** In force since 1 September 2025. Requires a visible label plus a hidden "AIGC" field in the file's metadata. We did not read the official text, so the exact field layout is **unverified**.
- **India and South Korea.** Both have labelling rules in force in 2026 (secondary sources).

## What research says about reliability

- **Checkers make mistakes in both directions.** OpenAI withdrew its 2023 text detector. It caught only 26% of AI text and wrongly flagged 9% of human text. NIST: detection "always comes with some probability of error".
- **Text marks weaken when the text is rewritten or translated.** Google says this about SynthID Text. Kirchenbauer et al. found a mark survives a strong human rewrite only once the text is about 800 tokens long (roughly 600 words).
- **Hidden image marks are fragile.** Zhao et al. 2023, Saberi et al. 2023 and the WAVES benchmark (2024) all show this. Marks built from public model parts are the easiest to lose.
- **Labels get lost in normal sharing.** NIST says social media often strips metadata. In a Washington Post test, only one of eight platforms told users that an AI video was AI.
- **Marks can be faked to frame someone.** Jovanović et al. 2024 forged text marks for under $50.
- **A missing label proves nothing.** A valid label proves who signed the file, not that the content is true (C2PA explainer, section 7).

## What a free checker can read on your own computer today

1. **Signed labels (C2PA).** Used by OpenAI, Anthropic files, Microsoft, Adobe, Google and Amazon. Library: `c2pa-python`.
2. **Metadata tags.** The IPTC "made by AI" tag and China's AIGC field. Readable from the file's XMP text.
3. **Stable Diffusion's open hidden mark.** Library: `invisible-watermark`. Works only on untouched files.
4. **Adobe TrustMark.** Library: `trustmark`.
5. **Signed labels hidden inside plain text** (invisible characters). The specification is still marked unstable.
6. **Meta's research marks (AudioSeal, Video Seal).** These are public models. Whether Meta's products use them is **unverified**.

These can only be checked through the company's own website or service:

- Google SynthID for images, audio and video
- OpenAI's image mark
- OpenAI's and Anthropic's text marks
- Microsoft's mark
- Amazon's mark
- ElevenLabs

## Gaps

- There is no open checker for any big lab's text mark.
- The hidden marks that survive sharing can only be checked by the company that made them.
- There is no neutral checker that covers several labs.
- Licence traps: Stable Signature is non-commercial only, and WAVES has no licence file.
- Honest wording rule:
  - "No label found" means "not proven to be AI". It never means "made by a human".
  - "Label valid" means "signed by X and unchanged since". It never means "true".

## Sources

- anthropic.com/news/claude-text-watermark
- support.claude.com/en/articles/16266773
- techcrunch.com/2026/10/05/openai-will-start-watermarking-chatgpts-text-in-the-eu/
- techcrunch.com/2026/05/19/openai-is-making-it-easier-to-check-if-an-image-was-made-by-their-models/
- deepmind.google/models/synthid
- ai.google.dev/responsible/docs/safeguards/synthid
- learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/provenance-disclosure
- docs.aws.amazon.com/ai/responsible-ai/nova-canvas/overview.html
- about.fb.com/news/2024/02/labeling-ai-generated-images-on-facebook-instagram-and-threads/
- artificialintelligenceact.eu/article/50/
- calmatters.digitaldemocracy.org/bills/ca_202520260ab853
- doi.org/10.6028/NIST.AI.100-4
- arxiv.org/abs/2306.04634
- arxiv.org/abs/2303.11156
- arxiv.org/abs/2306.01953
- arxiv.org/abs/2310.00076
- arxiv.org/abs/2401.08573
- arxiv.org/abs/2402.19361
- spec.c2pa.org (2.2 explainer)
- github.com/contentauth/c2pa-python
- github.com/ShieldMnt/invisible-watermark
- github.com/adobe/trustmark
- github.com/facebookresearch/audioseal
- github.com/facebookresearch/videoseal
