# Research: checking whether a text was written by AI

This note collects what we learned before building `markcheck text`.
It was written on 2026-11-01.
Every claim has a link.
**[checked]** means we read the claim on the original page or in the paper's summary.
**[second-hand]** means we saw it only in news, a review or a search summary.
**UNVERIFIED** means we could not confirm it at all.

## The short answer

- No one can tell for sure whether a text was written by AI.
- The best open methods need no training data. They ask a small language model how surprised it is by each word.
- A language model is a program that guesses the next word. AI text tends to use the words such a program expects.
- We chose a method called Binoculars, run with two small open models on the computer's main processor (CPU).
- We turn its score into a chance (a percentage) using public examples of AI and human text, and we check that chance on examples we kept aside.
- We refuse to give a number for short texts, we show a wide "can't tell" band, and we always warn that the estimate can be wrong and unfair.

## Methods we looked at

| Method | How it works, in plain words | Licence | Runs on a laptop CPU? | Reported results |
|---|---|---|---|---|
| [Binoculars](https://github.com/ahans30/Binoculars) ([paper](https://arxiv.org/abs/2401.12070)) | Compares how surprised one model is by the text with how surprised it is by a second model's guesses. Dividing the two cancels part of the "this topic is just easy to predict" effect. | BSD-3 code [checked] | The paper's pair is two 7-billion-word-piece models, about 29 GB. Too big. Small pairs are untested in print. | Catches 93-99% of AI text while wrongly flagging 1 in 10,000 human texts, on the paper's own test sets [checked]. On the RAID benchmark it catches 79.6% while wrongly flagging 5% [checked, [RAID paper](https://arxiv.org/abs/2405.07940)]. |
| [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) ([paper](https://arxiv.org/abs/2310.05130)) | Checks whether the text is more likely than the text the model would write itself. One pass of one model. | MIT [checked] | Yes with a small model. | AUROC 0.93 on ChatGPT and GPT-4 text with a 2.7-billion model [checked]. |
| [DetectGPT](https://github.com/eric-mitchell/detect-gpt) | Same idea, but rewrites the text about 100 times to estimate it. | UNVERIFIED | No: hundreds of times slower. | AUROC 0.72 on ChatGPT text (from the Fast-DetectGPT page) [second-hand]. |
| [GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) | Colours each word by how high it was in GPT-2's list of guesses, for a person to look at. | Apache-2.0 [checked] | Yes. | Helped people spot AI text 72% of the time instead of 54% ([paper](https://arxiv.org/abs/1906.04043)) [checked]. |
| [Ghostbuster](https://github.com/vivek3141/ghostbuster) | Trains a simple classifier on word-probability features. | UNVERIFIED | No: its standard set-up sends every text to OpenAI's online service. | 99.0 F1 on its own data [second-hand]. |
| [RADAR](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B) | A trained detector that practised against a rewording model. | Non-commercial weights [checked] | Possibly. | 70.9% caught at 5% wrongly flagged on RAID [checked]. |
| Trained classifiers on Hugging Face: [OpenAI's RoBERTa detector](https://huggingface.co/openai-community/roberta-base-openai-detector), [Hello-SimpleAI](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta), [desklib](https://huggingface.co/desklib/ai-text-detector-v1.01), [fakespot](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1), [Pangram EditLens](https://huggingface.co/pangram/editlens_roberta-large) | A model trained on labelled examples. | Mixed; Pangram's is non-commercial and forbids use to enforce AI policies [checked] | Yes, 125-400 million parameters. | OpenAI's own card warns it does not work on ChatGPT text [checked]. Pangram's open model wrongly flagged 9.9% of 91 human texts [checked]. |
| 2025 methods: [Lastde](https://arxiv.org/abs/2410.06072), [AdaDetectGPT](https://arxiv.org/abs/2510.01268), [DNA-DetectLLM](https://arxiv.org/abs/2509.15550) | Refinements of the "how surprised is the model" idea. | Open code | Lastde yes; others slower. | Claim gains over Fast-DetectGPT [checked: abstracts only]. |

Trained classifiers do well on text like their training data.
They do badly on text from AI models they never saw ([RAID](https://arxiv.org/abs/2405.07940)) [checked].
Methods that need no training go out of date more slowly.
Smaller models can be better at this than big ones: a 125-million-parameter model beat a 6-billion one on ChatGPT text ([Mireshghallah et al.](https://arxiv.org/abs/2305.09859)) [checked].

## Benchmarks: what "good" looks like

- **[RAID](https://github.com/liamdugan/raid)** has more than 10 million texts from 11 AI models in 8 kinds of writing, with rewording attacks. Licence MIT [checked]. Its leaderboard ranks detectors by how much AI text they catch while wrongly flagging 5% of human texts. The training file alone is 11.8 GB, so we did not use it.
- **[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)** has about 24,000 questions, each answered by people and by ChatGPT (early 2023). Licence CC-BY-SA-4.0 [checked]. Small files per topic. **We use it to fit and check our numbers.** We download it once; we do not copy it into this project.
- **[MAGE](https://huggingface.co/datasets/yaful/MAGE)** has 447,000 texts from 27 AI models. Its small "out of distribution" test files hold GPT-4 text on topics HC3 does not cover, plus reworded versions. Licence Apache-2.0 on Hugging Face, CC BY 4.0 in the README; both are open [checked]. **We use them as a harder test we never fit on.**
- **[M4](https://github.com/mbzuai-nlp/M4)** has no licence on GitHub, so we did not use it [checked].
- **[DetectRL](https://github.com/NLP2CT/DetectRL)** had no published licence or data when we looked [checked].

## Writers whose first language is not English

- [Liang et al. (2023)](https://arxiv.org/abs/2304.02819) gave 91 essays written by non-native students for the TOEFL English test to seven detectors [checked].
- On average the detectors wrongly flagged 61% of them as AI-written.
- At least one detector flagged 89 of the 91.
- Essays by US eighth-graders were almost never flagged.
- The reason: simpler, more common words are easier for a model to predict.
- Their essays are public on [GitHub](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias). The README shows an MIT badge, but the repository has no licence file [checked]. **We use them only to measure, on your own computer. We do not copy them into this project.**

## Commercial tools and their claims

- **Turnitin** shows no number below 20%, because wrong flags are more common there. It needs at least 300 words. It claims under 1% wrong flags for documents scored above 20% ([University of Melbourne guide](https://lms.unimelb.edu.au/staff/guides/canvas/assignments/turnitin-accessing-and-interpreting-the-ai-writing-report)) [second-hand].
- **Pangram** reports wrongly flagging 70 of 781,981 web pages, and needs at least 50 words ([model card](https://www.pangram.com/research/model-card/pangram-4-1)) [checked]. An independent study found it close to zero on both kinds of error ([NBER w34223](https://ideas.repec.org/p/nbr/nberwo/34223.html)) [checked: summary only].
- **GPTZero** claims 0.24% wrong flags on its own test, which it ran itself ([GPTZero](https://gptzero.me/news/gptzero-vs-copyleaks-vs-originality/)) [checked, not independent].
- **Copyleaks** advertises 0.2% wrong flags ([Copyleaks](https://copyleaks.com/features/ai-content-detector)) [checked, not independent].
- **Originality.ai**: we found no claim on its own pages. UNVERIFIED.
- **OpenAI** withdrew its own detector on 20 July 2023 for its "low rate of accuracy". It caught 26% of AI text and wrongly flagged 9% of human text ([The Register](https://www.theregister.com/2023/07/26/openai_ai_classifier/)) [second-hand].

## Short texts

- OpenAI said its detector was "very unreliable" under 1,000 characters, about 150-200 words ([Decrypt](https://decrypt.co/149826/openai-quietly-shutters-its-ai-text-classifier-due-to-low-accuracy)) [second-hand].
- Turnitin needs 300 words. Pangram needs 50 and is less sure under 150.
- The Binoculars demo suggests 200-300 words ([demo notes](https://huggingface.co/spaces/tomg-group-umd/Binoculars/raw/refs%2Fpr%2F2/README-doc.md)) [checked].
- **Our plan:** refuse below a minimum length and say why. We pick the minimum from our own measurements by length (see `docs/rounds.md`).

## Why AI-text detection is unreliable

- **Rewording beats detectors.** Light, repeated rewording defeats every kind of detector ([Sadasivan et al.](https://arxiv.org/abs/2303.11156)) [checked]. One rewording tool cut DetectGPT from catching 70.3% of AI text to 4.6% ([Krishna et al.](https://arxiv.org/abs/2303.13408)) [checked].
- **New AI models and new kinds of writing fool detectors** ([RAID](https://arxiv.org/abs/2405.07940)) [checked].
- **Mixed text has no single right answer.** A person may edit AI text, or the reverse.
- **Plain, predictable writing looks like AI.** The US Constitution was rated 92% AI by one tool ([AI Incident Database](https://incidentdatabase.ai/reports/3228)) [second-hand]. Legal text, lists and code are likely to suffer the same way; that is our inference.
- **The percentage depends on the mix.** A "72%" means "72% of texts like this, in our test mix, were AI-written". Our test mix is half AI and half human. If AI text is rarer where you are, the true chance is lower ([Saerens et al.](https://mlanthology.org/neco/2002/saerens2002neco-adjusting)) [checked: record only].

## Harm from false accusations

- **Texas A&M-Commerce, 2023.** An instructor asked ChatGPT whether it wrote his students' papers, and diplomas were held up ([ADN](https://www.adn.com/nation-world/2023/05/18/a-professor-falsely-accused-his-class-of-using-chatgpt-their-diplomas-are-in-jeopardy/)) [second-hand].
- **Vanderbilt University turned off Turnitin's AI detector in August 2023.** Even 1 wrong flag in 100 would mean about 750 of its students wrongly flagged each year. It also cited bias against non-native writers ([Vanderbilt](https://www.vanderbilt.edu/brightspace/2023/08/16/guidance-on-ai-detection-and-why-were-disabling-turnitins-ai-detector/)) [checked].
- **Jisc (UK)** said AI detectors "cannot prove conclusively" that text was written by AI ([quoted by Aberystwyth](https://wordpress.aber.ac.uk/e-learning/2023/03/27/launch-of-turnitin-ai-writing-and-chatgpt-detection-capability/)) [second-hand].
- **The MLA-CCCC task force** advises using detectors with caution or not at all ([working papers](https://mlaai.hcommons.org/working-papers/)) [second-hand].
- **Stanford HAI** advises against relying on detectors in education ([HAI](https://hai.stanford.edu/news/ai-detectors-biased-against-non-native-english-writers)) [checked].

## How a responsible tool should word its answer

No published standard exists; we found none.
These rules follow from the sources above:

- Give a chance, never a verdict: "estimated chance this was AI-written: 72%".
- Show a middle "can't tell" band and say plainly that inside it we cannot tell.
- Say how often the tool is wrong, measured on held-out examples.
- Always say: this is an estimate, it is wrong sometimes, it can be unfair to non-native writers, and it should not be the only basis for any decision about a person.
- Never use the words "human-written", "proof", "plagiarism" or "cheating".
- Give reasons that come from the numbers for this text, not fixed text.
- Never give tips for changing a text so it scores differently. markcheck is a checker only.

## Calibration, in plain words

- A chance is **calibrated** when it comes true as often as it says. Of all texts given about 80%, about 80% should really be AI-written.
- **Calibration error** is the average gap between the stated chance and how often it came true, across bands of chances ([Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)) [checked].
- With a few thousand examples, a simple S-shaped curve (logistic, or "Platt", calibration) is safer than a step-shaped one (isotonic), which can over-fit small data ([scikit-learn guide](https://scikit-learn.org/stable/modules/calibration.html)) [checked].
- A chance calibrated on one kind of text does not carry over to another. We say which data we used.

## What we will build

- **Score:** Binoculars with a small base model and its chat-tuned twin from the same family, so they share a vocabulary. We start with SmolLM2-360M (Apache-2.0, about 1.4 GB for the pair), and compare it with GPT-2 and Qwen2.5-0.5B in round 1.
- **Extra signals from the same pass:** the Fast-DetectGPT curvature, how often words were in the model's top 10 guesses, and how even the sentence lengths are. Each only stays in if it helps on held-out data.
- **Chance:** logistic calibration fitted on half of HC3 (split by question), checked on the other half and on MAGE's GPT-4 files.
- **Bands:** a "can't tell" band, and a "likely AI" level set so that no more than about 1 in 100 held-out human texts reaches it.
- **Length:** refuse below a minimum chosen from our length results.
- **Packaging:** the models live in an optional extra, `pip install 'markcheck[text]'`, and are fetched once with `python -m markcheck.fetch_models --text`. After that, nothing goes online. Tests in CI use made-up scores, so they stay fast and offline.

## Round 4 data sources

In round 4 we added more AI models, more kinds of writing, and more writing by people who are learning English.
This section says where each set comes from, what we may do with it, and how much we used.
It was written on 2026-11-01.

**Nothing here is committed to this project.**
Every file is downloaded to `~/.cache/markcheck/textdata/` on your own computer and stays there.
We use the texts only to measure and fit the check.

### Sources we used

| Source | Licence | AI models | Kinds of writing | Texts we used | Size on disk |
|---|---|---|---|---|---|
| [RAID](https://huggingface.co/datasets/liamdugan/raid) ([code](https://github.com/liamdugan/raid)) | MIT, on the dataset card [checked] | GPT-2, GPT-3, ChatGPT, GPT-4, Cohere, Cohere chat, Llama 2 chat, Mistral, Mistral chat, MPT, MPT chat (2019-2023) | abstracts, books, news, poetry, recipes, Reddit posts, reviews, Wikipedia | RAID_UNTOUCHED untouched, RAID_PARA reworded | RAID_DISK |
| [DetectRL-X](https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X) ([paper](https://aclanthology.org/2026.acl-long.1773/)) | MIT, on the dataset card [checked] | GPT-4o, Gemini 2.5 Flash, DeepSeek-V3, Qwen-Max (2024-2025) | academic writing, news, novels, web marketing (SEO), web text, Wikipedia | DRX_AI AI and DRX_HUM by people | 81 MB (English part only) |
| [Beemo](https://huggingface.co/datasets/toloka/beemo) ([code](https://github.com/Toloka/beemo)) | MIT, on the dataset card and GitHub [checked]. The people's answers come from No Robots, which is CC BY-NC 4.0 (non-commercial). | Llama 2 (7B, 13B, 70B) chat, Gemma 2B and 7B, Mistral 7B, Mixtral 8x7B, Zephyr 7B, Tulu 2 (7B, 13B) (2023-2024) | answering questions, writing, rewriting, summarising | BEEMO_AI AI and BEEMO_HUM by people | 8 MB |
| [PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0) | CC BY 4.0, in the README [checked] | none: all by people | essays by US students in grades 6-12 | 400 by English learners, 400 by other students | 617 MB |
| [ELLIPSE](https://github.com/scrosseye/ELLIPSE-Corpus) | CC BY 4.0, in the README [checked] | none: all by people | essays by English learners in grades 8-12 | 300 | 10 MB |

What each one gives us:

- **RAID** adds many older AI models and eight kinds of writing.
  It also has a "paraphrase" set: AI texts reworded by another AI to dodge detectors.
- **DetectRL-X** adds four of the newest models we could find with an open licence.
  Each AI text was written from a person's text, and we keep both.
- **Beemo** adds ten open chat models answering everyday requests that people also answered.
- **PERSUADE 2.0** says, for every essay, whether the student is an English language learner (ELL).
  That lets us compare learners with other students who wrote on the same prompt in the same grade.
- **ELLIPSE** adds more essays by English learners, from the same research group.

### How we sampled them

- **RAID** is one 11.8 GB file, so we did not download it all.
  We read 24 KB every 2 MB to map where each model and attack sits, then read small pieces only where we needed them, about RAID_DOWNLOADED in all.
  We used only the labelled training file; the test file hides its answers.
  We kept about RAID_PER_CELL texts per kind of writing and model, and about 190 reworded texts per kind of writing.
  We left out texts made with a "repetition penalty", a setting RAID added to make detection harder; the rest mix the two ways RAID let models pick words (always the likeliest word, or a random likely word).
  People's abstracts in RAID keep the page-width line breaks from arXiv, and its AI abstracts do not. We join those lines so the check cannot learn the line breaks instead of the writing.
- **DetectRL-X** is sorted by language, so we read only the two stretches of its "general" file that hold English (about 95 MB of an 879 MB file).
  We took 30 AI texts for each model and kind of writing, plus the person's text each was written from.
- **Beemo**: 30 answers per model, plus the people's answers to the same requests.
- **PERSUADE 2.0**: 400 essays by learners, chosen at random, each matched with an essay by a non-learner on the same prompt in the same grade.
  "Non-learner" means the school did not mark the student as an English learner. It does not prove English is their first language.
- **ELLIPSE**: 300 essays at random. We skipped 37 that are also in PERSUADE.
- Every set keeps texts of at least 30 words, as before.

### Sources we could not use

| Source | Why not |
|---|---|
| [ICNALE](https://language.sakura.ne.jp/icnale/download.html) (essays by Asian learners of English) | You must fill in a registration form to get the password for the files. We do not sign up for things on your behalf. |
| [Feedback Prize: English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) | Needs a Kaggle account and agreeing to the competition rules. Its essays come from ELLIPSE, which we used instead. |
| [M4](https://github.com/mbzuai-nlp/M4) and [M4GT-Bench](https://github.com/mbzuai-nlp/M4GT-Bench) | No licence on GitHub [checked]. |
| [DetectRL](https://github.com/NLP2CT/DetectRL) (the 2024 version) | No licence on GitHub [checked]. Its 2026 follow-up, DetectRL-X, is MIT and we used that. |
| "MIRAGE" | The dataset under that name on Hugging Face is about image editing, not text [checked]. We found no text set under that name with an open licence. |

### How to rebuild it

Run these from the project folder, in this order.
`-I` keeps Python from loading code from the download folders.

```
.venv/bin/python -I scripts/text_fetch_newmodels.py
.venv/bin/python -I scripts/text_fetch_fairness.py
.venv/bin/python -I scripts/text_fetch_raid.py
.venv/bin/python scripts/text_data.py
.venv/bin/python scripts/text_features.py HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-360M-Instruct
```

- Each fetch script picks its sample with a fixed seed, so it picks the same texts again.
- `text_data.py` writes the old rows first, unchanged, then the new ones. Old row numbers never move, so old scores stay valid.
- Every row now says where it came from, which model wrote it ("human" for people), the kind of writing, whether it was reworded, and which source document it belongs to.
  Splitting by source document keeps an AI text and the person's text it came from on the same side.
- Scoring the new texts took SCORE_TIME on this laptop's processor.
