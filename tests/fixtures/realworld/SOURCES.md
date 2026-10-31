# Files labelled by real AI products

Every other test file in markcheck was labelled by us. These were labelled by real
tools (Adobe Photoshop and Firefly, OpenAI's image service, AUTOMATIC1111 with SDXL)
or by the label makers' own sample code. Each is copied unchanged from the source
below. Fetched 2026-10-31. The licence of each file is the licence of its source; the
rest of markcheck is MIT.

## Committed

| File | Source | Author | Licence | Label it carries |
|---|---|---|---|---|
| openai_gpt_image_commons_map.png | https://commons.wikimedia.org/wiki/File:Australians_with_Bangladeshi_ancestry.png (original file: https://upload.wikimedia.org/wikipedia/commons/f/f1/Australians_with_Bangladeshi_ancestry.png) | GMWiki2026 (uploaded 2026-09-12) | CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ | C2PA signed label from OpenAI: signer "OpenAI Media Service (OpenAI OpCo, LLC)", certificate from Trufo's C2PA CA, tool "OpenAI Media Service API", AI named "gpt-image 2.0", source type trainedAlgorithmicMedia. On the official C2PA trust list. |
| adobe_photoshop_firefly_ghost.png | https://github.com/adobe/trustmark/blob/main/images/ghost.png (commit 6df5a0a29b51cbec832695fdf02051a1937e37bd) | Adobe | MIT (repository LICENSE, Copyright 2023 Adobe) | C2PA signed label from Adobe Photoshop 26.4.1 naming Adobe Firefly, source type compositeWithTrainedAlgorithmicMedia, signed 2025-03-20 by "Adobe C2PA" (Adobe Product Services G3). On the older interim trust list, not the official one. |
| adobe_firefly_label_online.jpg | https://github.com/contentauth/c2pa-node/blob/main/tests/fixtures/cloud-only-firefly.jpg (commit 89d2ee25b1de5b143dcd3bf2698efd4366623b52) | Adobe / Content Authenticity Initiative | MIT (repository licence) | Adobe Firefly output whose C2PA label is kept only online (cai-manifests.adobe.com). markcheck reports the pointer and does not fetch it. |
| adobe_trustmark_q_ufo.png | https://github.com/adobe/trustmark/blob/main/images/ufo_240_Q.png (commit 6df5a0a…) | Adobe | MIT | Adobe TrustMark variant Q hidden mark, message "mysecret". |
| adobe_trustmark_softbinding_signed.jpg | https://github.com/adobe/trustmark/blob/main/c2pa/example_wm_signed.jpg (commit 6df5a0a…) | Adobe | MIT | C2PA label signed with the C2PA test certificate, with a soft-binding assertion naming TrustMark Q. The current TrustMark Q model cannot read the mark in this 2024 file (the library's own decode fails too), so only the label is expected. |
| adobe_trustmark_unmarked_ufo.jpg | https://github.com/adobe/trustmark/blob/main/images/ufo_240.jpg (commit 6df5a0a…) | Adobe | MIT | None. Control: the picture Adobe's TrustMark samples start from. |
| sdxl_automatic1111_ostris.png | https://huggingface.co/datasets/ostris/sdxl_10_reg/blob/main/random_1024_1024/1690612304_1_0_2366855343.png (revision b1554b708d12fa8f6d8b436dfb3ab8285d5d9376) | ostris | MIT (dataset card) | SDXL 1.0 base output from AUTOMATIC1111 v1.5.1. PNG text "parameters" names Model: sd_xl_base_1.0. No Stable Diffusion hidden mark: AUTOMATIC1111 does not add one. |

SHA-256:

```
a456f518683f0d48c892883cea6248937de18c93a3ea36db158b9b362b28c4bb  adobe_photoshop_firefly_ghost.png
f388997ad006ae755e332a0fe3bae2a6a5f77bd8222f0e71ed09301c289e6922  adobe_trustmark_q_ufo.png
075f97d7cf1206d26c3166a580219043671a14e380fc91455b9a23f5ff002318  openai_gpt_image_commons_map.png
8b010a4de78588dd6f982e2b08fb9f6e1a41e2659bb0ddc9c29b0b7225c33bae  sdxl_automatic1111_ostris.png
fea3d318f22d6d0a06637fd39f3e8a7ec830392d8ea68c8f81454c7c8805be14  adobe_firefly_label_online.jpg
1d219db4fd618ae2cd4e0906024afae6f0113ba4d9fc705452f07a3513228ce2  adobe_trustmark_softbinding_signed.jpg
193203acc400e688083441cb8eaf4df2d2486f33eb81ece806dba9d5c788af1b  adobe_trustmark_unmarked_ufo.jpg
```

## Checked but not redistributed

Wikimedia Commons: we listed about 1,500 candidate files ("Category:AI-generated images" and its subcategories, plus search hits for C2PA, Content Credentials, gpt-image, Firefly, Imagen, Gemini, DALL-E and Bing Image Creator) and read the first 64 KB of each looking for label bytes. Commons rate limits stopped the scan early, so most were never read. Two files read so far carried a signed label, both from OpenAI; no IPTC-only, AIGC, Firefly, Google or Microsoft label turned up.

| URL | Licence / why not committed | What markcheck reported |
|---|---|---|
| https://commons.wikimedia.org/wiki/File:1960%27s_art_of_cow_getting_abducted_by_UFO_in_midwest_(GPT_Image_2.5_September_2026).png (uploaded by Karl432, 2026-09-17) | Public domain, so it could be committed, but it is 2.9 MB and would take the folder over the 5 MB budget. | OpenAI signed label, signer OpenAI Media Service, certificate from SSL.com's C2PA CA (a second OpenAI signing path), tool OpenAI Media Service API, AI named "gpt-image 2.0" (the Commons title says 2.5), made by an AI model, signature fine, unchanged, on the official trust list. |
| https://github.com/contentauth/c2pa-rs/blob/main/sdk/tests/fixtures/ocsp.jpg | MIT/Apache-2.0, but the label names a private person and links a LinkedIn profile; `adobe_photoshop_firefly_ghost.png` covers the same case. | Photoshop 26.9.0 + Firefly label, signer Adobe C2PA, signature fine, unchanged, not on the official list, on the interim list (before the fix: "anyone could have made a certificate with this name"). |
| https://github.com/adobe/trustmark/blob/main/images/ghost_P.png and images/ufo_240_P.png | MIT. Not committed because markcheck misses them (see below), and a known miss cannot sit in the manifest without failing the "every labelled file is found" check. | **Missed.** TrustMark variant P marks. markcheck reads variant Q only; it now says so ("variant Q checked; marks made with B, C, P are not read"). Reading P needs Adobe's P model files, which we did not install. |
| https://github.com/adobe/trustmark/blob/main/images/ghost_Q.png | MIT; same picture as ghost.png, so left out to save space. | TrustMark Q found (0 bits fixed). The C2PA label of ghost.png is gone from this copy. |
| https://github.com/contentauth/c2pa-rs/blob/main/sdk/tests/fixtures/cloud.jpg | MIT/Apache-2.0; same case as adobe_firefly_label_online.jpg. | Label kept online at cai-manifests.adobe.com, not checked. |
| https://github.com/contentauth/c2pa-rs/blob/main/sdk/tests/fixtures/prerelease.jpg | MIT/Apache-2.0; a pre-release label format, not a real product. | "Seems to carry a signed label, but the label is damaged and could not be read." |
| https://github.com/contentauth/c2pa-rs/blob/main/sdk/tests/fixtures/sample1.heic, sample1.heif, sample1.avif, mars.webp, sample1.png, sample1.webp | MIT/Apache-2.0; no labels inside (they are inputs for signing tests). | No label found. HEIC/HEIF: the two hidden-mark checks say "could not open this file as a picture" (Pillow has no HEIF reader). AVIF opened and was checked. We found no real labelled HEIC or AVIF file under an open licence. |
| https://huggingface.co/datasets/ostris/sdxl_10_reg files 1690612315_2_0_2366855344.png and 1690612325_3_0_2366855345.png | MIT; one is enough, and one prompt names a real actor. | AUTOMATIC1111 settings found (model sd_xl_base_1.0); no Stable Diffusion mark (68-71 of 136 bits, 98 needed). |
| https://huggingface.co/datasets/parrotzone/sdxl-1.0 | OpenRAIL++ (use restrictions; not a plain open licence), not downloaded. | Not checked. |
