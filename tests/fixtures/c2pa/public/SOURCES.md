# Public C2PA sample files

Copied unchanged from the C2PA public test files repository.

- Source: https://github.com/c2pa-org/public-testfiles/tree/main/legacy/1.4/image/jpeg
- Downloaded: 2026-10-31, repository commit 22beccc075707475b038d8789d0136c009e43143
- Licence: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0),
  https://creativecommons.org/licenses/by-sa/4.0/ . Files by Adobe, published by the
  Coalition for Content Provenance and Authenticity (C2PA). These files stay under
  CC BY-SA 4.0; the rest of markcheck is MIT.

They were signed with the C2PA tool's own test certificate, so none is on the C2PA trust list.

| File | What the source says it is | SHA-256 |
|---|---|---|
| adobe-20220124-A.jpg | No Content Credentials | f999fd78bfe8a83c96e468a078830ba94485bc1bc6fd086fb94a43bd29dd0f23 |
| adobe-20220124-CA.jpg | Valid label, one ingredient | cafc48c53e651f7ba4622d1f72783827074211e42b9634cc863ec3be3c7651b3 |
| adobe-20220124-E-dat-CA.jpg | Error: picture data changed after signing (hash mismatch) | dae9d121060cec4b6f27ee8acda85ad461cf75f2261d90b463319b787342d7f9 |
| adobe-20220124-E-sig-CA.jpg | Error: signature does not validate | 0d4c2774f1b7e94b9613bb952b0a76b6a178d22ac6d206d257d2af1376cbbff2 |
| adobe-20220124-E-uri-CA.jpg | Error: a part of the label was tampered with | 56014faf6b97532252c9bb90dcf484923a8aa7bba197e84a8555044a35e4f3a6 |
