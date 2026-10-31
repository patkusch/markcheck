# Bundled data

## C2PA-TRUST-LIST.pem

The official C2PA Trust List: the certificates of organisations that passed the
C2PA conformance programme. markcheck uses it, offline, to say whether the
certificate that signed a label chains up to one of these organisations.

- Source: https://github.com/c2pa-org/conformance-public/blob/main/trust-list/C2PA-TRUST-LIST.pem
- Downloaded: 2026-10-31, repository commit d6dec63476e2636095d1291d24b29ffa2d4e1ac0
  (list issue date 2026-10-23, next update due 2027-01-21)
- SHA-256: b3949271a07b007e92ef1d18050b55388f5e0d2686a16fd864176c48421af8ae
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ . Copyright the Coalition for
  Content Provenance and Authenticity (C2PA). Redistributed unchanged.

The list changes over time. To refresh it, download the file again from the
source above and update the date, commit and checksum here.

Being on the list is not a statement that a picture is accurate. Not being on it
does not mean a label is forged: some real makers use certificates from older
lists or have not joined yet.

## INTERIM-TRUST-ANCHORS.pem and INTERIM-TRUST-ALLOWED.pem

The older Content Credentials interim trust list (ITL): root certificates
(anchors) and individual signing certificates (allowed list). C2PA froze it on
1 January 2026 and replaced it with the official list above. Real makers still
sign with certificates on it; for example Adobe's Photoshop and Firefly
certificate in 2025 chains to it but not to the official list. markcheck reports
it on a separate line ("on the older interim list"), as C2PA asks checkers to
keep the two apart, and only checks it when the official list says no.

- Source: https://github.com/contentauth/verify-site/tree/main/static/trust
  (anchors.pem and allowed.pem, renamed)
- Downloaded: 2026-10-31, repository commit 83b2e30684086a62e132f5d4c148abd0151e9486
- SHA-256: anchors db57e50b9bd4e4c786cbb91f38fd85e2af6a9d5412438a8cddcc42688df07ae5,
  allowed dfe328032e56f98b83f27f48a3642314124568b9f71ff471dffabf9b307c65d5
- Licence: Apache License 2.0 (the verify-site repository's licence),
  https://www.apache.org/licenses/LICENSE-2.0 . Copyright Adobe. Redistributed unchanged.
- About the freeze: https://opensource.contentauthenticity.org/docs/conformance/trust-lists
