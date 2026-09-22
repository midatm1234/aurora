---
type: Model Contract
title: Aurora model and asset identity
description: Use the fork implementation with official 0.4-degree air-pollution weights
  and static data.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: example
  resource: https://microsoft.github.io/aurora/example_cams.html
  title: aurora-cams
---

# Assets

Use `AuroraAirPollution`, `aurora-0.4-air-pollution.ckpt`, and `aurora-0.4-air-pollution-static.pickle`. Retrieve only these files from the pinned Microsoft Hugging Face revision in the artifact registry and verify their checksums. The official example uses this air-pollution static asset; weather static fields alone omit chemical context.[^example]

Upstream source is a pinned reference, while the fork is the executable implementation. Inspect the imported `aurora.__file__` and source commit to detect package shadowing. A valid checksum authenticates bytes relative to a trusted manifest; it does not make arbitrary pickle input safe. Reject untrusted pickle and unverified checkpoint paths.

Trained refinement checkpoints are separate from the public backbone. Local or authorized-store supplied artifacts need both SHA256 and architecture/scaler compatibility metadata. An inaccessible trained model is `requires_artifact`; replay must never become training implicitly. Hardware, precision, kernels and library versions affect numerical reproducibility.

[^example]: Official CAMS example.
