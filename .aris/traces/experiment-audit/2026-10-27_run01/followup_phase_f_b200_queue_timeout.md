# Phase F B200 scheduler timeout

Date: 2026-10-27 (UTC)
Profile: `billionaireproject8` (workspace `kieusontung6`)
Run tag: `phasef20261027r3r6a`

## Setup

The CPU setup entrypoint completed and registered the prepared Modal volume. Its manifest reports the pinned DeepSeek-R1-Distill-Qwen-1.5B revision, 11,244 training rows, the existing export hashes, and the user's accepted-overlap decision. The setup mode was `registered_existing_exports_without_rebuild`; the final registration did not rerun the leakage checker.

The failed setup attempt exposed that the checker did not read Verl's nested `reward_model.ground_truth` field. Local code was updated to support that field and an explicit allow-overlaps mode that records overlap evidence without removing rows. Those checker changes were not exercised by the final registration path and remain unverified.

## GPU submission

- The project budget gate passed at $37.99260592 metered with a $4.9372 R3-R6-only reserve. The local cap is $27 additional with a $25.65 hard stop.
- Modal app: `ap-grnjMhfMqpjowvFaDXx9cQ` ([dashboard](https://modal.com/apps/kieusontung6/main/ap-grnjMhfMqpjowvFaDXx9cQ)).
- R3 function call: `fc-01M5Y8KHS88RDDV9VPJ6NKFYBT`, enqueued at 2026-10-27T08:46:06.639673Z. It remained `pending` with no container assignment for 3,900 seconds.
- The CPU coordinator hit its 3,900-second timeout. The app was stopped; Modal then reported the R3 call as `terminated`, with zero active containers. No B200 container, trainer update, checkpoint, timing, or queue artifact was produced. R4-R6 were never submitted.

## Billing

Final Modal summary: $38.23260592 metered and $0 billed after credits. This is $0.28144098 above the recorded $37.95116494 baseline. The usage came from CPU setup/coordinator and associated setup activity; no B200 container was allocated.

Launcher output and the budget preflight are under `remote_artifacts/phasef20261027r3r6a/`.
