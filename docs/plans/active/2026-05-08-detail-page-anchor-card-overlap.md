# Detail-page anchor card vs. anchored-proof viewer overlap

> Deferred from the Phase E follow-up consistency review (2026-05-08).
> Captures three design options discussed; pick one when picking this
> back up.

## Context

Each detail page currently shows ar.io anchor information in **two**
places:

1. **`ar.io anchor` card** (left rail of the detail-layout) —
   Transaction · Status badge · Signer Key · Block · "View on ar.io →"
2. **`Anchored proof` section** (full-width below the verify card) —
   left panel "In MLflow" with canonical bytes JSON; right panel
   "On ar.io" with signed commitment JSON, TX, and "View on ar.io →"

The TX, the "View on ar.io" link, and the signer-key/payload-hash
fields all appear in both places (the card surfaces them as labelled
rows; the proof viewer's signed commitment JSON contains them).
Reading the page top-to-bottom, an auditor sees three layers of
"ar.io is doing something":

- editorial-header status badge
- anchor card with status + TX + key
- proof viewer with same TX + signed commitment

Feels like demo-ish stacking rather than a single product surface.

## Three options considered

### Option A — Drop the anchor card entirely

Move what little it adds (live status, block height) into the proof
viewer's "On ar.io" panel header. One section per concern.

- **Pros**: simplest. Detail page left rails become Identity + (Activate
  on Models). Proof viewer is the single source of truth for what's on
  ar.io.
- **Cons**: live polling animation (TX appearing → status pulsing →
  Confirmed) currently lives in the anchor card body. Needs to move
  to the proof viewer panel header — small JS rewire, ~20 lines.
- **What's lost**: Block height + Signer Key as labelled rows. Both
  are still in the signed-commitment JSON below.

### Option B — Merge: panel header absorbs anchor-card content

The proof viewer's "On ar.io" panel grows a denser header carrying
TX + status + "View on ar.io" + key meta lines. Body stays the
signed-commitment JSON.

- **Pros**: visually one unit. Symmetric with the "In MLflow" panel.
- **Cons**: panel header gets busier. Proof viewer was deliberately
  stripped of explainer copy in Phase E to feel "less demo, more
  feature"; adding metadata lines may undermine the minimalism.

### Option C — Trim the anchor card, keep both

Reduce the anchor card to just what *isn't* in the proof viewer
footer:

- Status badge (live-updating)
- Block height (when available)
- *(Drop Transaction row + "View on ar.io" — both are in the proof
  viewer)*

- **Pros**: preserves live-status animation, no JS rewire. Anchor card
  becomes a small "delivery status" widget; proof viewer is the data.
- **Cons**: still two boxes about ar.io. Less visually clean.

## Recommendation

**Option A** if the live-anchoring animation isn't load-bearing for the
demo; **Option C** if it is.

The polling JS rewire for Option A is small but it's a non-trivial
piece of UX (first-time users training a model see the TX arrive
live and the badge flip green). Worth leaving until we know whether
that moment is actually used in demos.

## Files involved

- `templates/dataset_detail.html` — anchor card lives in left rail
- `templates/run_detail.html` — anchor card + polling JS targets
  `#anchoring-body` / `#turbo-status-badge` / `#signer-key-row` /
  `#block-row`
- `templates/model_detail.html` — has BOTH training and registration
  anchor cards (plural) — single proof viewer would need to absorb
  both, or split into two proof viewers
- `templates/decision_detail.html` — anchor card + same polling JS
  pattern as run_detail
- `templates/_proof_viewer.html` — would gain panel-header metadata
  if Option B
- `app/main.py` — `_hydrate_*_envelope` paths persist the JSON;
  no backend changes needed for any of A/B/C

## When to pick this up

After at least one customer demo with the current shape so we have
a signal on whether the live-anchoring animation is part of the
demo moment people actually point at.
