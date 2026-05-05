# Verification Correctness — Pieces B + C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use opus 4.7 (highest thinking mode) for all subagent work.

**Goal:** Close two correctness gaps in the demo's verification UI: (B) make "Proof Found" mean *the proof was retrieved from Arweave* instead of inferring from signature validity, and (C) make "Signature Confirmed" verify the proof was signed by an *expected* user, not just that the signature is mathematically valid.

**Architecture:** Plugin-first. New plugin functions in `ario_mlflow/verify.py` and CLI flags become available to any external consumer of `ario-mlflow`. Demo wraps the plugin as a thin presentation layer. Demo-specific UX (tamper button, tooltip copy, auto-detect default) stays demo-side.

**Tech Stack:** FastAPI + Jinja2 (demo), `ario-mlflow` plugin (Python), Ed25519 signatures, MLflow 3.x, Arweave via Turbo gateway.

---

## Context — why this exists

### Piece A — already shipped

Shipped in PR #9 commit `5c828bf` — `templates/run_detail.html:178` Signature Confirmed row no longer requires `attestation_level >= 3`. Mirrors the round-1 fix on `decision_detail.html`. Listed here only for completeness; no work remaining.

### Piece B — `proof_found` correctness

**Today**: the "Proof Found" row infers from `permanent_copy_found` and falls back to `arweave_tx_id` truthiness. That's a proxy for "did the signature verify?", not a real "was the proof found?" check. Logically incorrect: a tampered proof on Arweave would still be *found* (we retrieved it), just *invalid* on the signature row.

**After**: the plugin gains `verify_proof_by_tx(tx_id, ...)` that explicitly returns `proof_found: True/False`. Demo passes that boolean to templates. External plugin consumers get a single fetch+verify entry point.

### Piece C — trusted-issuer-key check

**Today**: `verify_signature` only checks the math (signature valid against the embedded public key). It does NOT check *who* the embedded public key belongs to. Any throwaway keypair would pass.

**After**: `verify_signature` accepts an optional `trusted_issuer_keys` set. When provided, the signature row passes only if the embedded key matches one of the trusted keys. CLI flag (`--trusted-issuer-key`) and demo env var (`VAIDR_TRUSTED_ISSUER_KEY`) configure it. Demo auto-detects the local signer key from `keys/ed25519_public.json` if no env var is set.

**Bonus tamper button**: "Use a proof signed by someone else" — demonstrates the new check by re-signing a saved envelope with a freshly-generated keypair. Without trusted-issuer-key, the demo has no way to *show* the signer-identity claim being enforced.

### Open planning questions (carry over from previous chat)

1. **Run all tasks back-to-back, or pause for review after Piece B before kicking off Piece C?** — User hadn't decided.
2. **Bonus tamper button placement: all three templates (decision_detail, run_detail, model_chain), or just decision_detail to start?** — User hadn't decided.

The new chat should ask these before dispatching.

---

## File Structure

| File | Role |
|---|---|
| `ario_mlflow/verify.py` | Plugin: extend `verify_signature`, `full_verify`; add new `verify_proof_by_tx` |
| `ario_mlflow/cli.py` | Plugin: add `--trusted-issuer-key` flag to verify subcommands |
| `app/config.py` | Demo: add `Settings.trusted_issuer_key` + auto-detect helper |
| `app/ui.py` | Demo: `_verify_envelope` switches to `verify_proof_by_tx`, threads trusted keys |
| `app/main.py` | Demo: `verify_decision` route same simplification; new tamper route for swap-signer |
| `app/tamper.py` | Demo: new `swap_signer` tamper kind |
| `templates/decision_detail.html` | Demo: Row 1 uses `proof_found`; Row 3 tooltip copy update; new tamper button |
| `templates/run_detail.html` | Same as decision_detail (mirror) |
| `templates/model_chain.html` | Same — both training and registration mini-verify rows |
| `tests/test_plugin_verify.py` | Plugin unit tests (verify_signature with/without trusted set, verify_proof_by_tx happy + sad paths, full_verify thread-through) |
| `tests/test_swap_signer_tamper.py` | Demo tamper backend test |
| `README.md` | Configuration section: `VAIDR_TRUSTED_ISSUER_KEY`, hex format, auto-detect default, demo caveat |
| `ROADMAP.md` | Mark single-key trusted-issuer check ✅ shipped under External identity binding |

---

## Task 1 — Plugin: `verify_proof_by_tx` helper (Piece B)

**Files:**
- Modify: `ario_mlflow/verify.py` (add new function near `full_verify` at line 664)
- Test: `tests/test_plugin_verify.py` (new or existing)

**Implementation:**

New function signature:
```python
def verify_proof_by_tx(
    tx_id: str,
    *,
    anchor: ArweaveAnchor,
    proof_engine: ProofEngine,
    mlflow_client=None,
    ario_client: "ArioVerifyClient | None" = None,
    trusted_issuer_keys: set[str] | None = None,
    min_attestation_level: int = DEFAULT_MIN_ATTESTATION_LEVEL,
) -> dict:
    """Fetch the envelope from Arweave by TX ID and run all four checks.

    Returns the same shape as ``full_verify`` plus a ``proof_found`` bool
    indicating whether the fetch succeeded. When the fetch fails, all
    sub-check ``ok`` fields are ``None`` (not ``False`` — the checks
    weren't actually run).
    """
    plugin_envelope = anchor.fetch_proof(tx_id)
    if plugin_envelope is None:
        return {
            "proof_found": False,
            "signature": {"ok": None, "reason": "no_envelope"},
            "anchored_bytes": {"ok": None, "reason": "no_envelope"},
            "source_of_truth": {"ok": None, "reason": "no_envelope"},
            "ario_attestation": {"ok": None, "reason": "no_envelope"},
            "overall": None,
        }
    plugin_envelope["_tx_id"] = tx_id
    result = full_verify(
        plugin_envelope,
        proof_engine=proof_engine,
        mlflow_client=mlflow_client,
        ario_client=ario_client,
        trusted_issuer_keys=trusted_issuer_keys,
        min_attestation_level=min_attestation_level,
    )
    result["proof_found"] = True
    return result
```

(`trusted_issuer_keys` parameter on `full_verify` is added in Task 3 — for now just thread it through; tests in this task use `None`.)

**Steps:**
- [ ] Write failing test `test_verify_proof_by_tx_returns_proof_found_false_when_fetch_fails`
- [ ] Write failing test `test_verify_proof_by_tx_returns_proof_found_true_with_valid_envelope`
- [ ] Implement `verify_proof_by_tx`
- [ ] Run tests, confirm pass
- [ ] Commit

---

## Task 2 — Demo: switch callers to `verify_proof_by_tx` (Piece B)

**Files:**
- Modify: `app/ui.py::_verify_envelope` (around line 112-145)
- Modify: `app/main.py::verify_decision` route (around line 599-665)

**Steps:**
- [ ] Replace inline `anchor.fetch_proof + full_verify` in `_verify_envelope` with single `verify_proof_by_tx` call
- [ ] Persist `proof_found` field on `result` dict so templates can read it
- [ ] Same swap in `verify_decision` route in `app/main.py`
- [ ] Run existing tests, confirm no regressions
- [ ] Commit

---

## Task 3 — Templates: use `v.proof_found` for Row 1 (Piece B)

**Files:**
- Modify: `templates/decision_detail.html` (Row 1 around line 230-245 — current `permanent_copy_found is sameas true` pattern)
- Modify: `templates/run_detail.html` (Row 1 — same pattern)
- Modify: `templates/model_chain.html` (training mini-verify Row 1 around line 209-215, registration mini-verify Row 1 around line 355-361)

**Pattern** (replace existing fallback heuristic with explicit check):
```jinja
{% if v.proof_found is sameas true %}
    <span class="check">PASS</span>
{% elif v.proof_found is sameas false %}
    <span class="cross">FAIL</span>
{% else %}
    <span class="badge badge-yellow">Pending</span>
{% endif %}
```

**Steps:**
- [ ] Apply pattern to `decision_detail.html` Row 1
- [ ] Apply pattern to `run_detail.html` Row 1
- [ ] Apply pattern to `model_chain.html` training mini-verify Row 1
- [ ] Apply pattern to `model_chain.html` registration mini-verify Row 1
- [ ] Manual smoke test: boot uvicorn locally, make a decision, verify Row 1 renders correctly
- [ ] Commit

**(Pause for user review after Task 3 if user chose checkpoint pause.)**

---

## Task 4 — Plugin: `trusted_issuer_keys` on `verify_signature` + `full_verify` (Piece C)

**Files:**
- Modify: `ario_mlflow/verify.py::verify_signature` (around line 130-200, the current signature check function)
- Modify: `ario_mlflow/verify.py::full_verify` (line 664, add parameter and thread to `verify_signature`)
- Modify: `ario_mlflow/verify.py::verify_proof_by_tx` (already accepts the param from Task 1; remove the placeholder)
- Test: `tests/test_plugin_verify.py`

**Implementation:**
```python
def verify_signature(
    envelope: dict,
    proof_engine: ProofEngine,
    *,
    trusted_issuer_keys: set[str] | None = None,
) -> dict:
    """Verify the envelope's Ed25519 signature against the embedded
    public key. When ``trusted_issuer_keys`` is provided (a set of hex
    public-key strings), additionally require the embedded key to be
    in the trusted set — returns ``ok=False, reason="untrusted_issuer"``
    on mismatch.
    """
    # ... existing math-validity check ...
    if not result["ok"]:
        return result
    if trusted_issuer_keys is not None:
        embedded_hex = result.get("embedded_public_key_hex") or _extract_pubkey_hex(envelope)
        if embedded_hex not in trusted_issuer_keys:
            return {
                "ok": False,
                "reason": "untrusted_issuer",
                "embedded_public_key_hex": embedded_hex,
                **{k: v for k, v in result.items() if k not in ("ok", "reason")},
            }
    return result
```

**Steps:**
- [ ] Write failing test `test_verify_signature_passes_when_embedded_key_in_trusted_set`
- [ ] Write failing test `test_verify_signature_fails_with_untrusted_issuer_when_trusted_set_provided`
- [ ] Write failing test `test_verify_signature_unchanged_when_trusted_keys_is_none`
- [ ] Implement the trusted-keys check in `verify_signature`
- [ ] Add `trusted_issuer_keys=None` to `full_verify` signature; thread to `verify_signature` call
- [ ] Run tests, confirm pass
- [ ] Commit

---

## Task 5 — Plugin: `--trusted-issuer-key` CLI flag (Piece C)

**Files:**
- Modify: `ario_mlflow/cli.py` (`build_parser` at line 485, plus the verify subcommand handlers)

**Steps:**
- [ ] Add `--trusted-issuer-key <hex>` flag (single value for v1; multi-key deferred per user)
- [ ] Plumb through to `full_verify` calls in CLI
- [ ] Add CLI test (or smoke test via `python -m ario_mlflow.cli verify-run ...`)
- [ ] Commit

---

## Task 6 — Demo: `Settings.trusted_issuer_key` + auto-detect (Piece C)

**Files:**
- Modify: `app/config.py::Settings` (add field), `from_env` already coerces strings
- Modify: `app/ui.py::_verify_envelope` (read setting, decode if base64, pass to plugin)
- Modify: `app/main.py::verify_decision` route (same)

**Implementation sketch:**
```python
@dataclass
class Settings:
    # ... existing fields ...
    trusted_issuer_key: str | None = None  # hex; auto-detected from public key file if None

    def get_trusted_issuer_keys(self) -> set[str] | None:
        """Resolve trusted issuer key: env var, then auto-detect from ed25519_public_key_path."""
        if self.trusted_issuer_key:
            return {self.trusted_issuer_key.lower()}
        try:
            with open(self.ed25519_public_key_path) as f:
                key_data = json.load(f)
            base64_key = key_data["key"]
            hex_key = base64.b64decode(base64_key).hex()
            return {hex_key}
        except Exception:
            return None
```

**Steps:**
- [ ] Add `trusted_issuer_key` field + auto-detect helper to `Settings`
- [ ] Add demo test for auto-detect (with fixtures)
- [ ] Update `_verify_envelope` to call `get_trusted_issuer_keys()` and pass to plugin
- [ ] Update `verify_decision` route same
- [ ] Run tests
- [ ] Commit

---

## Task 7 — Templates: Row 3 tooltip copy + bonus tamper button (Piece C)

**Files:**
- Modify: `templates/decision_detail.html` (Row 3 tooltip + new tamper button)
- Modify: `templates/run_detail.html` (same)
- Modify: `templates/model_chain.html` (same)
- Modify: `app/tamper.py` (new `swap_signer` tamper kind)
- Modify: `app/main.py` (new `/tamper/swap-signer/{event_type}/{event_id}` endpoint, gated behind `demo_mode`)

**Tooltip (Option A from prior chat — minimum change from today's wording):**
```
The proof carries a valid signature from the expected user. ar.io independently
re-verifies the signature. FAIL means the proof was altered after signing or
signed by an unrecognized party.
```

**Swap-signer tamper backend:**
- Generate fresh Ed25519 keypair
- Read existing envelope from saved artifact (`payload.json`)
- Re-sign canonical bytes with the new private key
- Replace envelope's `signature` and embedded `public_key` fields
- Write back to saved artifact path
- Snapshot original envelope for restore

**Steps:**
- [ ] Add `swap_signer` tamper kind to `app/tamper.py`
- [ ] Add `/tamper/swap-signer/{event_type}/{event_id}` route to `app/main.py` (under `demo_mode` gate)
- [ ] Update Row 3 tooltip on `decision_detail.html`
- [ ] Update Row 3 tooltip on `run_detail.html`
- [ ] Update Row 3 tooltip on `model_chain.html` mini-verify rows
- [ ] Add "Use a proof signed by someone else" button to tamper section in each template (per user's "all three templates" or "just decision_detail" decision)
- [ ] Wire button JS to call new endpoint (mirror existing tamper button pattern)
- [ ] Manual smoke test: trigger swap-signer on a decision → verify Row 3 shows FAIL with `untrusted_issuer` reason
- [ ] Add test for swap-signer endpoint
- [ ] Commit

---

## Task 8 — README + ROADMAP updates (Piece C)

**Files:**
- Modify: `README.md` (Configuration section)
- Modify: `ROADMAP.md` (External identity binding entry)

**README addition** (under Configuration / Verification):

```markdown
### Trusted issuer key

The verification flow checks not just that the proof's signature is mathematically valid,
but also that it was signed by an *expected* user. Configure the trusted signer via:

```
VAIDR_TRUSTED_ISSUER_KEY = <64-character hex Ed25519 public key>
```

If unset, the demo auto-detects the trusted key from `keys/ed25519_public.json` (the
local signer). This means in the demo's default configuration, the trusted-key check
verifies that proofs were signed by the demo's own signer — a tautology in normal
operation, but it catches the "Use a proof signed by someone else" tamper button which
re-signs a proof with a freshly-generated keypair.

Production deployments should set `VAIDR_TRUSTED_ISSUER_KEY` explicitly to the
operator's authorized signer key, independent of the local key file. Multi-key trusted
lists (comma-separated) are not yet supported — see ROADMAP.
```

**ROADMAP update**: under "External identity binding", mark single-key trusted-issuer check as ✅ shipped; note multi-key list as future iteration.

**Steps:**
- [ ] Add README section
- [ ] Update ROADMAP entry
- [ ] Commit

---

## Task 9 — Final code review + open PR

- [ ] Run full test suite (`pytest -v`); all pass
- [ ] Manual smoke test: boot uvicorn, walk through every verify page, click each tamper button, confirm rows render correctly
- [ ] Open PR with detailed body covering both pieces, the architectural rationale, and the test plan

---

## Deferred / out of scope

- Multi-key trusted issuer list (env var as comma-separated hex) — user opted for single-key v1
- Lifecycle backfill from MLflow on startup (made redundant by the Reset feature)
- Anything from the strategic backlog (`memory/project_strategic_backlog.md`)

## Constraints

- Use opus 4.7 for all subagent work (highest thinking mode)
- Implementer → spec reviewer → code-quality reviewer pattern per `superpowers:subagent-driven-development`
- Per-phase manual review per `feedback_per_phase_manual_review.md` (the new chat should ask the user about checkpoint pause vs. back-to-back before dispatching)
- Each task: small commit, clear message
- No backwards-compat shims; pre-prod codebase
- Match existing code style (`dict | None`, snake_case, minimal comments)
