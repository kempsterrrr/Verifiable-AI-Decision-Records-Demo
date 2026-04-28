"""Re-derive a prediction's hashes from MLflow trace storage and compare to
the canonical proof on Arweave.

Architecture: MLflow's recorded span data is the source of truth for what
the model was actually given as input and what it returned. Arweave is the
source of truth for what was anchored. This function compares the two — if
someone has mutated the trace's span data after anchoring, the hashes will
diverge and we report a mismatch.

Tags (``ario.decision_id``, ``ario.arweave_tx``) are used purely for
navigation between these two sources of truth; nothing about the
verification rests on the tag values being correct beyond pointing to the
right records.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import mlflow

from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.proof import canonical_json, hash_data

logger = logging.getLogger(__name__)


@dataclass
class PredictionVerificationResult:
    decision_id: str
    found: bool
    arweave_tx_id: Optional[str]
    match_input: Optional[bool]
    match_output: Optional[bool]
    anchored_input_hash: Optional[str]
    recomputed_input_hash: Optional[str]
    anchored_output_hash: Optional[str]
    recomputed_output_hash: Optional[str]
    error: Optional[str] = None

    @property
    def overall_match(self) -> Optional[bool]:
        """Both checks must pass and have actually run for overall PASS."""
        if self.match_input is None or self.match_output is None:
            return None
        return bool(self.match_input and self.match_output)


def _unwrap_span_inputs(raw: object) -> object:
    """Strip the function-argument wrapping MLflow adds to span.inputs.

    @mlflow.trace captures the wrapped function's args as a dict keyed by
    parameter name. VerifiedModel.predict's parameter is ``input_data``, so
    the recorded inputs come back as ``{"input_data": <actual_dict>}``. The
    proof was hashed over the actual dict, not the wrapper.
    """
    if isinstance(raw, dict) and len(raw) == 1 and "input_data" in raw:
        return raw["input_data"]
    return raw


def _parse_span_output_prediction(raw_outputs: object) -> object:
    """Recover the structured prediction value from span.outputs.

    MLflow's span auto-serializer stringifies non-JSON values. A numpy
    output ``[1]`` is recorded as the string ``"[1]"``. The proof was
    hashed over ``{"prediction": [1]}``, so we parse the string back via
    ast.literal_eval (safe — only Python literals) before re-hashing.

    Returns the {"prediction": <parsed>} dict ready for hashing, or
    falls back to the raw outputs if the parse fails.

    Note: this strategy is specific to the demo's sklearn output shape
    (list-of-ints). Non-list predictions or custom output schemas may
    require a different unwrapping strategy.
    """
    if not isinstance(raw_outputs, dict):
        return {"prediction": raw_outputs}
    pred = raw_outputs.get("prediction")
    if isinstance(pred, str):
        try:
            return {"prediction": ast.literal_eval(pred)}
        except (ValueError, SyntaxError):
            return {"prediction": pred}
    return {"prediction": pred}


def verify_prediction(
    decision_id: str,
    tracking_uri: Optional[str] = None,
    arweave: Optional[ArweaveAnchor] = None,
) -> PredictionVerificationResult:
    """Verify a single prediction by re-deriving hashes from MLflow span data.

    Steps:
    1. Find the trace tagged with ``ario.decision_id == decision_id``.
       (Tag use is navigational — points to the trace; doesn't claim what
       the trace contains.)
    2. Read the trace's recorded inputs/outputs from span data.
    3. Look up ``ario.arweave_tx`` to fetch the canonical proof.
    4. Re-canonicalize and hash the recorded data, compare to the proof's
       ``record.input_hash`` and ``record.output_hash``.

    Returns a :class:`PredictionVerificationResult` with per-field match
    flags. ``overall_match`` is True iff both input and output match.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    try:
        # Multi-experiment search to find the trace.
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments] or ["0"]
        # Note: caller is responsible for sanitizing decision_id; this is a
        # plugin function so we accept the value as given (UUIDs in the
        # demo). A real production caller would escape if it ever flowed
        # from untrusted input.
        traces = client.search_traces(
            experiment_ids=experiment_ids,
            filter_string=f"tags.`ario.decision_id` = '{decision_id}'",
            max_results=1,
        )
    except Exception as e:
        return PredictionVerificationResult(
            decision_id=decision_id, found=False, arweave_tx_id=None,
            match_input=None, match_output=None,
            anchored_input_hash=None, recomputed_input_hash=None,
            anchored_output_hash=None, recomputed_output_hash=None,
            error=f"Trace lookup failed: {e}",
        )

    if not traces:
        return PredictionVerificationResult(
            decision_id=decision_id, found=False, arweave_tx_id=None,
            match_input=None, match_output=None,
            anchored_input_hash=None, recomputed_input_hash=None,
            anchored_output_hash=None, recomputed_output_hash=None,
            error="No trace found for decision_id.",
        )

    trace_info = traces[0].info
    trace_id = getattr(trace_info, "trace_id", None) or getattr(trace_info, "request_id", None)
    tags = dict(getattr(trace_info, "tags", {}) or {})
    arweave_tx_id = tags.get("ario.arweave_tx")

    if not arweave_tx_id:
        return PredictionVerificationResult(
            decision_id=decision_id, found=True, arweave_tx_id=None,
            match_input=None, match_output=None,
            anchored_input_hash=None, recomputed_input_hash=None,
            anchored_output_hash=None, recomputed_output_hash=None,
            error="Decision is not anchored yet (no ario.arweave_tx).",
        )

    # Fetch canonical proof from Arweave.
    if arweave is None:
        arweave = ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )
    if not arweave.enabled:
        return PredictionVerificationResult(
            decision_id=decision_id, found=True, arweave_tx_id=arweave_tx_id,
            match_input=None, match_output=None,
            anchored_input_hash=None, recomputed_input_hash=None,
            anchored_output_hash=None, recomputed_output_hash=None,
            error="Arweave anchor is disabled — cannot fetch proof.",
        )
    proof = arweave.fetch_proof(arweave_tx_id)
    if not proof:
        return PredictionVerificationResult(
            decision_id=decision_id, found=True, arweave_tx_id=arweave_tx_id,
            match_input=None, match_output=None,
            anchored_input_hash=None, recomputed_input_hash=None,
            anchored_output_hash=None, recomputed_output_hash=None,
            error=f"Could not fetch proof from Arweave for tx {arweave_tx_id}.",
        )

    record = proof.get("record", {}) or {}
    anchored_input_hash = record.get("input_hash")
    anchored_output_hash = record.get("output_hash")

    # Read what MLflow CURRENTLY records for this trace.
    try:
        full_trace = client.get_trace(trace_id)
        spans = full_trace.data.spans if full_trace and full_trace.data else []
    except Exception as e:
        return PredictionVerificationResult(
            decision_id=decision_id, found=True, arweave_tx_id=arweave_tx_id,
            match_input=None, match_output=None,
            anchored_input_hash=anchored_input_hash,
            recomputed_input_hash=None,
            anchored_output_hash=anchored_output_hash,
            recomputed_output_hash=None,
            error=f"Could not load trace data: {e}",
        )

    if not spans:
        return PredictionVerificationResult(
            decision_id=decision_id, found=True, arweave_tx_id=arweave_tx_id,
            match_input=None, match_output=None,
            anchored_input_hash=anchored_input_hash,
            recomputed_input_hash=None,
            anchored_output_hash=anchored_output_hash,
            recomputed_output_hash=None,
            error="Trace has no span data.",
        )

    span = spans[0]
    raw_inputs = getattr(span, "inputs", None)
    raw_outputs = getattr(span, "outputs", None)

    unwrapped_input = _unwrap_span_inputs(raw_inputs)
    output_for_hash = _parse_span_output_prediction(raw_outputs)

    recomputed_input_hash = hash_data(canonical_json(unwrapped_input))
    recomputed_output_hash = hash_data(canonical_json(output_for_hash))

    return PredictionVerificationResult(
        decision_id=decision_id,
        found=True,
        arweave_tx_id=arweave_tx_id,
        match_input=(recomputed_input_hash == anchored_input_hash) if anchored_input_hash else None,
        match_output=(recomputed_output_hash == anchored_output_hash) if anchored_output_hash else None,
        anchored_input_hash=anchored_input_hash,
        recomputed_input_hash=recomputed_input_hash,
        anchored_output_hash=anchored_output_hash,
        recomputed_output_hash=recomputed_output_hash,
        error=None,
    )


def verify_model_lifecycle(
    model_name: str,
    model_version: str,
    anchor: ArweaveAnchor,
    ario_client,  # ArioMlflowClient — typed via duck-typing to avoid circular import
    proof_engine=None,  # ProofEngine | None
) -> dict:
    """Verify the full anchored lifecycle of a single model version.

    For each event in the version's chain (dataset_anchored,
    training_complete, model_registered, stage_transition), fetches the
    proof from Arweave, re-derives the content hash, optionally verifies
    the signature, and checks that the event's previous_hash links to the
    prior event's tx_id (or to a recognized chain root for the first
    event).

    The caller passes ``ArweaveAnchor`` and ``ArioMlflowClient`` instances
    — the plugin doesn't manage caching; that's the caller's choice.

    Args:
        model_name: Registered model name.
        model_version: Specific model version (string).
        anchor: ArweaveAnchor for fetching canonical proofs from the gateway.
        ario_client: ArioMlflowClient for ``lifecycle_for_model`` traversal.
        proof_engine: Optional. Required for signature checks; if ``None``,
            ``signature_valid`` is reported as ``None`` for each event.

    Returns:
        A dict with keys:

        - ``model_name``, ``model_version`` — echo back for caller convenience
        - ``events`` — list of per-event status dicts (see below)
        - ``all_intact`` — True iff every event in the lifecycle has
          ``intact=True``. False if any event is missing on Arweave, has
          a content hash mismatch, fails its signature, or fails its
          chain link.
        - ``verified_at`` — ISO-8601 of when this ran (for the demo's
          "verified at HH:MM:SS" UI badge).
        - ``registration_tx`` — convenience: the registration_tx if found
          (for callers building chain-link checks downstream).
        - ``promotion_tx`` — convenience: the latest promotion_tx if found.

        Each event dict has:

        - ``event_type`` — one of ``"dataset_anchored"``, ``"training_complete"``,
          ``"model_registered"``, ``"stage_transition"``.
        - ``tx_id`` — Arweave tx that anchors this event (from
          ``lifecycle_for_model``).
        - ``previous_tx`` — the tx_id this event chains to (from
          ``lifecycle_for_model``).
        - ``found_on_arweave`` — bool: did ``fetch_proof`` return a record?
        - ``content_hash_valid`` — bool | None: does
          ``hash_data(canonical_json(record)) == record_hash``? ``None``
          if the proof wasn't fetched.
        - ``signature_valid`` — bool | None: from
          ``proof_engine.verify_local`` (when ``proof_engine`` provided).
        - ``link_valid`` — bool | None: does
          ``proof.previous_hash == prior_event.tx_id``? ``None`` for the
          first event (no prior to chain to).
        - ``intact`` — bool: AND of all checks (treating ``None`` as not
          a failure for signature/link, BUT requiring
          ``found_on_arweave AND content_hash_valid is not False``).
    """
    raw_events = ario_client.lifecycle_for_model(model_name, version=model_version)

    events_returned = []
    prior_tx_id = None
    registration_tx = None
    promotion_tx = None

    for event in raw_events:
        tx_id = event.get("tx_id")
        previous_tx = event.get("previous_tx")
        event_type = event.get("event_type", "")

        # Skip events that were never anchored — not "broken," just not yet anchored.
        if not tx_id:
            continue

        proof = anchor.fetch_proof(tx_id)

        if proof is None:
            event_status = {
                "event_type": event_type,
                "tx_id": tx_id,
                "previous_tx": previous_tx,
                "found_on_arweave": False,
                "content_hash_valid": None,
                "signature_valid": None,
                "link_valid": None,
                "intact": False,
            }
        else:
            # Re-derive content hash.
            record = proof.get("record", {}) or {}
            recomputed = hash_data(canonical_json(record))
            content_hash_valid = (recomputed == proof.get("record_hash"))

            # Optional signature check.
            if proof_engine is not None:
                sig_result = proof_engine.verify_local(proof)
                signature_valid = sig_result.get("signature_valid")
            else:
                signature_valid = None

            # Chain-link check: previous_hash in proof must equal prior event's tx_id.
            if prior_tx_id is None:
                link_valid = None
            else:
                link_valid = (proof.get("previous_hash") == prior_tx_id)

            # intact: all present checks must pass.
            intact = (
                content_hash_valid is True
                and (signature_valid is not False)
                and (link_valid is not False)
            )

            event_status = {
                "event_type": event_type,
                "tx_id": tx_id,
                "previous_tx": previous_tx,
                "found_on_arweave": True,
                "content_hash_valid": content_hash_valid,
                "signature_valid": signature_valid,
                "link_valid": link_valid,
                "intact": intact,
            }

        events_returned.append(event_status)

        # Track convenience fields.
        if event_type == "model_registered":
            registration_tx = tx_id
        if event_type == "stage_transition":
            promotion_tx = tx_id  # keeps the latest one if multiple

        # Advance the chain pointer only for found events so the next
        # event's link check compares against the most recent anchored tx.
        prior_tx_id = tx_id

    all_intact = bool(events_returned) and all(e["intact"] for e in events_returned)

    return {
        "model_name": model_name,
        "model_version": model_version,
        "events": events_returned,
        "all_intact": all_intact,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "registration_tx": registration_tx,
        "promotion_tx": promotion_tx,
    }
