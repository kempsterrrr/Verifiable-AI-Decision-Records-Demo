from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.decision_record import canonical_json, hash_data

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    app = request.app
    records = app.state.store.list_all()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "records": records,
            "model_info": app.state.model_info,
            "arweave_enabled": app.state.anchor.enabled if app.state.anchor else False,
            "ario_verify_enabled": app.state.ario_verify.enabled if app.state.ario_verify else False,
        },
    )


@router.get("/ui/decisions/{decision_id}", response_class=HTMLResponse)
def decision_detail(request: Request, decision_id: str, verify: bool = False):
    app = request.app
    envelope = app.state.store.get_by_id(decision_id)

    if not envelope:
        return HTMLResponse("<h1>Decision not found</h1>", status_code=404)

    # Local verification (always — instant, no network)
    local = app.state.proof_engine.verify_local(envelope)

    # Full verification (on-demand — user-triggered via ?verify=true)
    if verify and envelope.get("arweave_tx_id"):
        result = {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "hash_valid": local["hash_valid"],
            "signature_valid": local["signature_valid"],
            "permanent_copy_found": False,
            "hash_match": False,
            "attestation_level": None,
            "report_url": None,
            "pdf_url": None,
            "attested_by": None,
            "attested_at": None,
        }

        # Fetch from ar.io gateway and compare
        arweave_data = app.state.anchor.fetch_proof(envelope["arweave_tx_id"])
        if arweave_data:
            arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
            result["permanent_copy_found"] = True
            result["hash_match"] = arweave_hash == arweave_data.get("record_hash")

        # ar.io Verify attestation
        if app.state.ario_verify.enabled:
            ario_result = app.state.ario_verify.submit_verification(envelope["arweave_tx_id"])
            if ario_result:
                normalized = app.state.ario_verify._normalize_result(ario_result)
                result["attestation_level"] = normalized.get("level")
                result["report_url"] = normalized.get("report_url")
                result["pdf_url"] = normalized.get("pdf_url")
                result["attested_by"] = normalized.get("attested_by")
                result["attested_at"] = normalized.get("attested_at")

        # Persist results on the envelope
        envelope["last_verification"] = result
        app.state.store.update(decision_id, envelope)

    return templates.TemplateResponse(
        request,
        "decision_detail.html",
        {
            "envelope": envelope,
            "local_verification": local,
        },
    )
