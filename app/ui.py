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
    verification = app.state.proof_engine.verify_local(envelope)

    # Full verification (on-demand — user-triggered via ?verify=true)
    arweave_verification = None
    ario_verification = None
    if verify and envelope.get("arweave_tx_id"):
        # Fetch from ar.io gateway and compare
        arweave_data = app.state.anchor.fetch_proof(envelope["arweave_tx_id"])
        if arweave_data:
            arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
            arweave_verification = {
                "data_found": True,
                "record_hash": arweave_hash,
                "hash_match": arweave_hash == arweave_data.get("record_hash"),
            }
        else:
            arweave_verification = {"data_found": False}

        # ar.io Verify attestation
        if app.state.ario_verify.enabled:
            ario_result = app.state.ario_verify.submit_verification(envelope["arweave_tx_id"])
            if ario_result:
                ario_verification = app.state.ario_verify._normalize_result(ario_result)

    return templates.TemplateResponse(
        request,
        "decision_detail.html",
        {
            "envelope": envelope,
            "verification": verification,
            "arweave_verification": arweave_verification,
            "ario_verification": ario_verification,
            "verify_requested": verify,
        },
    )
