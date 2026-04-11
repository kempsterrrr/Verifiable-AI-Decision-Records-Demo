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
def decision_detail(request: Request, decision_id: str):
    app = request.app
    envelope = app.state.store.get_by_id(decision_id)

    if not envelope:
        return HTMLResponse("<h1>Decision not found</h1>", status_code=404)

    # Run local verification
    verification = app.state.proof_engine.verify_local(envelope)

    # Run external verification if Arweave-anchored
    external_verification = None
    if envelope.get("arweave_tx_id"):
        arweave_data = app.state.anchor.fetch_proof(envelope["arweave_tx_id"])
        if arweave_data:
            arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
            external_verification = {
                "arweave_data_found": True,
                "arweave_record_hash": arweave_hash,
                "arweave_matches_original": arweave_hash == arweave_data.get("record_hash"),
                "local_tampered": not verification["overall"],
            }
        else:
            external_verification = {"arweave_data_found": False}

    return templates.TemplateResponse(
        request,
        "decision_detail.html",
        {
            "envelope": envelope,
            "verification": verification,
            "external_verification": external_verification,
        },
    )
