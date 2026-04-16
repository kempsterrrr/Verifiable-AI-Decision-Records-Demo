"""Generate self-contained HTML verification reports for MLflow artifact viewer."""

import html


def generate_verification_html(
    proof: dict,
    anchor_result: dict | None,
    artifact_hash: str | None = None,
    artifact_verified: bool | None = None,
) -> str:
    record = proof.get("record", {})
    event_type = record.get("event_type", "unknown")
    run_id = record.get("run_id", record.get("source_run_id", ""))
    timestamp = record.get("timestamp", "")

    status = "anchored" if anchor_result else "signed"
    tx_id = anchor_result["tx_id"] if anchor_result else None
    arweave_url = anchor_result.get("url", "") if anchor_result else ""

    record_hash = proof.get("record_hash", "")
    previous_hash = proof.get("previous_hash", "")
    signature = proof.get("signature", "")
    public_key = proof.get("public_key", "")

    status_color = "#22c55e" if status == "anchored" else "#eab308"
    status_label = "Anchored" if status == "anchored" else "Signed (local)"

    integrity_row = ""
    if artifact_hash:
        if artifact_verified is True:
            integrity_row = _row("Artifact Integrity", _badge("Verified", "#22c55e"))
        elif artifact_verified is False:
            integrity_row = _row("Artifact Integrity", _badge("MISMATCH", "#ef4444"))
        else:
            integrity_row = _row("Artifact Integrity", _badge("Unchecked", "#6b7280"))
        integrity_row += _row("Artifact Hash", _mono(artifact_hash))

    arweave_row = ""
    if tx_id:
        link = f'<a href="{html.escape(arweave_url)}" target="_blank" rel="noopener">{html.escape(tx_id)}</a>'
        arweave_row = _row("Arweave TX", link)
        arweave_row += _row("Gateway URL", f'<a href="{html.escape(arweave_url)}" target="_blank" rel="noopener">{html.escape(arweave_url)}</a>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ar.io Verification Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #fafafa; color: #1a1a1a; padding: 24px; }}
  .container {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font-size: 18px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: #6b7280; font-size: 13px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; margin-bottom: 20px; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #f3f4f6; font-size: 13px; vertical-align: top; }}
  td:first-child {{ width: 160px; color: #6b7280; font-weight: 500; }}
  .mono {{ font-family: "SF Mono", "Fira Code", monospace; font-size: 12px; word-break: break-all; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; color: #fff; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .verify-section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px; margin-top: 8px; }}
  .verify-section h2 {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; }}
  .verify-section pre {{ font-size: 12px; background: #f9fafb; padding: 12px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }}
  .verify-section code {{ font-family: "SF Mono", "Fira Code", monospace; }}
</style>
</head>
<body>
<div class="container">
  <h1>ar.io Verification Report</h1>
  <div class="subtitle">{html.escape(event_type)} &mdash; {html.escape(timestamp)}</div>
  <table>
    {_row("Status", _badge(status_label, status_color))}
    {_row("Event Type", html.escape(event_type))}
    {_row("Run ID", _mono(run_id))}
    {_row("Timestamp", html.escape(timestamp))}
    {arweave_row}
    {integrity_row}
    {_row("Record Hash", _mono(record_hash))}
    {_row("Previous Hash", _mono(previous_hash))}
    {_row("Public Key", _mono(public_key))}
    {_row("Signature", _mono(signature[:64] + "..." if len(signature) > 64 else signature))}
  </table>

  <div class="verify-section">
    <h2>Independent Verification</h2>
    <p style="font-size:13px;color:#6b7280;margin-bottom:10px;">
      To verify this proof independently:
    </p>
    <pre><code># 1. Fetch the proof from Arweave
curl https://turbo-gateway.com/{html.escape(tx_id or 'TX_ID')}/raw

# 2. Verify: re-hash the record field with SHA-256
#    and compare to record_hash

# 3. Verify the Ed25519 signature over:
#    canonical_json({{"record_hash", "previous_hash", "timestamp"}})
#    using the public key above</code></pre>
  </div>
</div>
</body>
</html>"""


def _row(label: str, value: str) -> str:
    return f"<tr><td>{html.escape(label)}</td><td>{value}</td></tr>\n"


def _mono(text: str) -> str:
    return f'<span class="mono">{html.escape(text)}</span>'


def _badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{html.escape(text)}</span>'
