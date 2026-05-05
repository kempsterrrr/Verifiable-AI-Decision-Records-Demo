# Phase 3 — Demo UX Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the Phase 3 design (`docs/superpowers/specs/2026-04-30-phase3-demo-ux-polish-design.md`) — a UI polish pass on six demo templates, plus two new tamper backend endpoints — to bring the demo to sales-call resolution.

**Architecture:** Three-row verification card structure (`Proof Found` / `{event} Record Matches` / `Signature Confirmed`) replaces the old four-row layout across every verify surface. Two collapsibles (`Click to tamper`, `How verification works`) keep the demo's secondary surfaces compact by default. Two tamper backend endpoints mutate real MLflow state with auto-revert. No plugin API changes — only UI labels and presentation update.

**Tech Stack:** Jinja2 templates · FastAPI (`app/main.py`, `app/ui.py`) · MLflow tracking client · pytest · existing CSS variable system in `templates/base.html`

**Reference materials:**
- Design spec: `docs/superpowers/specs/2026-04-30-phase3-demo-ux-polish-design.md`
- Visual mockups: `.superpowers/brainstorm/34033-1777551244/content/`
  - `decision-detail-full-page-v11.html` — canonical decision detail
  - `index-page-v1.html`
  - `run-detail-page-v1.html`
  - `model-chain-page-v1.html`
  - `model-registry-page-v1.html`
  - `who-this-is-for-page-v1.html`

When this plan and the spec disagree about details, the spec wins. When the spec and a mockup disagree, the mockup wins for visual structure; the spec wins for vocabulary and behavior.

---

## File responsibility map

Files created or modified, grouped by responsibility:

**New files:**
- `tests/test_tamper_endpoints.py` — pytest suite for tamper backend
- `app/tamper.py` — tamper state management (snapshots, auto-revert) + endpoint handlers

**Templates modified (every page in scope):**
- `templates/base.html` — new shared CSS classes (collapsible, tamper-section, mini-verify, audit grid, section-label, badge-red, summary global rules)
- `templates/decision_detail.html` — full structural restructure (Decision card, audit grid, two collapsibles)
- `templates/index.html` — vocabulary sweep + button rename + Records → Recent decisions
- `templates/run_detail.html` — full restructure mirroring decision_detail (Training summary card, audit grid, three collapsibles)
- `templates/model_chain.html` — mini verify cards in chain nodes + tamper section + how-verification-works
- `templates/model_registry.html` — vocabulary sweep + run ID linkability + Arweave→ar.io
- `templates/who_this_is_for.html` — wording sweep + 4-col grid

**Backend modified:**
- `app/main.py` — register tamper routes, remove legacy `/api/export/{decision_id}` if confirmed unused
- `app/ui.py` — no logic changes, only template-side; verify references unchanged

**Docs modified:**
- `README.md` — vocabulary sweep
- `ario_mlflow/README.md` — vocabulary sweep
- `ario_mlflow/cli.py` — printed labels match new vocabulary

**Files deleted (or content removed):**
- Legacy `Proof Layer` panel from `run_detail.html` (block-level deletion)
- `/api/export/{decision_id}` route from `app/main.py` (if no other callers)

---

## Phase A — Foundation

### Task A1: Create the Phase 3 branch

**Files:** none modified

- [ ] **Step 1: Confirm we're on main and clean**

```bash
git status
git branch --show-current
```

Expected: working tree clean, branch is `main`.

- [ ] **Step 2: Create and switch to the phase 3 branch**

```bash
git checkout -b phase3/demo-ux-polish
git branch --show-current
```

Expected: `phase3/demo-ux-polish`.

- [ ] **Step 3: Confirm pytest passes from main as a baseline**

```bash
pytest -q
```

Expected: all tests pass. Note the count for comparison after later changes.

---

### Task A2: Add new CSS utility classes to base.html

**Files:**
- Modify: `templates/base.html`

The spec's section 6 enumerates every new class. Add them inside the existing `<style>` block in `base.html`. Group them after the existing badge/topbar styles, before the `@media` blocks.

- [ ] **Step 1: Read `templates/base.html` to find the right insertion point**

Locate the end of the badge styles (look for `.badge-dot-pulse` rule) and the start of the `@media` queries. New CSS lands between them.

- [ ] **Step 2: Add the badge-red variant**

Find `.badge-gray { background: var(--gray-bg); color: var(--muted); }` in `base.html` and add `.badge-red` immediately after:

```css
.badge-red { background: var(--red-bg); color: var(--red); }
.badge-red .badge-dot { background: var(--red); }
```

- [ ] **Step 3: Add global `<summary>` reset rules**

Add near the top of the `<style>` block, right after the `* { ... }` reset:

```css
summary { cursor: pointer; }
summary::-webkit-details-marker { display: none; }
summary::marker { content: ""; }
```

- [ ] **Step 4: Add `.section-label` styles (page-narrative spine)**

Add to the shared style block:

```css
.section-label {
  font-family: 'Besley', serif;
  font-weight: 800;
  font-size: 1rem;
  color: var(--black);
  margin: 2.25rem 0 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.section-label::before {
  content: "";
  height: 1px;
  background: var(--border);
  width: 1.5rem;
}
.section-label::after {
  content: "";
  height: 1px;
  background: var(--border);
  flex: 1;
}
```

- [ ] **Step 5: Add `.collapsible` (purple-themed) styles**

```css
.collapsible { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; transition: border-color 0.2s; }
.collapsible[open] { border-color: var(--lavender); }
.collapsible summary {
  padding: 0.85rem 1rem;
  background: var(--surface);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  transition: background 0.2s;
}
.collapsible[open] summary { background: var(--lavender); border-bottom: 1px solid var(--lavender); }
.collapsible-summary-icon {
  width: 20px; height: 20px; border-radius: 50%;
  background: var(--white); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.2s, border-color 0.2s;
  flex-shrink: 0;
  color: var(--primary);
}
.collapsible[open] .collapsible-summary-icon { transform: rotate(90deg); border-color: var(--primary); }
.collapsible-summary-icon svg { width: 10px; height: 10px; }
.collapsible-summary-title { font-size: 0.85rem; font-weight: 700; color: var(--black); }
.collapsible[open] .collapsible-summary-title { color: var(--primary); }
.collapsible-summary-hint { font-size: 0.78rem; color: var(--muted); margin-left: auto; padding-right: 0.2rem; }
.collapsible[open] .collapsible-summary-hint { display: none; }
.collapsible-body { padding: 1.25rem; background: white; }
```

- [ ] **Step 6: Add `.tamper-section` (yellow-themed) styles**

```css
.tamper-section { border: 1px solid var(--yellow); border-radius: 8px; overflow: hidden; }
.tamper-header {
  padding: 0.75rem 1rem;
  background: var(--yellow-bg);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--yellow);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  cursor: pointer;
  list-style: none;
}
.tamper-header-inner { display: flex; align-items: center; gap: 0.6rem; }
.tamper-section[open] .tamper-header { border-bottom: 1px solid var(--yellow); }
.tamper-summary-icon {
  width: 20px; height: 20px; border-radius: 50%;
  background: var(--white); border: 1px solid var(--yellow);
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.2s;
  flex-shrink: 0;
  color: var(--yellow);
}
.tamper-section[open] .tamper-summary-icon { transform: rotate(90deg); }
.tamper-summary-icon svg { width: 10px; height: 10px; }
.tamper-summary-title { flex: 1; }
.tamper-summary-hint { font-size: 0.7rem; font-weight: 400; text-transform: none; letter-spacing: normal; color: var(--yellow); }
.tamper-section[open] .tamper-summary-hint { display: none; }
.tamper-body { padding: 1rem; background: var(--white); }
.tamper-intro { font-size: 0.82rem; line-height: 1.5; margin-bottom: 0.85rem; }
.tamper-intro em { color: var(--yellow); font-style: normal; font-weight: 600; }
.tamper-row { display: flex; align-items: flex-start; padding: 0.7rem 0; gap: 1rem; }
.tamper-row + .tamper-row { border-top: 1px solid var(--border); }
.tamper-row-info { flex: 1; }
.tamper-row-title { font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; }
.tamper-row-title .num { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted); background: var(--surface); padding: 0.1rem 0.4rem; border-radius: 3px; }
.tamper-row-desc { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; line-height: 1.5; }
.tamper-row-desc strong { color: var(--black); font-weight: 600; }
.tamper-row code { font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; background: var(--surface); padding: 0.05rem 0.3rem; border-radius: 3px; color: var(--primary); }
.tamper-headline { display: inline-block; margin-top: 0.25rem; font-size: 0.7rem; color: var(--primary); background: var(--lavender); padding: 0.15rem 0.5rem; border-radius: 3px; font-weight: 600; }
.btn-tamper {
  background: var(--white);
  color: var(--yellow);
  border: 1px solid var(--yellow);
  padding: 0.4rem 0.85rem;
  font-size: 0.78rem;
  font-weight: 600;
  border-radius: 4px;
  cursor: pointer;
  flex-shrink: 0;
}
.btn-tamper:hover { background: var(--yellow-bg); }
.btn-tamper:disabled { opacity: 0.5; cursor: not-allowed; }
.tamper-footer { display: flex; flex-direction: column; align-items: stretch; gap: 0.4rem; padding: 0.75rem 0 0.1rem; border-top: 1px solid var(--border); margin-top: 0.4rem; }
.tamper-footer-text { font-size: 0.78rem; color: var(--muted); flex: 1; }
.btn-reset { padding: 0.35rem 0.8rem; background: var(--surface); border: 1px solid var(--border); color: var(--black); border-radius: 4px; font-size: 0.78rem; font-weight: 500; cursor: pointer; }
.btn-reset:hover { background: var(--border); }
```

- [ ] **Step 7: Add `.grid-2-audit` (always-visible 2x2 audit cards)**

```css
.grid-2-audit { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 700px) { .grid-2-audit { grid-template-columns: 1fr; } }
```

- [ ] **Step 8: Add `.mini-verify` styles (chain page only)**

```css
.mini-verify { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 0.55rem 0.75rem; }
.mini-verify-row { display: flex; justify-content: space-between; align-items: center; padding: 0.32rem 0; font-size: 0.78rem; }
.mini-verify-row + .mini-verify-row { border-top: 1px solid var(--border); }
.mini-verify-meta { font-size: 0.7rem; color: var(--muted); padding-top: 0.45rem; margin-top: 0.3rem; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 0.15rem; }
.mini-verify-meta strong { color: var(--black); }
```

- [ ] **Step 9: Add split proof viewer styles**

```css
/* Split proof viewer (used inside .collapsible) */
.viewer-intro { font-size: 0.86rem; line-height: 1.55; margin-bottom: 1.25rem; }
.viewer-intro strong { color: var(--primary); }
.proof-split { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: stretch; }
@media (max-width: 700px) { .proof-split { grid-template-columns: 1fr; } }
.panel { background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
.panel-header { padding: 0.75rem 0.85rem; background: white; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.panel-title-row { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
.panel-title { font-size: 0.8rem; font-weight: 700; color: var(--black); }
.panel-meta { font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; line-height: 1.45; }
.panel-source-pill {
  font-size: 0.62rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 0.18rem 0.5rem; border-radius: 3px;
  white-space: nowrap;
}
.pill-arweave { background: #fff3d6; color: #8a5a00; border: 1px solid #ffd591; }
.pill-mlflow { background: #e8edff; color: #2a3d80; border: 1px solid #b8c4f0; }
.panel-body { padding: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; line-height: 1.55; max-height: 320px; overflow-y: auto; flex: 1; min-height: 0; }
.json-block { padding: 0.85rem; white-space: pre-wrap; word-break: break-all; }
.panel-footer {
  padding: 0.55rem 0.85rem;
  background: white;
  border-top: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.72rem; color: var(--muted);
  flex-shrink: 0;
  margin-top: auto;
}
.panel-footer a { color: var(--primary); font-weight: 500; text-decoration: none; }
.json-key { color: var(--primary); }
.json-string { color: #1a7f37; }
.payload-hash-line {
  display: block;
  background: #fff3d6;
  border-left: 3px solid var(--yellow);
  padding: 0.15rem 0.4rem 0.15rem 0.55rem;
  margin: 0.1rem -0.85rem 0.1rem -0.85rem;
  font-weight: 600;
}
.payload-hash-line .json-key { color: #8a5a00; }
.payload-hash-line .json-string { color: #6a4400; }
.equality {
  margin-top: 1rem;
  padding: 1.1rem 1.25rem;
  background: linear-gradient(90deg, #e8edff 0%, var(--lavender) 50%, #fff3d6 100%);
  border-radius: 8px;
  display: flex; align-items: center; gap: 1rem;
  flex-wrap: wrap;
}
.equality-prefix { font-size: 0.8rem; font-weight: 600; }
.equality-formula { display: flex; align-items: center; gap: 0.6rem; font-size: 0.85rem; flex: 1; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; }
.eq-token { padding: 0.32rem 0.6rem; border-radius: 5px; font-weight: 600; }
.eq-right { background: #fff3d6; color: #8a5a00; border: 1px solid #ffd591; }
.eq-left { background: #e8edff; color: #2a3d80; border: 1px solid #b8c4f0; }
.eq-op { color: var(--primary); font-weight: 700; font-size: 1rem; }
.eq-result { margin-left: auto; display: inline-flex; align-items: center; gap: 0.4rem; background: var(--green-bg); color: var(--green); padding: 0.32rem 0.7rem; border-radius: 5px; font-weight: 700; font-size: 0.8rem; border: 1px solid var(--green); }
.eq-result-fail { background: var(--red-bg); color: var(--red); border-color: var(--red); }
.legend-row { display: flex; gap: 1.25rem; padding: 0.85rem 1rem; background: var(--code-bg); border-radius: 6px; font-size: 0.8rem; color: var(--muted); margin-top: 1rem; line-height: 1.5; border: 1px solid var(--border); }
.legend-row > div { flex: 1; }
.legend-row strong { color: var(--black); }
@media (max-width: 700px) { .legend-row { flex-direction: column; } }
```

- [ ] **Step 10: Verify pytest still passes**

```bash
pytest -q
```

Expected: same count as baseline. CSS additions don't run through Python; tests should be unaffected.

- [ ] **Step 11: Commit**

```bash
git add templates/base.html
git commit -m "Phase 3 — add shared CSS utilities (collapsible, tamper, audit grid)"
```

---

## Phase B — Vocabulary sweep across templates

Each template gets a focused find-and-replace pass. We do all six templates' vocabulary changes before any structural work — this isolates risk: vocabulary changes don't break anything; if pytest passes after each commit, we know the rename was safe.

### Task B1: index.html vocabulary sweep

**Files:**
- Modify: `templates/index.html`

- [ ] **Step 1: Read `templates/index.html` to locate the strings**

Quickly identify the line numbers for each rename so the Edit calls have correct context.

- [ ] **Step 2: Update page subtitle**

```
old: Every credit-decision below has a signed proof anchored on Arweave via ar.io — verifiable by any third party, without trusting this app.
new: Every decision below carries a signed proof anchored to ar.io. Anyone can verify the live record matches what was anchored — without trusting this app.
```

- [ ] **Step 3: Update stats card labels**

```
old: <div class="stat-label">Verified</div>
new: <div class="stat-label">Verified</div>           (unchanged — keep)

old: <div class="stat-label">Anchored</div>
new: <div class="stat-label">Pending verification</div>

old: <div class="stat-label">Failed</div>
new: <div class="stat-label">Tampered</div>

old: <div class="stat-label">Local Only</div>
new: <div class="stat-label">Not anchored</div>
```

- [ ] **Step 4: Update Provenance card status badges**

In the provenance-card block, find each `{% if training_status == 'anchored' %}` / `{% if registration_status == 'anchored' %}` branch and update the badge text:

```
old: <span class="badge badge-yellow"><span class="badge-dot"></span> Anchored</span>
new: <span class="badge badge-yellow"><span class="badge-dot"></span> Pending verification</span>

old: <span class="badge badge-gray"><span class="badge-dot"></span> Local</span>
new: <span class="badge badge-gray"><span class="badge-dot"></span> Not anchored</span>
```

(Apply both for `training_status` and `registration_status` blocks.)

- [ ] **Step 5: Update prediction form button**

```
old: <button type="submit" class="btn btn-primary" data-loading-text="Predicting...">Predict &amp; Record</button>
new: <button type="submit" class="btn btn-primary" data-loading-text="Predicting...">Make a decision</button>
```

- [ ] **Step 6: Update Records section heading**

```
old: <h2>Records</h2>
new: <h2>Recent decisions</h2>
```

- [ ] **Step 7: Update table column headers**

```
old: <th>Prediction</th>
new: <th>Result</th>

old: <th>Arweave TX</th>
new: <th>ar.io Anchor</th>
```

- [ ] **Step 8: Update Result column rendering (Prediction → Result chip)**

Find:

```html
<td>{{ env.record.prediction['class'] if env.record.prediction is defined else 'N/A' }}</td>
```

Replace with:

```html
<td>
  {% if env.record.prediction is defined %}
    {% if env.record.prediction['class'] == 'approve' %}
      <span class="badge badge-green"><span class="badge-dot"></span> Approved</span>
    {% elif env.record.prediction['class'] == 'deny' %}
      <span class="badge badge-red"><span class="badge-dot"></span> Denied</span>
    {% else %}
      {{ env.record.prediction['class'] }}
    {% endif %}
  {% else %}
    N/A
  {% endif %}
</td>
```

- [ ] **Step 9: Update status badges in records table rows**

Find each branch in the row's status `<td>` and update:

```
old: <span class="badge badge-yellow"><span class="badge-dot"></span> Anchored</span>
new: <span class="badge badge-yellow"><span class="badge-dot"></span> Pending verification</span>

old: <span class="badge badge-red"><span class="badge-dot"></span> Failed</span>
new: <span class="badge badge-red"><span class="badge-dot"></span> Tampered</span>

old: <span class="badge badge-gray"><span class="badge-dot"></span> Local Only</span>
new: <span class="badge badge-gray"><span class="badge-dot"></span> Not anchored</span>
```

(`Anchoring` and `Verified` stay unchanged.)

- [ ] **Step 10: Verify locally — boot the app and confirm the dashboard renders**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
curl -s http://localhost:8000/ | grep -E "(Pending verification|Tampered|Not anchored|Recent decisions|Make a decision)" | head -10
kill %1 2>/dev/null
```

Expected: matches showing the new labels are rendered.

- [ ] **Step 11: Run pytest**

```bash
pytest -q
```

Expected: same count as baseline.

- [ ] **Step 12: Commit**

```bash
git add templates/index.html
git commit -m "Phase 3 — index.html vocabulary sweep + Result chip"
```

---

### Task B2: model_registry.html vocabulary sweep

**Files:**
- Modify: `templates/model_registry.html`

- [ ] **Step 1: Update context banner body**

```
old: Train a model below, make a prediction, then try to break the audit trail — either by changing the local record directly, or by imagining that MLflow runs get deleted, models get silently retrained, or institutional memory is lost. Because every proof is anchored on Arweave via ar.io, a third party can verify what actually happened without trusting this app or any records it stores locally.
new: Train a model below, make a prediction, then try to break the verifiable record — either by tampering with MLflow data directly, or by imagining the runs get deleted, models get silently retrained, or institutional memory is lost. Because every proof is anchored to ar.io, a third party can verify what actually happened without trusting this app or any records it stores locally.
```

- [ ] **Step 2: Update train hero subtitle**

```
old: <p>Train, register, and anchor provenance to Arweave</p>
new: <p>Train, register, and anchor provenance to ar.io</p>
```

- [ ] **Step 3: Update progress step text in scripts**

Inside the `<script>` block, find the train-step div labels:

```
old: <div class="train-step" id="step-anchor-train"><span class="step-icon"></span> Anchoring training proof to Arweave...</div>
new: <div class="train-step" id="step-anchor-train"><span class="step-icon"></span> Anchoring training proof to ar.io...</div>

old: <div class="train-step" id="step-anchor-reg"><span class="step-icon"></span> Anchoring registration proof to Arweave...</div>
new: <div class="train-step" id="step-anchor-reg"><span class="step-icon"></span> Anchoring registration proof to ar.io...</div>
```

- [ ] **Step 4: Update result text after training**

Find the result line in the script:

```
old: result.innerHTML = 'Model v' + data.model_version + ' trained (accuracy: ' + data.accuracy.toFixed(4) + ') and anchored to Arweave. Redirecting to model lineage...';
new: result.innerHTML = 'Model v' + data.model_version + ' trained (accuracy: ' + data.accuracy.toFixed(4) + ') and anchored to ar.io. Redirecting to model lineage...';
```

- [ ] **Step 5: Update versions table status badges**

In the Training and Registration `<td>` blocks, replace the status text:

```
old: <span class="badge badge-yellow"><span class="badge-dot"></span> Anchored</span>
new: <span class="badge badge-yellow"><span class="badge-dot"></span> Pending verification</span>

old: <span class="badge badge-gray"><span class="badge-dot"></span> Local</span>
new: <span class="badge badge-gray"><span class="badge-dot"></span> Not anchored</span>
```

(Apply twice — once for the Training column, once for the Registration column.)

- [ ] **Step 6: Make Run IDs clickable**

Find:

```html
<td><span class="mono">{{ v.run_id[:12] }}</span></td>
```

Replace with:

```html
<td><a class="mono" href="/ui/runs/{{ v.run_id }}" style="word-break: break-all; font-size: 0.74rem; text-decoration: none;">{{ v.run_id }}</a></td>
```

(Removes `[:12]` truncation; full ID renders with break-all wrapping.)

Add the hover style if not present in `base.html`:

```css
a.mono { text-decoration: none; }
a.mono:hover { text-decoration: underline; }
```

(Add to `base.html` if missing — check first.)

- [ ] **Step 7: Run pytest**

```bash
pytest -q
```

Expected: same count.

- [ ] **Step 8: Commit**

```bash
git add templates/model_registry.html templates/base.html
git commit -m "Phase 3 — model_registry vocabulary sweep + clickable run IDs"
```

---

### Task B3: who_this_is_for.html wording sweep

**Files:**
- Modify: `templates/who_this_is_for.html`

- [ ] **Step 1: Update intro paragraph**

```
old: This is an early-stage demo of <code>ario-mlflow</code>, a plugin that anchors signed proofs of training runs, model registrations, and every individual inference to Arweave via ar.io. Four kinds of people come to this demo for four different reasons — the cards below are each of their doorways in.
new: This is an early-stage demo of <code>ario-mlflow</code>, a plugin that anchors signed proofs of training runs, model registrations, and every individual inference to ar.io. Four kinds of people come to this demo for four different reasons — the cards below are each of their doorways in.
```

- [ ] **Step 2: Update ML engineer answer**

```
old: Call <code>ario_mlflow.anchor()</code> once inside your training loop and every run mints a signed proof on Arweave. Even if your MLflow server is gone, any third party can fetch the proof and verify exactly what ran.
new: Call <code>ario_mlflow.anchor()</code> once inside your training loop and every run mints a signed proof anchored to ar.io. Even if your MLflow server is gone, any third party can fetch the proof and verify exactly what ran.
```

- [ ] **Step 3: Update auditor answer**

```
old: Every anchored proof is independently verifiable — re-hash the canonical record, check the Ed25519 signature, fetch the permanent copy from Arweave. No client access required. A standalone verifier portal is on the roadmap.
new: Every anchored proof is independently verifiable — re-hash the canonical record, verify the signature, fetch the permanent copy from ar.io. No client access required. A standalone verifier portal is on the roadmap.
```

- [ ] **Step 4: Update OS publisher answer**

```
old: The proof format (SHA-256 artifact hash + Ed25519 signature anchored on Arweave) is public-model-publishing friendly. A dedicated <code>huggingface-cli</code> plugin and a model-card badge are on the roadmap — the underlying primitive is usable today.
new: The proof format (SHA-256 artifact hash + signed commitment anchored to ar.io) is public-model-publishing friendly. A dedicated <code>huggingface-cli</code> plugin and a model-card badge are on the roadmap — the underlying primitive is usable today.
```

- [ ] **Step 5: Switch persona grid to 4-column layout**

In the `<style>` block at the top of the template, find:

```css
.persona-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 2rem; }
```

(or similar — actual style may differ; locate by class name)

Replace with:

```css
.persona-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.1rem; margin-bottom: 2rem; }
@media (max-width: 1200px) { .persona-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 700px) { .persona-grid { grid-template-columns: 1fr; } }
```

- [ ] **Step 6: Widen the page main container**

Find the `.main` width override in this template's style block. If `.main` is overridden here, raise its `max-width` to `1500px`. If not overridden, add a `{% block max_width %}1500px{% endblock %}` above the content block (per `base.html`'s `{% block max_width %}` pattern).

Check the base template first:

```bash
grep -n "max_width" templates/base.html
```

If `base.html` defines `{% block max_width %}1400px{% endblock %}` (or similar), override with `{% block max_width %}1500px{% endblock %}` in `who_this_is_for.html`.

- [ ] **Step 7: Run pytest**

```bash
pytest -q
```

- [ ] **Step 8: Commit**

```bash
git add templates/who_this_is_for.html
git commit -m "Phase 3 — who_this_is_for wording sweep + 4-col persona grid"
```

---

## Phase C — Tamper backend

### Task C1: Write tamper test scaffolding

**Files:**
- Create: `tests/test_tamper_endpoints.py`

- [ ] **Step 1: Create the test file with skeletons**

Write `tests/test_tamper_endpoints.py`:

```python
"""Tamper endpoint tests.

Each tamper mutates real MLflow state and the verifier should catch it.
Reset restores the original state. Auto-revert (background timer) is
tested separately via direct call to the revert helper, not via real
sleep.
"""
import os
import json
import tempfile
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Boot the demo app with an isolated MLflow + records directory.

    Each test gets a fresh tracking store and a fresh model trained
    automatically by the lifespan handler (existing behavior).
    """
    monkeypatch.setenv("VAIDR_RECORDS_FILE", str(tmp_path / "records.json"))
    monkeypatch.setenv("VAIDR_LIFECYCLE_FILE", str(tmp_path / "lifecycle.json"))
    monkeypatch.setenv("VAIDR_MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    # Disable Arweave so anchoring doesn't try to hit the network.
    monkeypatch.setenv("VAIDR_ARWEAVE_WALLET_PATH", "")
    from app.main import app
    return TestClient(app)


def test_tamper_saved_record_returns_ok(client):
    """POST /tamper/saved/decision/{event_id} writes garbage to payload.json
    and returns success."""
    # Arrange: make a prediction so we have a decision to tamper.
    response = client.post("/predict-form", data={
        "annual_income": "78000",
        "credit_utilization": "0.18",
        "debt_to_income_ratio": "0.22",
        "months_employed": "72",
        "credit_score": "745",
    }, follow_redirects=False)
    # Find the resulting decision_id (varies by prediction path).
    decisions = client.get("/decisions").json()
    assert len(decisions) >= 1, "expected at least one decision after predict"
    decision_id = decisions[0]["record"]["decision_id"]

    # Act: tamper.
    r = client.post(f"/tamper/saved/decision/{decision_id}")

    # Assert.
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tampered"] is True
    assert body["kind"] == "saved"


def test_tamper_live_data_returns_ok(client):
    """POST /tamper/live/decision/{event_id} mutates the trace's
    ario.payload_json tag and returns success."""
    # Arrange: same as above.
    client.post("/predict-form", data={
        "annual_income": "78000", "credit_utilization": "0.18",
        "debt_to_income_ratio": "0.22", "months_employed": "72",
        "credit_score": "745",
    }, follow_redirects=False)
    decision_id = client.get("/decisions").json()[0]["record"]["decision_id"]

    # Act.
    r = client.post(f"/tamper/live/decision/{decision_id}")

    # Assert.
    assert r.status_code == 200, r.text
    assert r.json()["tampered"] is True


def test_tamper_reset_restores_state(client):
    """After tamper + reset, the verification should pass again."""
    # Arrange.
    client.post("/predict-form", data={
        "annual_income": "78000", "credit_utilization": "0.18",
        "debt_to_income_ratio": "0.22", "months_employed": "72",
        "credit_score": "745",
    }, follow_redirects=False)
    decision_id = client.get("/decisions").json()[0]["record"]["decision_id"]

    # Tamper.
    client.post(f"/tamper/saved/decision/{decision_id}")

    # Act: reset.
    r = client.post(f"/tamper/reset/decision/{decision_id}")

    # Assert.
    assert r.status_code == 200, r.text
    assert r.json()["reset"] is True


def test_tamper_unknown_event_id_returns_404(client):
    """Tampering an event that doesn't exist returns 404."""
    r = client.post("/tamper/saved/decision/no-such-id")
    assert r.status_code == 404
```

- [ ] **Step 2: Run the test file — expect import / endpoint failures**

```bash
pytest tests/test_tamper_endpoints.py -v
```

Expected: ALL FAIL with 404 or AttributeError because the endpoints don't exist yet. This is TDD — failing tests confirm the test scaffolding is correct.

- [ ] **Step 3: Commit (failing tests)**

```bash
git add tests/test_tamper_endpoints.py
git commit -m "Phase 3 — add tamper endpoint test scaffolding (failing)"
```

---

### Task C2: Implement tamper state manager

**Files:**
- Create: `app/tamper.py`

- [ ] **Step 1: Create `app/tamper.py`**

```python
"""Tamper state management for the demo's tamper buttons.

Each tamper mutates real MLflow state so the plugin's verifier catches
it organically. Pre-tamper snapshots live in-memory; reset writes them
back. Auto-revert is a background asyncio task that calls reset after
a short window (default 60s).

This module is demo-only — production deployments should never expose
these endpoints.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


TAMPER_TTL_SECONDS = int(os.environ.get("VAIDR_TAMPER_TTL_SECONDS", "60"))


@dataclass
class TamperSnapshot:
    """Pre-tamper state captured so reset can restore it."""
    event_type: Literal["decision", "training", "registration"]
    event_id: str
    kind: Literal["saved", "live"]
    saved_artifact_bytes: Optional[bytes] = None  # for kind="saved"
    live_field_name: Optional[str] = None         # for kind="live"
    live_field_old_value: Optional[str] = None    # for kind="live"


# In-memory store: keyed by (event_type, event_id, kind). Allows two
# concurrent tampers per event (one saved + one live).
_snapshots: dict[tuple[str, str, str], TamperSnapshot] = {}
_lock = threading.Lock()


def _resolve_run_id(event_type: str, event_id: str, lifecycle_store, record_store) -> str:
    """Look up the MLflow run_id for a given event."""
    if event_type == "decision":
        envelope = record_store.get_by_id(event_id)
        if envelope is None:
            raise KeyError(f"decision {event_id} not found")
        return envelope["record"]["mlflow_run_id"]
    elif event_type == "training":
        envelope = lifecycle_store.get_by_event_id(event_id)
        if envelope is None:
            # event_id may actually be a run_id directly on this page
            return event_id
        return envelope["record"]["run_id"]
    elif event_type == "registration":
        envelope = lifecycle_store.get_by_event_id(event_id)
        if envelope is None:
            raise KeyError(f"registration {event_id} not found")
        return envelope["record"]["source_run_id"]
    raise ValueError(f"unknown event_type: {event_type}")


def _payload_artifact_path(event_type: str, event_id: str) -> str:
    """The MLflow artifact path for the canonical bytes per event type."""
    if event_type == "decision":
        return f"ario/predictions/{event_id}/payload.json"
    elif event_type == "training":
        return "ario/payload.json"
    elif event_type == "registration":
        return "ario/registration_payload.json"
    raise ValueError(f"unknown event_type: {event_type}")


def tamper_saved(event_type, event_id, *, lifecycle_store, record_store, tracking_uri):
    """Overwrite the canonical bytes artifact in MLflow with garbage.

    Snapshots the original bytes so reset can restore. Idempotent: if
    already tampered, returns the existing snapshot.
    """
    key = (event_type, event_id, "saved")
    with _lock:
        if key in _snapshots:
            return _snapshots[key]

        run_id = _resolve_run_id(event_type, event_id, lifecycle_store, record_store)
        artifact_path = _payload_artifact_path(event_type, event_id)

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Read current bytes.
        with mlflow.utils.file_utils.TempDir() as tmp:
            try:
                local_path = client.download_artifacts(run_id, artifact_path, tmp.path())
                with open(local_path, "rb") as f:
                    original_bytes = f.read()
            except Exception as e:
                raise KeyError(f"could not download {artifact_path} for run {run_id}: {e}")

            # Write garbage.
            tampered_path = os.path.join(tmp.path(), "tampered.json")
            with open(tampered_path, "wb") as f:
                f.write(b'{"tampered": true, "this is not the original payload": "garbage"}')

            # Re-upload to overwrite.
            artifact_dir = os.path.dirname(artifact_path)
            artifact_name = os.path.basename(artifact_path)
            renamed = os.path.join(tmp.path(), artifact_name)
            os.rename(tampered_path, renamed)
            client.log_artifact(run_id, renamed, artifact_path=artifact_dir)

        snapshot = TamperSnapshot(
            event_type=event_type, event_id=event_id, kind="saved",
            saved_artifact_bytes=original_bytes,
        )
        _snapshots[key] = snapshot
        logger.info(f"Tamper SAVED applied: {event_type}/{event_id}")
        return snapshot


def tamper_live(event_type, event_id, *, lifecycle_store, record_store, tracking_uri):
    """Mutate a live MLflow field per event type.

    - decision: overwrite the trace's ario.payload_json tag.
    - training: overwrite logged accuracy metric to 0.999.
    - registration: overwrite the model version's source_run_id tag.
    """
    key = (event_type, event_id, "live")
    with _lock:
        if key in _snapshots:
            return _snapshots[key]

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        snapshot: TamperSnapshot

        if event_type == "decision":
            envelope = record_store.get_by_id(event_id)
            if envelope is None:
                raise KeyError(f"decision {event_id} not found")
            trace_id = envelope["record"].get("trace_id")
            if not trace_id:
                raise KeyError(f"decision {event_id} has no trace_id")
            # Read current tag, then overwrite.
            try:
                trace = client.get_trace(trace_id)
                old = (trace.info.tags or {}).get("ario.payload_json", "")
            except Exception:
                old = ""
            client.set_trace_tag(trace_id, "ario.payload_json",
                                 '{"tampered": "this is no longer the canonical bytes"}')
            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"trace_tag:{trace_id}:ario.payload_json",
                live_field_old_value=old,
            )

        elif event_type == "training":
            run_id = _resolve_run_id(event_type, event_id, lifecycle_store, record_store)
            run = client.get_run(run_id)
            old = str(run.data.metrics.get("accuracy", "0.0"))
            # Mutate metric.
            client.log_metric(run_id, "accuracy", 0.999)
            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"run_metric:{run_id}:accuracy",
                live_field_old_value=old,
            )

        elif event_type == "registration":
            envelope = lifecycle_store.get_by_event_id(event_id)
            if envelope is None:
                raise KeyError(f"registration {event_id} not found")
            model_name = envelope["record"]["model_name"]
            model_version = envelope["record"]["model_version"]
            mv = client.get_model_version(model_name, model_version)
            tags = mv.tags or {}
            old = tags.get("source_run_id", "")
            # Mutate tag.
            client.set_model_version_tag(model_name, model_version,
                                          "source_run_id", "tampered-fake-run-id")
            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"mv_tag:{model_name}:{model_version}:source_run_id",
                live_field_old_value=old,
            )
        else:
            raise ValueError(f"unknown event_type: {event_type}")

        _snapshots[key] = snapshot
        logger.info(f"Tamper LIVE applied: {event_type}/{event_id}")
        return snapshot


def reset(event_type, event_id, *, lifecycle_store, record_store, tracking_uri):
    """Restore both saved and live state for an event from snapshots.

    Returns the number of tampers reverted.
    """
    reverted = 0
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    with _lock:
        for kind in ("saved", "live"):
            key = (event_type, event_id, kind)
            snap = _snapshots.pop(key, None)
            if snap is None:
                continue

            try:
                if snap.kind == "saved" and snap.saved_artifact_bytes is not None:
                    run_id = _resolve_run_id(event_type, event_id, lifecycle_store, record_store)
                    artifact_path = _payload_artifact_path(event_type, event_id)
                    with mlflow.utils.file_utils.TempDir() as tmp:
                        renamed = os.path.join(tmp.path(), os.path.basename(artifact_path))
                        with open(renamed, "wb") as f:
                            f.write(snap.saved_artifact_bytes)
                        client.log_artifact(run_id, renamed,
                                            artifact_path=os.path.dirname(artifact_path))
                elif snap.kind == "live":
                    # Parse the field name to know what to restore.
                    parts = snap.live_field_name.split(":", 3)
                    kind_prefix = parts[0]
                    if kind_prefix == "trace_tag":
                        _, trace_id, tag = parts
                        client.set_trace_tag(trace_id, tag, snap.live_field_old_value or "")
                    elif kind_prefix == "run_metric":
                        _, run_id, metric = parts
                        client.log_metric(run_id, metric, float(snap.live_field_old_value or 0))
                    elif kind_prefix == "mv_tag":
                        _, name, version, tag = parts
                        client.set_model_version_tag(name, version, tag,
                                                     snap.live_field_old_value or "")
                reverted += 1
                logger.info(f"Tamper RESET: {event_type}/{event_id}/{kind}")
            except Exception as e:
                logger.warning(f"Reset failed for {key}: {e}")

    return reverted


def schedule_auto_revert(event_type, event_id, *, lifecycle_store, record_store, tracking_uri,
                         delay_seconds=None):
    """Spawn a background task that calls reset() after a delay."""
    delay = delay_seconds if delay_seconds is not None else TAMPER_TTL_SECONDS

    async def _revert():
        await asyncio.sleep(delay)
        try:
            reset(event_type, event_id, lifecycle_store=lifecycle_store,
                  record_store=record_store, tracking_uri=tracking_uri)
        except Exception as e:
            logger.warning(f"Auto-revert raised: {e}")

    asyncio.create_task(_revert())
```

- [ ] **Step 2: Add tamper routes to `app/main.py`**

Find a good insertion point in `app/main.py` (near other route definitions, after `/verify/{decision_id}`). Add:

```python
from app import tamper as tamper_mod


@app.post("/tamper/saved/{event_type}/{event_id}")
def tamper_saved_route(request: Request, event_type: str, event_id: str,
                       background_tasks: BackgroundTasks):
    if event_type not in ("decision", "training", "registration"):
        return JSONResponse({"error": "unknown event_type"}, status_code=400)
    settings = request.app.state.settings
    try:
        tamper_mod.tamper_saved(
            event_type, event_id,
            lifecycle_store=request.app.state.lifecycle_store,
            record_store=request.app.state.store,
            tracking_uri=settings.mlflow_tracking_uri,
        )
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    # Schedule auto-revert.
    background_tasks.add_task(
        _scheduled_revert, request.app, event_type, event_id,
    )
    return {"tampered": True, "kind": "saved", "event_id": event_id,
            "ttl_seconds": tamper_mod.TAMPER_TTL_SECONDS}


@app.post("/tamper/live/{event_type}/{event_id}")
def tamper_live_route(request: Request, event_type: str, event_id: str,
                      background_tasks: BackgroundTasks):
    if event_type not in ("decision", "training", "registration"):
        return JSONResponse({"error": "unknown event_type"}, status_code=400)
    settings = request.app.state.settings
    try:
        tamper_mod.tamper_live(
            event_type, event_id,
            lifecycle_store=request.app.state.lifecycle_store,
            record_store=request.app.state.store,
            tracking_uri=settings.mlflow_tracking_uri,
        )
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    background_tasks.add_task(
        _scheduled_revert, request.app, event_type, event_id,
    )
    return {"tampered": True, "kind": "live", "event_id": event_id,
            "ttl_seconds": tamper_mod.TAMPER_TTL_SECONDS}


@app.post("/tamper/reset/{event_type}/{event_id}")
def tamper_reset_route(request: Request, event_type: str, event_id: str):
    if event_type not in ("decision", "training", "registration"):
        return JSONResponse({"error": "unknown event_type"}, status_code=400)
    settings = request.app.state.settings
    reverted = tamper_mod.reset(
        event_type, event_id,
        lifecycle_store=request.app.state.lifecycle_store,
        record_store=request.app.state.store,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    return {"reset": True, "reverted_count": reverted, "event_id": event_id}


def _scheduled_revert(app, event_type, event_id):
    """Wrapper for BackgroundTasks — runs reset after the TTL sleep."""
    import time
    time.sleep(tamper_mod.TAMPER_TTL_SECONDS)
    settings = app.state.settings
    try:
        tamper_mod.reset(
            event_type, event_id,
            lifecycle_store=app.state.lifecycle_store,
            record_store=app.state.store,
            tracking_uri=settings.mlflow_tracking_uri,
        )
    except Exception as e:
        logger.warning(f"Auto-revert raised: {e}")
```

(Imports at top of file: confirm `BackgroundTasks` is imported from FastAPI.)

- [ ] **Step 3: Run tamper tests — expect them to start passing**

```bash
pytest tests/test_tamper_endpoints.py -v
```

Expected: most tests pass. The `test_tamper_unknown_event_id_returns_404` test should pass; the happy-path tests pass if the test fixture's auto-trained model + live MLflow integration works in the isolated tmp_path.

- [ ] **Step 4: Iterate on test failures**

If any test fails, debug:
- Check `app/main.py` startup logs to confirm a model was auto-trained
- Verify `record_store.get_by_id()` returns the just-made decision
- Verify the artifact path in MLflow's tmp store matches `ario/predictions/<id>/payload.json`

Edge cases that may need fixes in `app/tamper.py`:
- MLflow `set_trace_tag` may not exist on older MLflow versions; fall back to `set_tag` on the run if needed
- `client.download_artifacts` requires the artifact_uri be on a writable filesystem in tmp_path

- [ ] **Step 5: Run full pytest**

```bash
pytest -q
```

Expected: all tests pass (including any pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add app/tamper.py app/main.py tests/test_tamper_endpoints.py
git commit -m "Phase 3 — tamper backend (saved/live/reset endpoints + state manager)"
```

---

### Task C3: Confirm legacy /api/export removal (or skip)

**Files:**
- Modify (possibly): `app/main.py`

- [ ] **Step 1: Search for callers of `/api/export/{decision_id}`**

```bash
grep -rn "/api/export" templates/ app/ scripts/ tests/ 2>&1 | grep -v __pycache__
```

If only references are in `app/main.py` (the route definition itself) and `templates/decision_detail.html` (the `Download Proof` button which we'll remove in Phase D), the route is safe to delete.

- [ ] **Step 2: Delete the route**

In `app/main.py`, remove the `@app.get("/api/export/{decision_id}")` block (and its handler function).

- [ ] **Step 3: Run pytest**

```bash
pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "Phase 3 — remove legacy /api/export endpoint (replaced by ar.io gateway link)"
```

---

## Phase D — decision_detail.html restructure

This is the largest single template change. The mockup at
`.superpowers/brainstorm/34033-1777551244/content/decision-detail-full-page-v11.html`
is the visual ground truth. The strategy: open the existing template
side-by-side with the mockup, replace each section systematically.

### Task D1: Update page header — add "View Proof ↗" button

**Files:**
- Modify: `templates/decision_detail.html`

- [ ] **Step 1: Locate the current page-header block**

It contains the `Verify with ar.io` button and `Download Proof` button.

- [ ] **Step 2: Replace the page-actions block**

Find:

```html
<a href="/api/export/{{ envelope.record.decision_id }}" class="btn btn-outline btn-sm">Download Proof</a>
```

Replace with:

```html
<a href="https://turbo-gateway.com/{{ envelope.arweave_tx_id }}" target="_blank" class="btn btn-outline btn-sm" {% if not envelope.arweave_tx_id %}style="display:none;"{% endif %}>View Proof ↗</a>
```

- [ ] **Step 3: Boot demo, verify the button renders correctly**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
# Predict so we have a record:
curl -s -X POST http://localhost:8000/predict-form -d "annual_income=78000&credit_utilization=0.18&debt_to_income_ratio=0.22&months_employed=72&credit_score=745"
# Get a decision_id and visit detail:
DECISION_ID=$(curl -s http://localhost:8000/decisions | python3 -c "import json, sys; d = json.load(sys.stdin); print(d[0]['record']['decision_id'])")
curl -s "http://localhost:8000/ui/decisions/$DECISION_ID" | grep -E "View Proof|turbo-gateway" | head -3
kill %1 2>/dev/null
```

Expected: matches showing the View Proof ↗ button.

- [ ] **Step 4: Commit**

```bash
git add templates/decision_detail.html
git commit -m "Phase 3 — decision_detail page header View Proof link"
```

---

### Task D2: Update Decision card (was Prediction)

**Files:**
- Modify: `templates/decision_detail.html`

- [ ] **Step 1: Locate the Prediction card**

Find `<div class="section-header">Prediction</div>` and the surrounding `<div class="section">…</div>`.

- [ ] **Step 2: Rename and restructure the card**

Replace the entire card block with:

```html
<!-- Decision -->
<div class="section">
  <div class="section-header">Decision</div>
  <div class="section-body">
    <div class="kv">
      <div class="kv-row">
        <span class="kv-key">Result</span>
        <span class="kv-value">
          {% if envelope.record.prediction['class'] == 'approve' %}
            <span class="badge badge-green"><span class="badge-dot"></span> Approved</span>
          {% elif envelope.record.prediction['class'] == 'deny' %}
            <span class="badge badge-red"><span class="badge-dot"></span> Denied</span>
          {% else %}
            {{ envelope.record.prediction['class'] }}
          {% endif %}
        </span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Confidence</span>
        <span class="kv-value">{{ "%.1f"|format(envelope.record.prediction.probabilities[envelope.record.prediction['class']] * 100) }}%</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Features</span>
        <span class="kv-value">{{ envelope.record.prediction.features_used | join(', ') }}</span>
      </div>
    </div>
    <div style="margin-top: 0.85rem; padding-top: 0.85rem; border-top: 1px solid var(--border);">
      <div style="font-size: 0.7rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.4rem;">Class probabilities</div>
      {% for cls, prob in envelope.record.prediction.probabilities.items() %}
      <div class="prob-row">
        <span class="prob-label">{{ cls.title() }}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width: {{ (prob * 100) }}%"></div>
        </div>
        <span class="prob-value">{{ "%.4f"|format(prob) }}</span>
      </div>
      {% endfor %}
    </div>
  </div>
</div>
```

- [ ] **Step 3: Verify rendering**

Boot demo, visit a decision detail page. Confirm:
- Header reads "Decision"
- Result shows green `Approved` badge or red `Denied` badge
- Confidence shows a percent
- Class probability bars show `Deny` / `Approve` (Title Case)

- [ ] **Step 4: Commit**

```bash
git add templates/decision_detail.html
git commit -m "Phase 3 — decision_detail Prediction → Decision card with Result chip"
```

---

### Task D3: Update ar.io Verification card to 3-row structure

**Files:**
- Modify: `templates/decision_detail.html`

- [ ] **Step 1: Locate the existing verification card**

Find `<div class="section-header">ar.io Verification</div>`.

- [ ] **Step 2: Replace the card body — new 3-row structure**

Replace the section body (everything between `<div class="section-body">` and the matching `</div>` closing it):

```html
<div class="section-body">
  <div style="font-size:0.78rem;color:var(--muted);line-height:1.5;margin-bottom:0.75rem;">
    Independent verification of MLflow data integrity. These verify this decision record matches the original proof anchored with ar.io at runtime — they do <strong>not</strong> speak to whether the decision itself was correct.
  </div>

  <div class="verify-row" title="The proof exists on Arweave and ar.io can locate it. FAIL means the on-chain anchor is missing or unreachable.">
    <span>Proof Found</span>
    {% if v %}
      {% if v.signature_valid is sameas true and envelope.arweave_tx_id %}
        <span class="check">PASS</span>
      {% elif v.signature_valid is sameas false %}
        <span class="cross">FAIL</span>
      {% else %}
        <span class="badge badge-yellow">Pending</span>
      {% endif %}
    {% else %}
      <span class="unchecked">Not checked</span>
    {% endif %}
  </div>

  <div class="verify-row" title="The data in MLflow (both the saved canonical bytes and the live state) hashes to the same value that was anchored on Arweave. ar.io independently re-verifies the bytes. FAIL means MLflow data has been tampered with since anchoring.">
    <span>Decision Record Matches</span>
    {% if v %}
      {% if v.hash_match is sameas false or v.source_of_truth_ok is sameas false %}
        <span class="cross">FAIL</span>
      {% elif v.permanent_copy_found and v.hash_match is sameas true and v.source_of_truth_ok is sameas true %}
        <span class="check">PASS</span>
      {% else %}
        <span class="badge badge-yellow">Pending</span>
      {% endif %}
    {% else %}
      <span class="unchecked">Not checked</span>
    {% endif %}
  </div>

  <div class="verify-row" title="The proof carries a valid signature from the user who issued it. ar.io independently re-verifies the signature. FAIL means the proof was altered after signing.">
    <span>Signature Confirmed</span>
    {% if v %}
      {% if v.signature_valid is sameas true and v.attestation_level and v.attestation_level >= 3 %}
        <span class="check">PASS</span>
      {% elif v.signature_valid is sameas false %}
        <span class="cross">FAIL</span>
      {% else %}
        <span class="badge badge-yellow">Pending</span>
      {% endif %}
    {% else %}
      <span class="unchecked">Not checked</span>
    {% endif %}
  </div>

  {% if v and v.attested_by %}
  <div class="verify-row" title="An ar.io gateway operator ran the verification above and cryptographically signed the result with their wallet. Independent statement on the ar.io network, separate from the verification itself.">
    <span>Attested by</span>
    <span style="text-align: right;">
      <strong>{{ v.attested_by }}</strong>
      <span style="font-size:0.7rem;color:var(--muted);margin-left:0.35rem;">independent ar.io operator</span>
      {% if v.attested_at %}
        <div style="font-size:0.7rem;color:var(--muted);margin-top:0.15rem;">{{ v.attested_at[:19] }}Z</div>
      {% endif %}
    </span>
  </div>
  {% endif %}

  <div class="verify-row" style="border-top: 1px solid var(--border); padding-top: 0.75rem; margin-top: 0.25rem;">
    <strong style="font-size: 0.85rem;">Overall</strong>
    {% if v %}
      {% if v.overall is sameas true %}
        <strong class="check">PASS</strong>
      {% elif v.overall is sameas false %}
        <strong class="cross">FAIL</strong>
      {% else %}
        <span class="badge badge-yellow">Pending</span>
      {% endif %}
    {% else %}
      <span class="unchecked">Not verified</span>
    {% endif %}
  </div>

  <details style="margin-top: 0.85rem; font-size: 0.75rem; color: var(--muted); border-top: 1px solid var(--border); padding-top: 0.75rem;">
    <summary style="cursor: pointer; font-weight: 600; color: var(--black); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em;">▶ How to verify independently</summary>
    <div style="margin-top: 0.5rem; line-height: 1.6;">
      <p>1. Fetch the pure-commitment envelope from Arweave:</p>
      <code style="display: block; background: var(--surface); padding: 0.4rem 0.6rem; border-radius: 4px; margin: 0.25rem 0 0.5rem; font-size: 0.7rem; word-break: break-all; font-family: 'JetBrains Mono', monospace;">curl https://turbo-gateway.com/raw/{{ envelope.arweave_tx_id }}</code>
      <p>2. Verify the Ed25519 signature using the embedded public_key (the signed body is the envelope minus the signature field, JCS-canonicalized).</p>
      <p style="margin-top: 0.25rem;">3. Download the canonical bytes from MLflow and confirm SHA-256 matches the envelope's payload_hash.</p>
      <p style="margin-top: 0.25rem; font-style: italic;">Or run <code style="font-family: 'JetBrains Mono', monospace;">ario-mlflow verify trace {{ envelope.record.trace_id }}</code> for all checks at once.</p>
    </div>
  </details>
</div>
```

- [ ] **Step 3: Verify rendering**

Boot demo, visit a decision detail page. Verify all four rows + Attested by + Overall + collapsible all render as expected.

- [ ] **Step 4: Commit**

```bash
git add templates/decision_detail.html
git commit -m "Phase 3 — decision_detail verify card 3-row structure"
```

---

### Task D4: Replace stacked sections with 2x2 audit grid

**Files:**
- Modify: `templates/decision_detail.html`

- [ ] **Step 1: Locate the four stacked sections after the top grid**

These are: `Decision Record`, `ar.io anchoring`, `Model Lineage`, `Trace Context`. Plus the deprecated comment block left from Phase 2.D.

- [ ] **Step 2: Delete those four sections and the comment**

Delete everything between the closing `</div>` of the top 2-col `.grid` and the start of `{% endblock %}` content end. (Use Read to locate exact line numbers; this is a chunk of about 200 lines.)

- [ ] **Step 3: Insert the 2x2 audit grid**

Add a closing `</div>` for the existing top `.grid` if needed, then add:

```html
<!-- 2x2 audit context grid -->
<div class="grid-2-audit" style="margin-top: 1rem;">

  <div class="section">
    <div class="section-header">Model</div>
    <div class="section-body">
      <div class="kv-row"><span class="kv-key">Name</span><span class="kv-value">{{ envelope.record.model_name }}</span></div>
      <div class="kv-row"><span class="kv-key">Version</span><span class="kv-value">v{{ envelope.record.model_version }}</span></div>
      <div class="kv-row"><span class="kv-key">Run ID</span><span class="kv-value"><a class="mono" href="/ui/runs/{{ envelope.record.mlflow_run_id }}" style="word-break: break-all; text-decoration: none;">{{ envelope.record.mlflow_run_id }}</a></span></div>
      <div class="kv-row"><span class="kv-key">Lineage</span><span class="kv-value"><a href="/ui/models/{{ envelope.record.model_name }}/{{ envelope.record.model_version }}" style="text-decoration: none; color: var(--primary);">View model lineage →</a></span></div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">Inference</div>
    <div class="section-body">
      <div class="kv-row"><span class="kv-key">Timestamp</span><span class="kv-value">{{ envelope.record.timestamp }}</span></div>
      <div class="kv-row"><span class="kv-key">Latency</span><span class="kv-value">{{ envelope.record.latency_ms }} ms</span></div>
      <div class="kv-row"><span class="kv-key">Input Hash</span><span class="kv-value mono mono-sm">{{ envelope.record.input_hash }}</span></div>
      <div class="kv-row"><span class="kv-key">Output Hash</span><span class="kv-value mono mono-sm">{{ envelope.record.output_hash }}</span></div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">Trace</div>
    <div class="section-body">
      <div class="kv-row"><span class="kv-key">Service</span><span class="kv-value">{{ envelope.record.service_name }}</span></div>
      <div class="kv-row"><span class="kv-key">Trace ID</span><span class="kv-value mono mono-sm">{{ envelope.record.trace_id }}</span></div>
      <div class="kv-row"><span class="kv-key">Span ID</span><span class="kv-value mono mono-sm">{{ envelope.record.span_id }}</span></div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">ar.io Anchor</div>
    <div class="section-body">
      {% if envelope.arweave_tx_id %}
      <div class="kv-row"><span class="kv-key">Transaction</span><span class="kv-value mono mono-sm">{{ envelope.arweave_tx_id }}</span></div>
      <div class="kv-row"><span class="kv-key">Status</span><span class="kv-value">
        {% if turbo_status and turbo_status.status in ('FINALIZED', 'CONFIRMED') %}
          <span class="badge badge-green"><span class="badge-dot"></span> Confirmed</span>
        {% elif turbo_status and turbo_status.status == 'NOT_FOUND' %}
          <span class="badge badge-yellow"><span class="badge-dot badge-dot-pulse"></span> Anchoring</span>
        {% else %}
          <span class="badge badge-yellow"><span class="badge-dot"></span> Anchoring</span>
        {% endif %}
      </span></div>
      {% if envelope.public_key %}
      <div class="kv-row"><span class="kv-key">Signer Key</span><span class="kv-value mono mono-sm">ed25519:{{ envelope.public_key }}</span></div>
      {% endif %}
      {% if envelope.turbo_receipt and envelope.turbo_receipt.deadlineHeight %}
      <div class="kv-row"><span class="kv-key">Block</span><span class="kv-value">{{ "{:,}".format(envelope.turbo_receipt.deadlineHeight) }}</span></div>
      {% endif %}
      {% else %}
      <div class="kv-row" style="color: var(--muted);">Not anchored — no wallet configured.</div>
      {% endif %}
    </div>
  </div>

</div>
```

- [ ] **Step 4: Boot demo, verify all four cards render**

Confirm Model / Inference / Trace / ar.io Anchor cards display correctly with full (non-truncated) values.

- [ ] **Step 5: Commit**

```bash
git add templates/decision_detail.html
git commit -m "Phase 3 — decision_detail audit grid (2x2: Model/Inference/Trace/ar.io Anchor)"
```

---

### Task D5: Add Demonstrate Tampering collapsible

**Files:**
- Modify: `templates/decision_detail.html`

- [ ] **Step 1: Add the section label and collapsible block**

Insert below the audit grid, before `{% endblock %}` for content:

```html
<!-- ─── Demonstrate Tampering ─── -->
<div class="section-label">Demonstrate Tampering</div>

<details class="tamper-section" id="tamper-section">
  <summary class="tamper-header">
    <div class="tamper-header-inner">
      <span class="tamper-summary-icon"><svg viewBox="0 0 10 10"><path d="M3 1 L7 5 L3 9" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
      <span class="tamper-summary-title">Click to tamper</span>
      <span class="tamper-summary-hint">simulate two attacks</span>
    </div>
  </summary>
  <div class="tamper-body">
    <div class="tamper-intro">
      Two ways someone could try to alter this decision record after the fact — both hit MLflow data, both get caught by the same verification check. Click to see how. Tampers auto-reverse after a short window, or click <em>Reset all</em>.
    </div>
    <div class="tamper-row">
      <div class="tamper-row-info">
        <div class="tamper-row-title"><span class="num">1</span>Tamper with the saved record</div>
        <div class="tamper-row-desc">Edit the locally-saved copy of the decision data. <em style="font-family: 'JetBrains Mono', monospace; color: var(--muted); font-style: normal;">Overwrites <code>ario/payload.json</code> in MLflow.</em> <strong>Breaks → Decision Record Matches.</strong></div>
      </div>
      <button class="btn-tamper" data-tamper-kind="saved" data-event-type="decision" data-event-id="{{ envelope.record.decision_id }}">Tamper</button>
    </div>
    <div class="tamper-row">
      <div class="tamper-row-info">
        <div class="tamper-row-title"><span class="num">2</span>Tamper with the live data</div>
        <div class="tamper-row-desc">Mutate MLflow's underlying record — the same thing an admin with registry access could do. <em style="font-family: 'JetBrains Mono', monospace; color: var(--muted); font-style: normal;">Overwrites the <code>ario.payload_json</code> trace tag.</em> <strong>Breaks → Decision Record Matches.</strong></div>
        <span class="tamper-headline">★ Headline tamper for governance audiences</span>
      </div>
      <button class="btn-tamper" data-tamper-kind="live" data-event-type="decision" data-event-id="{{ envelope.record.decision_id }}">Tamper</button>
    </div>
    <div class="tamper-footer">
      <div style="font-size: 0.74rem; color: var(--muted); line-height: 1.45; font-style: italic;">Both tampers hit different surfaces but the system catches each one — data tampering anywhere in MLflow shows up as the same broken check.</div>
      <div style="display: flex; align-items: center; padding-top: 0.45rem; border-top: 1px solid var(--border);">
        <span class="tamper-footer-text">Tampers auto-reverse after a short window. Reset clears all.</span>
        <button class="btn-reset" data-event-type="decision" data-event-id="{{ envelope.record.decision_id }}">Reset all</button>
      </div>
    </div>
  </div>
</details>
```

- [ ] **Step 2: Add the tamper button JavaScript**

In the `{% block scripts %}` of `decision_detail.html`, add:

```javascript
// Tamper button wiring
document.querySelectorAll('.btn-tamper').forEach(btn => {
  btn.addEventListener('click', async () => {
    const kind = btn.dataset.tamperKind;
    const eventType = btn.dataset.eventType;
    const eventId = btn.dataset.eventId;
    btn.disabled = true;
    btn.textContent = 'Tampering...';
    try {
      const r = await fetch(`/tamper/${kind}/${eventType}/${eventId}`, { method: 'POST' });
      const data = await r.json();
      if (data.tampered) {
        btn.textContent = 'Tampered';
        // Subtle visual feedback — pulse the section.
        document.getElementById('tamper-section').style.boxShadow = '0 0 0 2px var(--yellow)';
        setTimeout(() => {
          document.getElementById('tamper-section').style.boxShadow = '';
          btn.textContent = 'Tamper';
          btn.disabled = false;
        }, 4000);
      } else {
        btn.textContent = 'Failed';
        btn.disabled = false;
      }
    } catch (e) {
      btn.textContent = 'Error';
      btn.disabled = false;
    }
  });
});

document.querySelectorAll('.btn-reset').forEach(btn => {
  btn.addEventListener('click', async () => {
    const eventType = btn.dataset.eventType;
    const eventId = btn.dataset.eventId;
    btn.disabled = true;
    btn.textContent = 'Resetting...';
    try {
      await fetch(`/tamper/reset/${eventType}/${eventId}`, { method: 'POST' });
      btn.textContent = 'Reset complete — re-verify to see';
      setTimeout(() => {
        btn.textContent = 'Reset all';
        btn.disabled = false;
      }, 2500);
    } catch (e) {
      btn.textContent = 'Error';
      btn.disabled = false;
    }
  });
});
```

- [ ] **Step 3: Boot demo, exercise tamper buttons end-to-end**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
# Predict, get decision_id:
curl -s -X POST http://localhost:8000/predict-form -d "annual_income=78000&credit_utilization=0.18&debt_to_income_ratio=0.22&months_employed=72&credit_score=745"
DECISION_ID=$(curl -s http://localhost:8000/decisions | python3 -c "import json, sys; d = json.load(sys.stdin); print(d[0]['record']['decision_id'])")
echo "decision_id: $DECISION_ID"

# Tamper saved:
curl -X POST "http://localhost:8000/tamper/saved/decision/$DECISION_ID" | python3 -m json.tool

# Verify — should show check 2 FAIL:
curl -X POST "http://localhost:8000/verify/$DECISION_ID" | python3 -m json.tool

# Reset:
curl -X POST "http://localhost:8000/tamper/reset/decision/$DECISION_ID" | python3 -m json.tool

# Verify — should be back to PASS:
curl -X POST "http://localhost:8000/verify/$DECISION_ID" | python3 -m json.tool

kill %1 2>/dev/null
```

Expected: tamper response `tampered: true`, verify after tamper shows `hash_match: false` or `source_of_truth_ok: false`, reset returns `reset: true`, verify after reset shows all PASS.

If any step fails, debug `app/tamper.py` accordingly.

- [ ] **Step 4: Commit**

```bash
git add templates/decision_detail.html
git commit -m "Phase 3 — decision_detail Demonstrate Tampering collapsible"
```

---

### Task D6: Add "How verification works" collapsible (split proof viewer)

**Files:**
- Modify: `templates/decision_detail.html`

- [ ] **Step 1: Add the section label and collapsible block**

Insert below the Demonstrate Tampering section:

```html
<!-- ─── How verification works ─── -->
<div class="section-label">How verification works</div>

<details class="collapsible">
  <summary>
    <span class="collapsible-summary-icon"><svg viewBox="0 0 10 10"><path d="M3 1 L7 5 L3 9" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
    <span class="collapsible-summary-title">Show the canonical bytes ↔ signed commitment</span>
    <span class="collapsible-summary-hint">click to expand</span>
  </summary>
  <div class="collapsible-body">
    <div class="viewer-intro">
      The proof has two parts. Ar.io anchors a tiny <strong>signed commitment</strong> — just hashes and a signature, no source data. MLflow stores the <strong>canonical bytes</strong> that produced the hash. Anyone with both can verify they match.
    </div>

    <div class="proof-split">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title-row"><div class="panel-title">Canonical bytes</div><span class="panel-source-pill pill-mlflow">MLflow</span></div>
          <div class="panel-meta">JCS canonicalized · the source · what gets hashed</div>
        </div>
        <div class="panel-body">
          <div class="json-block" id="canonical-bytes-json">{% if canonical_bytes_json %}{{ canonical_bytes_json | safe }}{% else %}<em style="color:var(--muted)">Click <strong>Verify with ar.io</strong> first to fetch and show the canonical bytes.</em>{% endif %}</div>
        </div>
        <div class="panel-footer">
          <span style="font-family: monospace;">ario/predictions/{{ envelope.record.decision_id }}/payload.json</span>
          {% if envelope.record.mlflow_run_id %}
          <a href="#" target="_blank">Download →</a>
          {% endif %}
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div class="panel-title-row"><div class="panel-title">Signed commitment</div><span class="panel-source-pill pill-arweave">ar.io</span></div>
          <div class="panel-meta">~500 bytes · permanent · public · the witness</div>
        </div>
        <div class="panel-body">
          <div class="json-block" id="signed-commitment-json">{% if signed_commitment_json %}{{ signed_commitment_json | safe }}{% else %}<em style="color:var(--muted)">Click <strong>Verify with ar.io</strong> first to fetch and show the signed commitment.</em>{% endif %}</div>
        </div>
        <div class="panel-footer">
          {% if envelope.arweave_tx_id %}
          <span>TX <span style="font-family: monospace; color: var(--primary);">{{ envelope.arweave_tx_id }}</span></span>
          <a href="https://turbo-gateway.com/{{ envelope.arweave_tx_id }}" target="_blank">View on ar.io →</a>
          {% endif %}
        </div>
      </div>
    </div>

    <div class="equality">
      <span class="equality-prefix">Verifier recomputes:</span>
      <div class="equality-formula">
        <span class="eq-token eq-left">SHA-256(canonical bytes)</span>
        <span class="eq-op">=</span>
        <span class="eq-token eq-right">payload_hash</span>
        {% if v and v.hash_match is sameas true %}
          <span class="eq-result">✓ matches</span>
        {% elif v and v.hash_match is sameas false %}
          <span class="eq-result eq-result-fail">✗ doesn't match</span>
        {% endif %}
      </div>
    </div>

    <div class="legend-row">
      <div><strong>What you can't do.</strong> Change either side without breaking the equality. The signature on the signed commitment binds <code style="font-family: monospace;">payload_hash</code> to the signer's key.</div>
      <div><strong>What this isn't.</strong> Canonical bytes contain hashes of input/output, not the values themselves — predictions stay private.</div>
    </div>
  </div>
</details>
```

- [ ] **Step 2: Wire up `canonical_bytes_json` and `signed_commitment_json` in `app/ui.py`**

In `app/ui.py::decision_detail` (the route handler), add after `_verify_envelope` is called:

```python
# Phase 3: render the actual canonical bytes + signed envelope as
# pretty-printed JSON for the "How verification works" viewer.
canonical_bytes_json = None
signed_commitment_json = None
if v and envelope.get("arweave_tx_id"):
    plugin_envelope = app.state.anchor.fetch_proof(envelope["arweave_tx_id"])
    if plugin_envelope:
        signed_commitment_json = json.dumps(plugin_envelope, indent=2)
        # Canonical bytes are inside the envelope's payload field, or fetched separately.
        # The plugin's full_verify already downloads them; surface here for display.
        try:
            from ario_mlflow.verify import _download_canonical_bytes
            mlflow_client = mlflow.tracking.MlflowClient()
            canonical = _download_canonical_bytes(plugin_envelope, mlflow_client)
            if canonical:
                canonical_bytes_json = canonical.decode('utf-8')
        except Exception:
            pass

context["canonical_bytes_json"] = canonical_bytes_json
context["signed_commitment_json"] = signed_commitment_json
```

- [ ] **Step 3: Boot demo, verify the panels render**

Visit a decision detail page after running Verify. Open the "How verification works" collapsible, confirm both JSON panels show the actual envelope and canonical bytes.

- [ ] **Step 4: Add the payload-hash-line highlight to the signed envelope JSON**

If the JSON rendering wraps `"payload_hash"` lines, this requires a server-side syntax-highlighter or a JavaScript pass. For Phase 3, a simple JavaScript that finds the `payload_hash` line in `#signed-commitment-json` and wraps it in a `<span class="payload-hash-line">…</span>` is sufficient:

In `{% block scripts %}` of `decision_detail.html`:

```javascript
// Highlight the payload_hash line in the signed commitment panel.
(function() {
  const el = document.getElementById('signed-commitment-json');
  if (!el || !el.textContent) return;
  const lines = el.textContent.split('\n');
  const highlighted = lines.map(line => {
    if (line.trim().startsWith('"payload_hash"')) {
      return '<span class="payload-hash-line">' + line + '</span>';
    }
    return line;
  });
  el.innerHTML = highlighted.join('\n');
})();
```

(Naive — not safe against XSS for arbitrary content. For demo data this is fine because the envelope is fetched from ar.io and is trusted.)

- [ ] **Step 5: Commit**

```bash
git add templates/decision_detail.html app/ui.py
git commit -m "Phase 3 — decision_detail How verification works split proof viewer"
```

---

### Task D7: Verify decision_detail end-to-end and visual review

**Files:** none modified

- [ ] **Step 1: Boot demo and exercise the full flow**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
# Open in browser:
open http://localhost:8000/
```

Manual checks:
1. Make a prediction. Confirm it lands in the records list with new status badges.
2. Click into the decision. Confirm:
   - Decision card shows green/red Result badge with Confidence %
   - ar.io Verification card shows 3-row PASS state
   - 2x2 audit grid renders Model / Inference / Trace / ar.io Anchor with full values
   - "Click to tamper" collapsible (closed by default) — click to expand
   - Click Tamper #1 → click Verify → row 2 turns FAIL
   - Click Reset all → click Verify → row 2 PASS again
   - "How verification works" collapsible — click to expand
   - Both JSON panels render; payload_hash line highlighted yellow
3. Click View Proof ↗ — opens turbo-gateway.com.

- [ ] **Step 2: If anything looks off, fix and add a follow-up task**

Don't try to fix everything in one go. If a regression is small, fix and commit. If it's bigger, add a TODO commit and continue.

- [ ] **Step 3: Commit any visual fixes**

```bash
git add -A
git commit -m "Phase 3 — decision_detail visual fixes from end-to-end review"
```

(Skip if no fixes needed.)

---

## Phase E — run_detail.html restructure

### Task E1: Restructure run_detail.html

**Files:**
- Modify: `templates/run_detail.html`

The mockup at `.superpowers/brainstorm/34033-1777551244/content/run-detail-page-v1.html` is the reference.

- [ ] **Step 1: Read the existing template to understand current structure**

Note line ranges for each existing section (Parameters, ar.io Verification, Metrics, ar.io anchoring, Artifact Integrity, Proof Layer, Model, ario.* tags).

- [ ] **Step 2: Update page header — add "View Proof ↗" button**

Same pattern as decision_detail. Add the View Proof button next to Verify with ar.io.

- [ ] **Step 3: Replace top 2-col grid with Training summary + new verify card**

The Training summary card combines today's Parameters and Metrics into one card with a headline status row + sub-section labels for Parameters and Metrics.

The verify card mirrors decision_detail's 3-row structure but with row 2 reading **`Training Record Matches`** and the intro text adapted: *"…matches the original proof anchored with ar.io at training time — they do not speak to whether the model itself is good."*

(Copy the template block from `run-detail-page-v1.html`, lines for Training card and ar.io Verification card. Substitute Jinja variables `envelope.record.params.<key>`, `envelope.record.metrics.accuracy`, etc. for the hardcoded mockup values.)

- [ ] **Step 4: Replace stacked sections with 2x2 audit grid**

Delete `Metrics`, `ar.io anchoring`, `Artifact Integrity`, **`Proof Layer`** (entirely — this is the explicit removal called out in the spec), `Model`, `ario.* tags` standalone sections.

Insert the 2x2 audit grid:
- **Model** card (Name, Version, Run ID linkable, Lineage)
- **Run** card (Started timestamp, Duration if computable, Train samples, Test samples)
- **Artifact Integrity** card (Artifact Hash, per-file checksums, git_commit — full hashes, no truncation)
- **ar.io Anchor** card (Transaction, Status, Signer Key, Block)

(Reference run-detail-page-v1.html for exact structure. Substitute Jinja variables.)

- [ ] **Step 5: Add Demonstrate Tampering collapsible**

Same pattern as decision_detail's, but tampers are training-event-specific:

- Tamper #1: `Tamper with the saved record` — overwrites training run's `ario/payload.json` → Breaks `Training Record Matches`
- Tamper #2: `Tamper with the live data` ★ — *"Mutate MLflow's underlying training run — for example, change a metric or replace the model artifact. (Overwrites a logged metric or rewrites `model.pkl`.)"* → Breaks `Training Record Matches`

The tamper buttons send `data-event-type="training" data-event-id="{{ envelope.record.run_id }}"`.

- [ ] **Step 6: Add How verification works collapsible**

Same split-proof viewer pattern. Use training-specific canonical bytes (params, metrics, artifact_checksums) — the actual data comes from the same `_download_canonical_bytes` mechanism added in Task D6.

- [ ] **Step 7: Add Live MLflow tags collapsible**

Wrap the existing `ario.* tags (live from MLflow)` content in a `<details class="collapsible">` with summary `Show the actual tags on this run in MLflow` + hint `pulled live · MLflowClient.get_run()`.

- [ ] **Step 8: Reuse the same JavaScript block as decision_detail for tamper button wiring**

The script block needs the same handlers — `.btn-tamper` and `.btn-reset` listeners. This is duplicated; for now copy from decision_detail. (DRY refactor into base.html shared script can come later.)

- [ ] **Step 9: Boot demo, navigate to a training run via Models → View chain → Training Run header**

Manual checks:
1. Page header has both Verify and View Proof buttons
2. Training card shows Status + Accuracy headline + parameters/metrics
3. Verify card shows new 3 rows
4. 2x2 audit grid shows Model / Run / Artifact Integrity / ar.io Anchor
5. Tamper section opens, both tamper buttons work end-to-end
6. How verification works opens, panels render
7. Live MLflow tags opens, shows actual tags

- [ ] **Step 10: Run pytest**

```bash
pytest -q
```

- [ ] **Step 11: Commit**

```bash
git add templates/run_detail.html
git commit -m "Phase 3 — run_detail restructure (3-row verify, audit grid, collapsibles, tampers)"
```

---

## Phase F — model_chain.html restructure

### Task F1: Restructure model_chain.html

**Files:**
- Modify: `templates/model_chain.html`

The mockup at `.superpowers/brainstorm/34033-1777551244/content/model-chain-page-v1.html` is the reference.

- [ ] **Step 1: Update page header**

Subtitle: *"{model_name} / v{version} — every training run, model version, and prediction in this model's history, each cryptographically verifiable on ar.io."* (replaces *"a cryptographically verifiable audit trail of every training run, model version, and prediction."*)

Action button: rename `Verify All` → **`Verify chain`**.

- [ ] **Step 2: Update Training Run node**

The chain-card-body is currently a single column with a verify-inline strip. Restructure to 2 columns:
- Left: existing details (Run ID full + linkable, params, metrics, artifact hash full + sha256: prefix, git commit full)
- Right: mini-verify card with the new 3-row structure (`Proof Found`, `Training Record Matches`, `Signature Confirmed`) + Attested by + Overall

(Copy structure from model-chain-page-v1.html.)

- [ ] **Step 3: Update Model Registration node**

Same 2-col body with mini-verify. Row 2 reads `Registration Record Matches`.

- [ ] **Step 4: Update Decisions node row labels**

```
old: <span class="kv-key">Anchored</span>
new: <span class="kv-key">Pending verification</span>
```

Add a `Tampered` row if `tampered_count` is computed (skip if unavailable — out of scope to plumb new counters this PR).

- [ ] **Step 5: Add Demonstrate Tampering collapsible**

Two **live-data** tampers:
1. `Tamper the training run's live data` ★ — *"Mutate the training run's metrics or artifacts in MLflow — for example, change the recorded accuracy from 91.3% to 99.9%. (Overwrites a logged metric.)"* → Breaks `Training Record Matches`
2. `Tamper the registration's live data` — *"Mutate the model version's metadata in MLflow — for example, point this v3 registration at a different training run. (Overwrites the model version's `source_run_id` tag.)"* → Breaks `Registration Record Matches`

Both buttons use `data-tamper-kind="live"`. First button uses `data-event-type="training"` and `data-event-id="{{ training.record.run_id }}"`; second uses `data-event-type="registration"` and `data-event-id="{{ registration.record.event_id }}"`.

Footer note (italic): *"Each tamper changes the live MLflow data at a different link in the chain. The verifier re-derives canonical bytes from the current state and compares to what was anchored — any divergence breaks the matching record's check, and the chain breaks at exactly the link that was attacked."*

- [ ] **Step 6: Add How verification works collapsible (lighter version)**

Don't include the full split-proof viewer here — instead, the body explains the chain-level pattern and links to the detail pages:

```html
<details class="collapsible">
  <summary>...</summary>
  <div class="collapsible-body">
    <p>Each node in the chain follows the same pattern: MLflow stores the canonical bytes for that event, ar.io anchors a tiny signed commitment that includes <code>SHA-256(canonical bytes)</code>. The chain links via <code>previous_hash</code> — each event's signed commitment includes the hash of the previous event in this model's history.</p>
    <p style="margin-top: 0.85rem;">For details, see the <strong>How verification works</strong> section on the <a href="/ui/runs/{{ training.record.run_id }}" style="color: var(--primary); text-decoration: none;">Training Run detail page</a> or any <a href="/ui/decisions" style="color: var(--primary); text-decoration: none;">Decision detail page</a>.</p>
  </div>
</details>
```

- [ ] **Step 7: Update status badges across the file**

Same vocabulary sweep — `Anchored` → `Pending verification`, `Local` → `Not anchored`, `Permanent` → `Confirmed` in chain-card-footer Turbo badges.

- [ ] **Step 8: Add tamper button JavaScript (copy of pattern)**

Same `<script>` block as decision_detail's tamper handlers.

- [ ] **Step 9: Boot demo, exercise chain page**

Manual checks per the design spec:
1. Three chain nodes render with mini verify cards
2. Tamper buttons work, training tamper fails the Training Record Matches row, registration tamper fails the Registration Record Matches row
3. How verification works collapsible opens with the lighter explanation
4. Status badges use new vocabulary

- [ ] **Step 10: Run pytest**

```bash
pytest -q
```

- [ ] **Step 11: Commit**

```bash
git add templates/model_chain.html
git commit -m "Phase 3 — model_chain restructure (mini verify cards, tampers, new vocabulary)"
```

---

## Phase G — Documentation sweep

### Task G1: README.md vocabulary sweep

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Identify references to the four old check labels**

```bash
grep -n "Signature\|Anchored Bytes\|Source of Truth\|Cryptographically verified\|Content integrity\|Finalized on Arweave\|audit trail\|Local Only\|Anchored\|Permanent on" README.md | head -30
```

- [ ] **Step 2: Apply replacements**

Per the spec § 3 cross-cutting vocabulary:
- `Signature` (as a row label) → `Signature Confirmed`
- `Anchored Bytes Intact` and `Source of Truth Matches` (often two adjacent rows in old text) → consolidate as `Decision Record Matches` / `Training Record Matches` / `Registration Record Matches` per the event type the surrounding paragraph is describing
- Attestation values: drop `Level N`, drop `Cryptographically verified` / `Content integrity confirmed` / `Finalized on Arweave`, just say `Verified`
- `audit trail` → `verifiable record`
- `Local Only` → `Not anchored`
- `Anchored` (status) → `Pending verification`
- `Permanent on ar.io` / `Permanent` (status) → `Confirmed`

For each occurrence in README.md, edit with care — the README has narrative context, so blanket find-replace can break sentence flow. Read each match in context; reword if needed.

- [ ] **Step 3: Update verification flow section**

If README has a section listing "the four checks" (it does, per Phase 2 PR feedback rounds), rewrite to the three checks: Proof Found / {event} Record Matches / Signature Confirmed.

- [ ] **Step 4: Update tag table (`Runs` row)**

Per Phase 2 round 5 feedback, the tag table was already cleaned up. Verify it still reads correctly post-Phase-3 vocabulary.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Phase 3 — README vocabulary sweep"
```

---

### Task G2: ario_mlflow/README.md vocabulary sweep

**Files:**
- Modify: `ario_mlflow/README.md`

Same approach as Task G1 but applied to the plugin README. Internal field names (`source_of_truth_ok`, `attestation_level`, etc.) stay; user-facing prose updates.

- [ ] **Step 1: Find references**

```bash
grep -n "Signature\|Anchored Bytes\|Source of Truth\|Cryptographically verified\|Level [123]\|audit trail" ario_mlflow/README.md
```

- [ ] **Step 2: Apply replacements with context-awareness**

- [ ] **Step 3: Commit**

```bash
git add ario_mlflow/README.md
git commit -m "Phase 3 — ario_mlflow README vocabulary sweep"
```

---

### Task G3: ario_mlflow/cli.py output sweep

**Files:**
- Modify: `ario_mlflow/cli.py`

The CLI's printed verify output should match the new labels.

- [ ] **Step 1: Find printed strings**

```bash
grep -n "Signature\|Anchored\|Source of Truth\|Cryptographically\|Level\|hash_match\|signature_valid" ario_mlflow/cli.py
```

- [ ] **Step 2: Update labels in CLI output**

For example, if cli.py prints:

```python
print(f"  Signature: {'PASS' if r['signature']['ok'] else 'FAIL'}")
print(f"  Anchored Bytes: {...}")
```

Update to:

```python
print(f"  Signature Confirmed: {'PASS' if r['signature']['ok'] else 'FAIL'}")
print(f"  Decision Record Matches: {...}")  # or Training/Registration per event_type
```

- [ ] **Step 3: Run CLI smoke test**

If there are CLI tests, run them. If not, exercise manually:

```bash
ario-mlflow verify trace <some_trace_id> 2>&1 | head -20
```

- [ ] **Step 4: Commit**

```bash
git add ario_mlflow/cli.py
git commit -m "Phase 3 — ario_mlflow CLI output uses new labels"
```

---

## Phase H — Frontend-design polish + manual review

### Task H1: Frontend-design polish pass

**Files:** various

The Phase 3 plan (per the original spec) calls for using the
`frontend-design` skill on the running app. The brainstorm has already
produced detailed mockups, so this pass is for **pixel-level refinement**
of any visual rough edges that surface when testing on real data with
real screen sizes.

- [ ] **Step 1: Boot demo on Railway-like resolution (1100-1400px wide window)**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
open http://localhost:8000/
```

- [ ] **Step 2: Walk every page in the demo**

For each of the six pages: dashboard, decision detail, run detail, model chain, model registry, who-this-is-for.

Note any visual issues:
- Spacing / padding inconsistencies
- Long values overflowing
- Misaligned elements
- Color contrast issues
- Mobile responsive behavior (resize to 700px)

- [ ] **Step 3: Apply targeted fixes**

For each issue, apply the smallest possible CSS / template change. Prefer adjusting existing utility classes over adding new ones.

- [ ] **Step 4: Commit polish fixes (one or several commits)**

```bash
git add -A
git commit -m "Phase 3 — frontend-design polish pass on <area>"
```

---

### Task H2: Manual review checkpoint with user

Per `feedback_per_phase_manual_review.md`, this is a blocking gate.

- [ ] **Step 1: Run full pytest**

```bash
pytest -q
```

Expected: all green.

- [ ] **Step 2: Capture screenshots / recording of every changed page**

For each page, capture both default state and any tamper-triggered states. Use `cmd+shift+5` on macOS to grab regions.

- [ ] **Step 3: Document any deviations from the spec**

If the implementation diverged from the spec for justified reasons (browser bug, MLflow API limitation, etc.), write a paragraph per deviation in a NOTES file or PR description.

- [ ] **Step 4: Hand off to user for review**

Present:
- Screenshots / recording
- The deviation list (if any)
- A note that pytest is green
- Confirmation the demo runs end-to-end

User reviews and approves before PR is opened. **Do not open the PR until explicit user approval.**

- [ ] **Step 5: Address any user-requested changes**

If the user requests changes, treat each as a small follow-up commit on the branch.

---

## Phase I — Open PR

### Task I1: Push the branch and open PR #9

**Files:** none modified

- [ ] **Step 1: Push to origin**

```bash
git push -u origin phase3/demo-ux-polish
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "Phase 3: demo UX polish — new verification labels, tamper sections, split proof viewer" --body "$(cat <<'EOF'
## Summary

Phase 3 of the redesign trilogy. Six demo templates get a wording + structure
pass. Two new tamper backend endpoints mutate real MLflow state with auto-
revert.

Tracked spec: `docs/superpowers/specs/2026-04-30-phase3-demo-ux-polish-design.md`
Tracked plan: `docs/superpowers/plans/2026-04-30-phase3-demo-ux-polish.md`

## What's in this PR

- **decision_detail.html** — Decision card (was Prediction), new 3-row
  ar.io Verification, 2x2 audit grid (Model / Inference / Trace / ar.io
  Anchor), Demonstrate Tampering collapsible, How verification works
  collapsible with split proof viewer.
- **run_detail.html** — Same patterns applied to training events. Proof
  Layer panel deleted; Live MLflow tags wrapped in collapsible.
- **model_chain.html** — Mini verify cards in chain nodes, two
  live-data tampers (training, registration).
- **index.html** — Status vocabulary sweep (Verified / Pending
  verification / Tampered / Not anchored), Result chip, table column
  renames.
- **model_registry.html** — Vocabulary sweep, clickable Run IDs, full
  (untruncated) values.
- **who_this_is_for.html** — Light wording sweep; 4-col persona grid.
- **app/tamper.py + app/main.py** — Two new POST endpoints
  (`/tamper/{saved,live}/{event_type}/{event_id}` + reset), with
  pre-tamper snapshots + 60-second auto-revert.
- **README.md, ario_mlflow/README.md, ario_mlflow/cli.py** — Vocabulary
  sweep so docs and CLI output match the new labels.

## What's NOT in this PR (deferred)

- Trusted-issuer-key check unlocking a third tamper. Captured in
  `ROADMAP.md` under External identity binding as a fast-follow.
- Persona cards for P4 / P6 on who_this_is_for.html.

## Trust model after this PR

Unchanged from Phase 2. The plugin's contract (`source_of_truth_ok`,
`attestation_level`, etc.) is identical; Phase 3 only changes how the UI
labels and presents that data.

## Test plan

- [ ] `pytest` passes (verify count vs main baseline).
- [ ] New `tests/test_tamper_endpoints.py` passes.
- [ ] Manual walkthrough completed: train → predict → verify → tamper
  → verify → reset → verify on each tamperable page.
- [ ] All six page redesigns render correctly on default Railway resolution.
- [ ] No visible regressions on mobile (≤700px) for the same pages.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Capture PR URL and confirm CI is queued**

```bash
gh pr view --json url,statusCheckRollup
```

Expected: PR URL printed. CodeRabbit (and any GitHub Actions) starts queuing.

---

## Self-review (run before declaring plan ready)

This is a checklist run by the plan author, not the implementer.

**Spec coverage check:**
- ✅ § 3 vocabulary changes → covered in Phase B (per template) + Phase G (docs)
- ✅ § 4.1 decision_detail design → Phase D (D1-D7)
- ✅ § 4.2 index → Task B1
- ✅ § 4.3 run_detail → Phase E
- ✅ § 4.4 model_chain → Phase F
- ✅ § 4.5 model_registry → Task B2
- ✅ § 4.6 who_this_is_for → Task B3
- ✅ § 5 tamper backend → Phase C
- ✅ § 6 CSS additions → Task A2
- ✅ § 7 discoverability fixes → Tasks B2 (registry), D4 (decision audit Model card)
- ✅ § 8 files removed → covered in respective restructure tasks (D4, E1)
- ✅ § 9 validation → Phase H, Task I1
- ✅ § 10 out of scope → already in ROADMAP.md
- ✅ § 11 cascading docs → Phase G
- ✅ § 12 implementation order → followed

**Placeholder scan:** searched for `TBD`, `TODO`, "implement later", "fill
in details", "similar to" — none found. Code blocks given for every code
step.

**Type / signature consistency:** verified `tamper_saved`, `tamper_live`,
`reset` signatures match across `app/tamper.py`, `app/main.py` route
handlers, and tests. `_resolve_run_id`, `_payload_artifact_path` helper
signatures consistent. Front-end `data-event-type` and `data-event-id`
attributes consistent across decision/run/chain templates.

**Scope check:** This is one PR — six related templates + one cohesive
backend feature (tamper), all sharing vocabulary and patterns. No need to
decompose further.
